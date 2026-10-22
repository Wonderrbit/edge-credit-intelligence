from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from mlops.config import ensure_dirs, load_config, resolve_paths, set_seed
from mlops.data.features import build_datasets
from mlops.data.lineage import sha256_file
from mlops.data.quality import data_quality_report
from mlops.data.schema import TARGETS
from mlops.data.synthetic import generate_synthetic_data
from mlops.evaluation.calibration import calibrate_heads, save_calibration
from mlops.evaluation.explainability import (
    gradient_x_input,
    per_client_drivers,
    permutation_importance,
    tree_shap_importance,
)
from mlops.evaluation.fairness import fairness_summary
from mlops.evaluation.metrics import binary_metrics
from mlops.evaluation.plots import (
    plot_calibration,
    plot_importance,
    plot_lift,
    plot_score_distribution,
)
from mlops.evaluation.policy import business_report, load_policies
from mlops.evaluation.stability import feature_psi, score_stability
from mlops.evaluation.stress import stress_test
from mlops.evaluation.train_serve_skew import check_train_serve_skew
from mlops.models.baselines import train_baselines
from mlops.models.collections_uplift import train_uplift
from mlops.models.edge_model import train_edge_model


def _targets_for_split(dataset: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    mask = dataset["split"] == split
    return {target: dataset["y"][target][mask] for target in TARGETS}


def _write_json(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _load_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(raw_dir / f"{name}.parquet")
        for name in ("clientes", "painel", "targets", "tratamentos")
    }


def _maybe_mlflow(
    enabled: bool,
    config: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[Path],
) -> None:
    if not enabled:
        return
    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config.get("mlflow", {}).get("experiment_name", "edge-credit-intelligence"))
        with mlflow.start_run(run_name=f"edge-credit-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"):
            mlflow.log_params(
                {
                    "n_clients": config["data"]["n_clients"],
                    "months": config["data"]["months"],
                    "seed": config["data"]["seed"],
                    "epochs": config["model"]["edge"]["epochs"],
                }
            )
            for target in TARGETS:
                mlflow.log_metric(f"edge_auc_{target}", float(metrics["edge_calibrated"][target].get("roc_auc", 0.0)))
                mlflow.log_metric(f"edge_ks_{target}", float(metrics["edge_calibrated"][target].get("ks", 0.0)))
            for artifact in artifacts:
                if artifact.exists():
                    mlflow.log_artifact(str(artifact))
    except Exception as exc:  # pragma: no cover - optional integration
        print(f"[mlflow] ignorado: {exc}")


def run_training(
    n_clients: int | None = None,
    months: int | None = None,
    seed: int | None = None,
    fast: bool = False,
    force_data: bool = False,
    enable_mlflow: bool | None = None,
) -> dict[str, Any]:
    config = load_config()
    paths = ensure_dirs(resolve_paths(config))
    data_config = config["data"]
    if n_clients is None:
        n_clients = 800 if fast else int(data_config["n_clients"])
    if months is None:
        months = int(data_config["months"])
    if seed is None:
        seed = int(data_config["seed"])
    set_seed(seed)

    raw_files = [paths.data_raw / f"{name}.parquet" for name in ("clientes", "painel", "targets", "tratamentos")]
    if force_data or not all(path.exists() for path in raw_files):
        generate_synthetic_data(
            output_dir=paths.data_raw,
            n_clients=n_clients,
            months=months,
            seed=seed,
            feature_months=int(data_config["feature_months"]),
            performance_months=int(data_config["performance_months"]),
            safras=tuple(data_config["safras"]),
        )

    raw_tables = _load_raw_tables(paths.data_raw)
    quality = data_quality_report(raw_tables, paths.eval / "data_quality.json")

    datasets = build_datasets(
        raw_dir=paths.data_raw,
        output_dir=paths.data_processed,
        train_safras=tuple(data_config["train_safras"]),
        validation_safras=tuple(data_config["validation_safras"]),
        test_safras=tuple(data_config["test_safras"]),
    )
    tabular = datasets["tabular"]
    numeric_features = datasets["feature_columns"]
    categorical_features = datasets["categorical_columns"]

    model_types = ("logistic", "lightgbm") if fast else ("logistic", "lightgbm", "xgboost")
    baseline = train_baselines(
        tabular=tabular,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_types=model_types,
        random_state=int(config["model"]["random_state"]),
        output_dir=paths.models,
    )

    edge = train_edge_model(
        dataset={
            "X_seq": datasets["seq"],
            "X_static": datasets["static"],
            "client_id": datasets["client_ids"],
            "split": datasets["split"],
            "y": datasets["y"],
        },
        config=config["model"],
        output_dir=paths.models,
        epochs=4 if fast else None,
    )

    y_validation = _targets_for_split(
        {"split": datasets["split"], "y": datasets["y"]}, "validation"
    )
    y_test = _targets_for_split(
        {"split": datasets["split"], "y": datasets["y"]}, "test"
    )
    scalers, calibration_report, calibrated_test = calibrate_heads(
        y_val=y_validation,
        probabilities_val=edge["val_probs"],
        y_test=y_test,
        probabilities_test=edge["test_probs"],
    )
    save_calibration(scalers, paths.models / "calibration.json")

    edge_calibrated_metrics = {
        target: binary_metrics(y_test[target], calibrated_test[target]) for target in TARGETS
    }
    edge_raw_metrics = {
        target: binary_metrics(y_test[target], edge["test_probs"][target]) for target in TARGETS
    }

    test_frame = baseline["test"].copy()
    test_frame = test_frame.merge(
        datasets["targets"][
            [
                "client_id",
                "valor_atrasado_cutoff",
                "consumo_30d",
                "consumo_performance_medio",
                "elegivel_limite",
            ]
        ],
        on="client_id",
        how="left",
        validate="one_to_one",
    )
    test_mask = datasets["split"] == "test"
    edge_test_frame = pd.DataFrame(
        {
            "client_id": datasets["client_ids"][test_mask],
            **{f"p_{target}": calibrated_test[target] for target in TARGETS},
            **{f"p_raw_{target}": edge["test_probs"][target] for target in TARGETS},
        }
    )
    test_frame = test_frame.merge(edge_test_frame, on="client_id", how="left", validate="one_to_one")
    for target in TARGETS:
        for model_type in model_types:
            predictions = baseline["predictions"][target][model_type]
            test_frame[f"p_baseline_{model_type}_{target}"] = predictions

    policies = load_policies()
    business = business_report(test_frame, policies, preset="equilibrado")
    fairness = fairness_summary(test_frame)
    reference = tabular[tabular["safra"].isin(tuple(data_config["train_safras"]))]
    stability = {
        "features": feature_psi(
            reference,
            test_frame,
            [column for column in numeric_features if column in test_frame.columns],
        ).to_dict(orient="records"),
        "score": score_stability(
            edge["val_probs"]["inadimplente_90d"],
            calibrated_test["inadimplente_90d"],
        ),
    }
    skew = check_train_serve_skew(
        reference,
        test_frame,
        [column for column in numeric_features if column in test_frame.columns],
    )

    uplift = train_uplift(
        tabular=tabular,
        treatments=raw_tables["tratamentos"],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        output_dir=paths.models,
    )

    try:
        lightgbm_pipeline = joblib.load(paths.models / "baseline_lightgbm_inadimplente_90d.joblib")
    except Exception:
        lightgbm_pipeline = joblib.load(paths.models / "baseline_logistic_inadimplente_90d.joblib")

    shap_importance = tree_shap_importance(lightgbm_pipeline, test_frame)
    permutation = permutation_importance(
        lightgbm_pipeline,
        test_frame,
        "inadimplente_90d",
        numeric_features,
        categorical_features,
    )
    edge_seq_importance, edge_static_importance = gradient_x_input(
        edge["model"],
        datasets["seq"][test_mask],
        datasets["static"][test_mask],
        datasets["schema"],
        target="inadimplente_90d",
    )
    drivers = per_client_drivers(
        edge["model"],
        datasets["seq"][test_mask],
        datasets["static"][test_mask],
        datasets["schema"],
        datasets["client_ids"][test_mask],
        target="inadimplente_90d",
        top_n=3,
    )

    stress = stress_test(
        lightgbm_pipeline,
        test_frame,
        calibrated_test["inadimplente_90d"],
    )

    plot_calibration(
        y_test["inadimplente_90d"],
        edge["test_probs"]["inadimplente_90d"],
        calibrated_test["inadimplente_90d"],
        "inadimplente_90d",
        paths.eval / "calibration_inadimplente_90d.png",
    )
    plot_lift(
        y_test["inadimplente_90d"],
        calibrated_test["inadimplente_90d"],
        "inadimplente_90d",
        paths.eval / "lift_inadimplente_90d.png",
    )
    plot_score_distribution(
        edge["val_probs"]["inadimplente_90d"],
        calibrated_test["inadimplente_90d"],
        "inadimplente_90d",
        paths.eval / "score_distribution_inadimplente_90d.png",
    )
    plot_importance(
        shap_importance or permutation,
        "Importância — inadimplente_90d",
        paths.eval / "importance_inadimplente_90d.png",
    )

    metrics = {
        "baseline": baseline["results"],
        "edge_raw": edge_raw_metrics,
        "edge_calibrated": edge_calibrated_metrics,
        "calibration": calibration_report,
        "business": business,
        "fairness": fairness,
        "stability": stability,
        "train_serve_skew": skew,
        "uplift": uplift["metrics"],
        "stress": stress,
        "data_quality": quality,
        "explainability": {
            "tree_shap": shap_importance,
            "permutation": permutation,
            "edge_sequence": edge_seq_importance,
            "edge_static": edge_static_importance,
        },
    }
    _write_json(metrics, paths.eval / "metrics.json")
    _write_json(
        {
            "calibration": calibration_report,
            "edge_metrics": edge_calibrated_metrics,
            "baseline_metrics": baseline["results"],
        },
        paths.eval / "model_metrics.json",
    )

    serving = test_frame.copy()
    serving.to_parquet(paths.serving / "serving_clients.parquet", index=False)
    np.savez_compressed(
        paths.serving / "serving_sequences.npz",
        X_seq=datasets["seq"][test_mask],
        X_static=datasets["static"][test_mask],
        client_id=datasets["client_ids"][test_mask],
        safra=datasets["safra"][test_mask],
        **{f"p_{target}": calibrated_test[target] for target in TARGETS},
    )
    _write_json(drivers, paths.serving / "client_drivers.json")

    schema = json.loads((paths.data_processed / "feature_schema.json").read_text(encoding="utf-8"))
    _write_json(schema, paths.edge / "preprocessing.json")
    sample_index = 0
    _write_json(
        {
            "client_id": str(datasets["client_ids"][test_mask][sample_index]),
            "sequence": datasets["seq"][test_mask][sample_index].round(6).tolist(),
            "static": datasets["static"][test_mask][sample_index].round(6).tolist(),
            "expected": {
                target: float(edge["test_probs"][target][sample_index]) for target in TARGETS
            },
        },
        paths.edge / "example_input.json",
    )

    reference_payload = {
        "score_mean": float(np.mean(calibrated_test["inadimplente_90d"])),
        "score_std": float(np.std(calibrated_test["inadimplente_90d"])),
        "features": {
            column: {
                "mean": float(reference[column].mean()),
                "std": float(reference[column].std()),
            }
            for column in numeric_features
            if column in reference.columns
        },
    }
    _write_json(reference_payload, paths.eval / "reference_distributions.json")

    model_path = paths.models / "edge_credit_multitask.keras"
    try:
        model_path_label = model_path.relative_to(paths.project_root).as_posix()
    except ValueError:
        model_path_label = model_path.as_posix()
    registry = {
        "model_name": "edge_credit_multitask",
        "version": datetime.now(UTC).strftime("%Y.%m.%d.%H%M"),
        "created_at": datetime.now(UTC).isoformat(),
        "n_clients": int(len(tabular)),
        "n_features_sequence": int(datasets["seq"].shape[2]),
        "n_features_static": int(datasets["static"].shape[1]),
        "n_parameters": int(edge["model"].count_params()),
        "model_path": model_path_label,
        "model_sha256": sha256_file(model_path) if model_path.exists() else None,
        "calibration": calibration_report,
        "edge_metrics": edge_calibrated_metrics,
        "thresholds": config["thresholds"],
        "split": {
            "train_safras": data_config["train_safras"],
            "validation_safras": data_config["validation_safras"],
            "test_safras": data_config["test_safras"],
        },
        "deployment": {
            "mode": "edge_tfjs_and_api",
            "privacy": "dados sensíveis podem ser pontuados no dispositivo",
            "latency_target_ms": 15,
        },
    }
    _write_json(registry, paths.artifacts / "registry.json")

    if enable_mlflow is None:
        enable_mlflow = os.getenv("MLOPS_ENABLE_MLFLOW", "0") == "1"
    _maybe_mlflow(
        enabled=enable_mlflow,
        config=config,
        metrics=metrics,
        artifacts=[
            paths.eval / "metrics.json",
            paths.eval / "model_metrics.json",
            paths.models / "calibration.json",
            paths.eval / "lift_inadimplente_90d.png",
        ],
    )

    return {
        "config": config,
        "paths": paths,
        "datasets": datasets,
        "baseline": baseline,
        "edge": edge,
        "metrics": metrics,
        "test_frame": test_frame,
        "business": business,
        "fairness": fairness,
        "registry": registry,
    }
