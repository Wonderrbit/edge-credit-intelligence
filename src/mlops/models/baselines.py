from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from mlops.data.schema import TARGETS
from mlops.evaluation.metrics import binary_metrics


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        remainder="drop",
    )


def build_estimator(model_type: str, random_state: int = 42) -> Any:
    if model_type == "logistic":
        return LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_type == "lightgbm":
        return LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    raise ValueError(f"Modelo desconhecido: {model_type}")


def fit_baseline(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    model_type: str,
    random_state: int = 42,
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    estimator = build_estimator(model_type, random_state=random_state)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(
        train[numeric_features + categorical_features],
        train[target].astype(int),
    )
    return pipeline


def predict_positive(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(frame)[:, 1]


def feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def tree_importance(pipeline: Pipeline, top_n: int = 20) -> list[dict[str, float]]:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return []
    names = feature_names(pipeline)
    values = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(-values)[:top_n]
    return [
        {"feature": names[index], "importance": float(values[index])}
        for index in order
    ]


def train_baselines(
    tabular: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    model_types: tuple[str, ...] = ("logistic", "lightgbm", "xgboost"),
    random_state: int = 42,
    output_dir: str | Path = "artifacts/models",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = tabular[tabular["safra"].isin((1, 2))]
    validation = tabular[tabular["safra"].isin((3, 4))]
    test = tabular[tabular["safra"].isin((5, 6))]

    results: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for target in TARGETS:
        results[target] = {}
        predictions[target] = {}
        for model_type in model_types:
            pipeline = fit_baseline(
                train,
                validation,
                target,
                numeric_features,
                categorical_features,
                model_type,
                random_state=random_state,
            )
            test_prob = predict_positive(pipeline, test)
            predictions[target][model_type] = test_prob
            metrics = binary_metrics(test[target].to_numpy(), test_prob, threshold=0.5)
            metrics["model"] = model_type
            metrics["target"] = target
            results[target][model_type] = metrics
            joblib.dump(pipeline, output_dir / f"baseline_{model_type}_{target}.joblib")
            if hasattr(pipeline.named_steps["model"], "feature_importances_"):
                (output_dir / f"importance_{model_type}_{target}.json").write_text(
                    json.dumps(tree_importance(pipeline), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "results": results,
        "predictions": predictions,
        "train": train,
        "validation": validation,
        "test": test,
    }
