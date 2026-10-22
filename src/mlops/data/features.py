from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops.config import Paths, ensure_dirs, resolve_paths
from mlops.data.quality import assert_no_target_leakage
from mlops.data.schema import (
    CUTOFF_COL,
    ID_COL,
    SAFRA_COL,
    SEQ_FEATURES,
    STATIC_CATEGORICAL,
    STATIC_NUMERIC,
    TARGET_COLUMNS,
    TARGET_END_COL,
    TARGETS,
)

META_COLUMNS = {
    ID_COL,
    SAFRA_COL,
    CUTOFF_COL,
    TARGET_END_COL,
    "persona",
}
LEAKAGE_COLUMNS = {
    "consumo_30d",
    "consumo_performance_medio",
    "elegivel_limite",
    "valor_atrasado_cutoff",
    "p_pay_simulado",
}


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _create_views(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> None:
    for table in ("clientes", "painel", "targets", "tratamentos", "transacoes", "cobrancas", "open_finance"):
        path = raw_dir / f"{table}.parquet"
        if path.exists():
            con.execute(
                f"CREATE OR REPLACE VIEW {table} AS "
                f"SELECT * FROM read_parquet('{_sql_path(path)}')"
            )


def run_sql_features(
    raw_dir: str | Path | None = None,
    sql_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = ensure_dirs(resolve_paths())
    raw_dir = Path(raw_dir) if raw_dir else paths.data_raw
    sql_dir = Path(sql_dir) if sql_dir else paths.project_root / "sql"
    output_dir = Path(output_dir) if output_dir else paths.data_processed
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    _create_views(con, raw_dir)
    consumo = con.execute((sql_dir / "10_features_consumo.sql").read_text(encoding="utf-8")).df()
    cobranca = con.execute((sql_dir / "30_features_cobranca.sql").read_text(encoding="utf-8")).df()
    cohorts = con.execute((sql_dir / "20_cohorts_safras.sql").read_text(encoding="utf-8")).df()

    features = consumo.merge(cobranca, on=ID_COL, how="left", validate="one_to_one")
    targets = con.execute("SELECT * FROM targets").df()
    clients = con.execute("SELECT * FROM clientes").df()
    features = features.merge(
        targets,
        on=[ID_COL, SAFRA_COL, CUTOFF_COL, TARGET_END_COL],
        how="left",
        validate="one_to_one",
    )
    features = features.merge(
        clients[[ID_COL, "persona"]],
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
    con.close()

    fill_zero = [
        "recencia_compra_meses",
        "atraso_max_90d",
        "contatos_90d",
        "promessas_90d",
        "taxa_pagamento_90d",
        "meses_desde_ultimo_pagamento",
        "tendencia_consumo",
        "tendencia_utilizacao",
        "taxa_promessa",
        "taxa_pagamento_historica",
        "frequencia_atraso",
        "atraso_sobre_renda",
        "consumo_volatilidade",
        "consumo_sobre_renda",
        "comprometimento_consumo",
        "exposicao_media",
    ]
    for column in fill_zero:
        if column in features.columns:
            features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    features = features.replace([np.inf, -np.inf], 0.0)

    if features[ID_COL].duplicated().any():
        raise ValueError("features_tabular contém clientes duplicados")
    if features[list(TARGETS)].isna().any().any():
        raise ValueError("features_tabular contém targets ausentes")

    features.to_parquet(output_dir / "features_tabular.parquet", index=False)
    cohorts.to_parquet(output_dir / "cohorts.parquet", index=False)
    return features, cohorts


def _fit_edge_preprocessing(
    seq: np.ndarray,
    static_numeric: np.ndarray,
    static_categorical: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, Any]:
    n_seq = seq.shape[2]
    seq_scaler = StandardScaler().fit(seq[train_mask].reshape(-1, n_seq))
    static_scaler = StandardScaler().fit(static_numeric[train_mask])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        static_categorical[train_mask]
    )

    seq_scaled = seq_scaler.transform(seq.reshape(-1, n_seq)).reshape(seq.shape).astype(np.float32)
    static_num_scaled = static_scaler.transform(static_numeric).astype(np.float32)
    static_cat_encoded = encoder.transform(static_categorical).astype(np.float32)
    static = np.concatenate([static_num_scaled, static_cat_encoded], axis=1)

    schema = {
        "seq_features": SEQ_FEATURES,
        "static_numeric": STATIC_NUMERIC,
        "static_categorical": STATIC_CATEGORICAL,
        "seq_mean": seq_scaler.mean_.tolist(),
        "seq_scale": seq_scaler.scale_.tolist(),
        "static_mean": static_scaler.mean_.tolist(),
        "static_scale": static_scaler.scale_.tolist(),
        "categories": [list(map(str, values)) for values in encoder.categories_],
        "static_feature_names": STATIC_NUMERIC
        + [
            f"{column}={category}"
            for column, values in zip(STATIC_CATEGORICAL, encoder.categories_, strict=False)
            for category in values
        ],
        "n_sequence_features": n_seq,
        "n_static_features": int(static.shape[1]),
        "feature_window": 12,
        "performance_window": 6,
    }
    return {
        "seq": seq_scaled,
        "static": static,
        "seq_scaler": seq_scaler,
        "static_scaler": static_scaler,
        "encoder": encoder,
        "schema": schema,
    }


def build_datasets(
    raw_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    train_safras: tuple[int, ...] = (1, 2),
    validation_safras: tuple[int, ...] = (3, 4),
    test_safras: tuple[int, ...] = (5, 6),
) -> dict[str, Any]:
    paths = ensure_dirs(resolve_paths())
    raw_dir = Path(raw_dir) if raw_dir else paths.data_raw
    output_dir = Path(output_dir) if output_dir else paths.data_processed
    output_dir.mkdir(parents=True, exist_ok=True)

    features, _ = run_sql_features(raw_dir=raw_dir, output_dir=output_dir)
    panel = pd.read_parquet(raw_dir / "painel.parquet")
    clients = pd.read_parquet(raw_dir / "clientes.parquet")
    targets = pd.read_parquet(raw_dir / "targets.parquet")

    feature_columns = [
        column
        for column in features.columns
        if column not in META_COLUMNS
        and column not in TARGET_COLUMNS
        and column not in LEAKAGE_COLUMNS
        and pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical_columns = [
        column for column in STATIC_CATEGORICAL if column in features.columns
    ]
    assert_no_target_leakage(feature_columns)

    tabular = features[
        [ID_COL, SAFRA_COL, "persona", *feature_columns, *categorical_columns, *TARGETS]
    ].copy()
    tabular.to_parquet(output_dir / "tabular_dataset.parquet", index=False)
    (output_dir / "feature_columns.json").write_text(
        json.dumps(
            {
                "numeric_features": feature_columns,
                "categorical_features": categorical_columns,
                "targets": list(TARGETS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    window = panel[
        (panel["mes"] >= panel[SAFRA_COL])
        & (panel["mes"] <= panel[SAFRA_COL] + 11)
    ].sort_values([ID_COL, "mes"])
    counts = window.groupby(ID_COL).size()
    if not (counts == 12).all():
        bad = counts[counts != 12]
        raise ValueError(f"Clientes sem 12 meses de features: {bad.head().to_dict()}")

    grouped = window.groupby(ID_COL, sort=True)
    client_ids = np.array(sorted(window[ID_COL].unique()))
    seq = np.stack(
        [grouped.get_group(client_id)[SEQ_FEATURES].to_numpy(dtype=np.float32) for client_id in client_ids]
    )

    clients_indexed = clients.set_index(ID_COL).loc[client_ids]
    features_indexed = features.set_index(ID_COL).loc[client_ids]
    targets_indexed = targets.set_index(ID_COL).loc[client_ids]

    static_numeric = features_indexed[STATIC_NUMERIC].to_numpy(dtype=np.float32)
    static_categorical = clients_indexed[STATIC_CATEGORICAL].astype(str).to_numpy()
    safra = features_indexed[SAFRA_COL].to_numpy()
    split = np.where(
        np.isin(safra, train_safras),
        "train",
        np.where(np.isin(safra, validation_safras), "validation", "test"),
    )
    train_mask = split == "train"
    if not train_mask.any():
        raise ValueError("Nenhum cliente no conjunto de treino")

    preprocessed = _fit_edge_preprocessing(
        seq=seq,
        static_numeric=static_numeric,
        static_categorical=static_categorical,
        train_mask=train_mask,
    )
    y = {target: targets_indexed[target].to_numpy(dtype=np.float32) for target in TARGETS}

    np.savez_compressed(
        output_dir / "edge_dataset.npz",
        X_seq=preprocessed["seq"],
        X_static=preprocessed["static"],
        client_id=client_ids,
        safra=safra,
        split=split,
        **{f"y_{target}": values for target, values in y.items()},
    )
    joblib.dump(preprocessed["seq_scaler"], output_dir / "seq_scaler.joblib")
    joblib.dump(preprocessed["static_scaler"], output_dir / "static_scaler.joblib")
    joblib.dump(preprocessed["encoder"], output_dir / "static_encoder.joblib")
    (output_dir / "feature_schema.json").write_text(
        json.dumps(preprocessed["schema"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "features": features,
        "tabular": tabular,
        "panel": panel,
        "clients": clients,
        "targets": targets,
        "seq": preprocessed["seq"],
        "static": preprocessed["static"],
        "client_ids": client_ids,
        "safra": safra,
        "split": split,
        "y": y,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "schema": preprocessed["schema"],
    }


def load_edge_dataset(data_dir: str | Path | None = None) -> dict[str, Any]:
    paths = resolve_paths()
    data_dir = Path(data_dir) if data_dir else paths.data_processed
    payload = np.load(data_dir / "edge_dataset.npz", allow_pickle=True)
    schema = json.loads((data_dir / "feature_schema.json").read_text(encoding="utf-8"))
    return {
        "X_seq": payload["X_seq"],
        "X_static": payload["X_static"],
        "client_id": payload["client_id"].astype(str),
        "safra": payload["safra"].astype(int),
        "split": payload["split"].astype(str),
        "y": {target: payload[f"y_{target}"].astype(np.float32) for target in TARGETS},
        "schema": schema,
    }


def load_tabular_dataset(data_dir: str | Path | None = None) -> pd.DataFrame:
    paths = resolve_paths()
    data_dir = Path(data_dir) if data_dir else paths.data_processed
    return pd.read_parquet(data_dir / "tabular_dataset.parquet")
