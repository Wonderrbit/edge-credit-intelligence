from __future__ import annotations

from typing import Iterable

import pandas as pd

from mlops.data.schema import REQUIRED_RAW_TABLES


class DataContractError(ValueError):
    """Raised when a dataset violates its declared contract."""


def validate_columns(frame: pd.DataFrame, required: Iterable[str], table: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataContractError(f"{table}: colunas obrigatórias ausentes: {missing}")


def validate_raw_tables(tables: dict[str, pd.DataFrame]) -> None:
    for table, columns in REQUIRED_RAW_TABLES.items():
        if table not in tables:
            raise DataContractError(f"Tabela obrigatória ausente: {table}")
        validate_columns(tables[table], columns, table)

    clients = tables["clientes"]
    if clients["client_id"].duplicated().any():
        raise DataContractError("clientes: client_id duplicado")
    if (clients["renda_mensal"] <= 0).any():
        raise DataContractError("clientes: renda_mensal deve ser positiva")
    if (clients["limite_atual"] < 0).any():
        raise DataContractError("clientes: limite_atual não pode ser negativo")

    targets = tables["targets"]
    for target in ("pagou_30d", "inadimplente_90d", "aceita_limite", "consumo_alto"):
        if not set(targets[target].dropna().unique()).issubset({0, 1}):
            raise DataContractError(f"targets: {target} deve ser binário")
    rates = {target: float(targets[target].mean()) for target in (
        "pagou_30d",
        "inadimplente_90d",
        "aceita_limite",
        "consumo_alto",
    )}
    for target, rate in rates.items():
        if not 0.01 <= rate <= 0.98:
            raise DataContractError(f"targets: taxa de {target} implausível: {rate:.3f}")

    panel = tables["painel"]
    if panel[["client_id", "mes"]].duplicated().any():
        raise DataContractError("painel: chave client_id + mes duplicada")
    if (panel["mes"] < 1).any():
        raise DataContractError("painel: mes inválido")
