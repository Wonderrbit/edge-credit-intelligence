from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

from mlops.config import resolve_paths

REQUIRED_CLIENT_COLUMNS = {
    "client_id",
    "renda_mensal",
    "limite_atual",
    "p_inadimplente_90d",
    "p_pagou_30d",
    "valor_atrasado_cutoff",
}


class SourceAdapter(Protocol):
    name: str

    def fetch(self) -> pd.DataFrame: ...


@dataclass
class ParquetSource:
    path: Path
    name: str = "parquet"

    def fetch(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Fonte ausente: {self.path}")
        return pd.read_parquet(self.path)


@dataclass
class DuckDBSource:
    database: Path
    query: str
    name: str = "duckdb"

    def fetch(self) -> pd.DataFrame:
        if not self.database.exists():
            raise FileNotFoundError(f"Banco ausente: {self.database}")
        with duckdb.connect(str(self.database), read_only=True) as connection:
            return connection.execute(self.query).df()


@dataclass
class OpenFinanceSource:
    path: Path
    name: str = "open_finance"

    def fetch(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(
                columns=[
                    "client_id",
                    "mes",
                    "saldo_open_finance",
                    "entradas_open_finance",
                    "saidas_open_finance",
                ]
            )
        return pd.read_parquet(self.path)


@dataclass
class BureauSource:
    path: Path
    name: str = "bureau"

    def fetch(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["client_id", "bureau_score"])
        return pd.read_parquet(self.path)


class DataQualityCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class DataQualityReport(BaseModel):
    passed: bool
    rows: int
    columns: int
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    checks: list[DataQualityCheck] = Field(default_factory=list)


def default_clients_source() -> ParquetSource:
    paths = resolve_paths()
    return ParquetSource(paths.serving / "serving_clients.parquet", name="serving_clients")


def default_raw_clients_source() -> ParquetSource:
    paths = resolve_paths()
    return ParquetSource(paths.data_raw / "clientes.parquet", name="raw_clients")


def default_collections_source() -> ParquetSource:
    paths = resolve_paths()
    return ParquetSource(paths.data_raw / "cobrancas.parquet", name="collections")


def default_open_finance_source() -> OpenFinanceSource:
    paths = resolve_paths()
    return OpenFinanceSource(paths.data_raw / "open_finance.parquet")


def validate_client_frame(frame: pd.DataFrame) -> DataQualityReport:
    checks: list[DataQualityCheck] = []
    missing = sorted(REQUIRED_CLIENT_COLUMNS - set(frame.columns))
    checks.append(
        DataQualityCheck(
            name="schema",
            passed=not missing,
            detail="ok" if not missing else f"colunas ausentes: {missing}",
        )
    )
    duplicate_ids = int(frame["client_id"].duplicated().sum()) if "client_id" in frame else 0
    checks.append(
        DataQualityCheck(
            name="client_id_unico",
            passed=duplicate_ids == 0,
            detail=f"{duplicate_ids} duplicados",
        )
    )
    null_report = {
        column: round(float(frame[column].isna().mean()), 4)
        for column in REQUIRED_CLIENT_COLUMNS
        if column in frame
    }
    nulls_ok = all(value <= 0.02 for value in null_report.values())
    checks.append(
        DataQualityCheck(
            name="nulos",
            passed=nulls_ok,
            detail=str(null_report),
        )
    )
    range_checks = {
        "p_inadimplente_90d": (0.0, 1.0),
        "p_pagou_30d": (0.0, 1.0),
        "renda_mensal": (0.0, float("inf")),
        "limite_atual": (0.0, float("inf")),
    }
    outside: dict[str, int] = {}
    for column, (low, high) in range_checks.items():
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = int(((values < low) | (values > high)).sum())
        if invalid:
            outside[column] = invalid
    checks.append(
        DataQualityCheck(
            name="faixas",
            passed=not outside,
            detail="ok" if not outside else str(outside),
        )
    )
    return DataQualityReport(
        passed=all(check.passed for check in checks),
        rows=int(len(frame)),
        columns=int(len(frame.columns)),
        checks=checks,
    )
