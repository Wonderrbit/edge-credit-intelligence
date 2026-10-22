from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops.data.schema import TARGETS


def _range_check(frame: pd.DataFrame, column: str, low: float, high: float) -> dict[str, Any]:
    series = pd.to_numeric(frame[column], errors="coerce")
    invalid = int(((series < low) | (series > high)).sum())
    return {
        "column": column,
        "expected_range": [low, high],
        "invalid_rows": invalid,
        "null_rows": int(series.isna().sum()),
        "passed": invalid == 0,
    }


def data_quality_report(
    tables: dict[str, pd.DataFrame],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    clients = tables["clientes"]
    panel = tables["painel"]
    targets = tables["targets"]

    checks: list[dict[str, Any]] = [
        _range_check(clients, "idade", 18, 100),
        _range_check(clients, "renda_mensal", 1, 1_000_000),
        _range_check(clients, "limite_atual", 0, 1_000_000),
        _range_check(clients, "bureau_score", 0, 1000),
        _range_check(panel, "atraso_dias", 0, 365),
        _range_check(panel, "utilizacao_limite", 0, 3),
        _range_check(panel, "pct_credito", 0, 1),
    ]

    completeness = {
        table: float(1.0 - frame.isna().mean().mean())
        for table, frame in tables.items()
        if len(frame.columns) > 0
    }
    target_rates = {target: float(targets[target].mean()) for target in TARGETS}
    duplicate_keys = int(panel.duplicated(subset=["client_id", "mes"]).sum())
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": {name: int(len(frame)) for name, frame in tables.items()},
        "columns": {name: int(len(frame.columns)) for name, frame in tables.items()},
        "completeness": completeness,
        "target_rates": target_rates,
        "duplicate_panel_keys": duplicate_keys,
        "checks": checks,
        "passed": all(check["passed"] for check in checks)
        and duplicate_keys == 0
        and all(rate > 0 for rate in target_rates.values()),
    }
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(report).to_json(destination, force_ascii=False, indent=2, default_handler=str)
    return report


def assert_no_target_leakage(
    feature_columns: list[str],
    target_columns: list[str] | None = None,
) -> None:
    forbidden = set(target_columns or TARGETS) | {
        "consumo_30d",
        "valor_atrasado_cutoff",
        "elegivel_limite",
        "p_accept",
        "p_pay_simulado",
    }
    leaked = sorted(set(feature_columns) & forbidden)
    if leaked:
        raise ValueError(f"Vazamento de alvo detectado nas features: {leaked}")


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_counts = np.histogram(expected, bins=edges)[0]
    actual_counts = np.histogram(actual, bins=edges)[0]
    expected_pct = np.clip(expected_counts / max(len(expected), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(len(actual), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
