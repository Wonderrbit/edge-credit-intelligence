from __future__ import annotations

from typing import Any

import pandas as pd

from mlops.data.quality import psi


def check_train_serve_skew(
    reference: pd.DataFrame,
    serving: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.20,
) -> dict[str, Any]:
    rows = []
    for column in columns:
        if column not in reference or column not in serving:
            continue
        value = psi(reference[column].to_numpy(), serving[column].to_numpy())
        rows.append(
            {
                "feature": column,
                "psi": value,
                "passed": value < threshold,
            }
        )
    return {
        "features": rows,
        "max_psi": max((row["psi"] for row in rows), default=0.0),
        "passed": all(row["passed"] for row in rows),
        "threshold": threshold,
    }
