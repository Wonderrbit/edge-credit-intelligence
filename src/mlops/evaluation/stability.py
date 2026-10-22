from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlops.data.quality import psi


def feature_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    bins: int = 10,
    threshold: float = 0.20,
) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in reference or column not in current:
            continue
        value = psi(reference[column].to_numpy(), current[column].to_numpy(), bins=bins)
        rows.append(
            {
                "feature": column,
                "psi": value,
                "status": "critico" if value >= threshold else ("atencao" if value >= 0.10 else "ok"),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def score_stability(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    threshold: float = 0.10,
) -> dict[str, Any]:
    value = psi(reference_scores, current_scores)
    return {
        "psi_score": value,
        "status": "critico" if value >= 0.20 else ("atencao" if value >= threshold else "ok"),
        "reference_mean": float(np.mean(reference_scores)),
        "current_mean": float(np.mean(current_scores)),
    }


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: list[str],
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
) -> dict[str, Any]:
    features = feature_psi(reference, current, feature_columns)
    scores = score_stability(reference_scores, current_scores)
    return {
        "features": features.to_dict(orient="records"),
        "score": scores,
        "passed": bool(features["psi"].max() < 0.20) if len(features) else True,
    }
