from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def group_report(
    frame: pd.DataFrame,
    group_column: str,
    y_true_column: str,
    score_column: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    rows = []
    for group, subset in frame.groupby(group_column, dropna=False):
        y_true = subset[y_true_column].to_numpy().astype(int)
        scores = subset[score_column].to_numpy(dtype=float)
        approved = (scores < threshold).mean() if y_true_column == "inadimplente_90d" else (scores >= threshold).mean()
        auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
        rows.append(
            {
                "group": str(group),
                "n": int(len(subset)),
                "positive_rate": float(y_true.mean()),
                "selection_rate": float(approved),
                "mean_score": float(scores.mean()),
                "auc": auc,
            }
        )
    result = pd.DataFrame(rows)
    rates = result["selection_rate"]
    disparate_impact = float(rates.min() / max(rates.max(), 1e-9)) if len(rates) else 1.0
    return {
        "group_column": group_column,
        "groups": result.to_dict(orient="records"),
        "disparate_impact_ratio": disparate_impact,
        "passed": bool(disparate_impact >= 0.80),
    }


def fairness_summary(
    frame: pd.DataFrame,
    groups: tuple[str, ...] = ("segmento", "regiao"),
) -> dict[str, Any]:
    if "p_inadimplente_90d" not in frame.columns:
        return {"passed": False, "reason": "p_inadimplente_90d ausente"}
    return {
        group: group_report(
            frame,
            group_column=group,
            y_true_column="inadimplente_90d",
            score_column="p_inadimplente_90d",
            threshold=0.5,
        )
        for group in groups
    }
