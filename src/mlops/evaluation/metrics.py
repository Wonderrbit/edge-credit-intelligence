from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from mlops.data.quality import psi


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"n": float(len(y_true)), "positive_rate": float(y_true.mean())}
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        "n": float(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ks": ks_statistic(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "precision": float(((y_pred == 1) & (y_true == 1)).sum() / max((y_pred == 1).sum(), 1)),
        "recall": float(((y_pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)),
        "f1": float(
            2
            * (((y_pred == 1) & (y_true == 1)).sum() / max((y_pred == 1).sum(), 1))
            * (((y_pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))
            / max(
                (((y_pred == 1) & (y_true == 1)).sum() / max((y_pred == 1).sum(), 1))
                + (((y_pred == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)),
                1e-9,
            )
        ),
        "threshold": float(threshold),
        "precision_curve_auc": float(np.trapezoid(precision, recall)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    return {
        "n": float(len(y_true)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / max(np.sum((y_true - y_true.mean()) ** 2), 1e-9)),
    }


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0:
        return 0.0
    n = max(int(len(y_true) * k), 1)
    order = np.argsort(-y_prob)[:n]
    base = max(y_true.mean(), 1e-9)
    return float(y_true[order].mean() / base)


def decile_table(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    frame["decile"] = pd.qcut(frame["p"].rank(method="first", ascending=False), 10, labels=False) + 1
    grouped = frame.groupby("decile", as_index=False).agg(
        n=("y", "size"), response=("y", "mean"), avg_score=("p", "mean")
    )
    grouped["lift"] = grouped["response"] / max(frame["y"].mean(), 1e-9)
    grouped["capture"] = grouped["response"] * grouped["n"] / max(frame["y"].sum(), 1)
    return grouped.sort_values("decile")


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        mask = (y_prob >= lower) & (y_prob < upper)
        if not mask.any():
            continue
        total += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(total)


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        mask = (y_prob >= lower) & (y_prob < upper)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "n": int(mask.sum()),
                "mean_predicted": float(y_prob[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def score_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    return psi(reference, current, bins=bins)


def summarize_metrics(metrics: dict[str, float]) -> str:
    keys = ("roc_auc", "pr_auc", "ks", "brier", "precision", "recall", "lift@10")
    parts = [f"{key}={metrics[key]:.3f}" for key in keys if key in metrics]
    return " | ".join(parts)
