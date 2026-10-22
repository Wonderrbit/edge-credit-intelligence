from __future__ import annotations

from typing import Any

import pandas as pd

from mlops.evaluation.metrics import binary_metrics
from mlops.models.baselines import fit_baseline, predict_positive


def walk_forward_evaluate(
    tabular: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target: str = "inadimplente_90d",
    model_type: str = "lightgbm",
    min_train_safras: int = 2,
) -> list[dict[str, float]]:
    safras = sorted(tabular["safra"].unique().tolist())
    results: list[dict[str, float]] = []
    for index, safra in enumerate(safras):
        if index < min_train_safras:
            continue
        train = tabular[tabular["safra"] < safra]
        test = tabular[tabular["safra"] == safra]
        if train.empty or test.empty or test[target].nunique() < 2:
            continue
        pipeline = fit_baseline(
            train,
            test,
            target,
            numeric_features,
            categorical_features,
            model_type,
        )
        probabilities = predict_positive(pipeline, test)
        metrics = binary_metrics(test[target].to_numpy(), probabilities)
        metrics["safra"] = int(safra)
        metrics["train_n"] = int(len(train))
        metrics["test_n"] = int(len(test))
        results.append(metrics)
    return results


def backtest_policy(
    frame: pd.DataFrame,
    policy_function: Any,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    results = []
    for safra, subset in frame.groupby("safra"):
        result = policy_function(subset, **kwargs)
        result["safra"] = int(safra)
        result["n"] = int(len(subset))
        results.append(result)
    return results
