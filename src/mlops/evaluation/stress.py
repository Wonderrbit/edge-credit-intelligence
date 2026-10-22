from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlops.models.baselines import predict_positive


def stress_scenarios(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    income_shock = frame.copy()
    income_shock["renda_mensal"] = income_shock["renda_mensal"] * 0.80
    income_shock["comprometimento_consumo"] = income_shock["comprometimento_consumo"] * 1.25

    delinquency_shock = frame.copy()
    delinquency_shock["atraso_max_12m"] = delinquency_shock["atraso_max_12m"] * 1.35
    delinquency_shock["atraso_medio"] = delinquency_shock["atraso_medio"] * 1.35
    delinquency_shock["frequencia_atraso"] = (delinquency_shock["frequencia_atraso"] * 1.30).clip(0, 1)

    utilization_shock = frame.copy()
    utilization_shock["utilizacao_media"] = (utilization_shock["utilizacao_media"] * 1.25).clip(0, 1.6)
    utilization_shock["utilizacao_max"] = (utilization_shock["utilizacao_max"] * 1.25).clip(0, 2.0)

    return {
        "choque_renda_20pct": income_shock,
        "choque_atraso_35pct": delinquency_shock,
        "choque_utilizacao_25pct": utilization_shock,
    }


def stress_test(
    pipeline: Any,
    frame: pd.DataFrame,
    reference_probabilities: np.ndarray,
) -> dict[str, Any]:
    baseline = {
        "mean_score": float(np.mean(reference_probabilities)),
        "high_risk_rate": float(np.mean(reference_probabilities >= 0.18)),
    }
    results: dict[str, Any] = {"baseline": baseline, "scenarios": {}}
    for name, scenario in stress_scenarios(frame).items():
        probabilities = predict_positive(pipeline, scenario)
        results["scenarios"][name] = {
            "mean_score": float(np.mean(probabilities)),
            "delta_mean_score": float(np.mean(probabilities) - np.mean(reference_probabilities)),
            "high_risk_rate": float(np.mean(probabilities >= 0.18)),
            "delta_high_risk_rate": float(
                np.mean(probabilities >= 0.18) - np.mean(reference_probabilities >= 0.18)
            ),
        }
    return results
