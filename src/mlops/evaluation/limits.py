from __future__ import annotations

from typing import Any


def _linear_score(
    p_accept: float,
    p_consumo: float,
    p_default: float,
    preset: dict[str, float],
) -> float:
    risk_weight = float(preset.get("risk_weight", 0.25))
    acceptance_weight = float(preset.get("acceptance_weight", 0.35))
    consumption_weight = float(preset.get("consumption_weight", 0.25))
    raw = (
        acceptance_weight * p_accept
        + consumption_weight * p_consumo
        - risk_weight * p_default
    )
    return max(0.0, min(1.0, raw + 0.35))


def recommend_limit(
    income: float,
    current_limit: float,
    p_accept: float,
    p_consumo: float,
    p_default: float,
    avg_spend: float,
    preset: dict[str, float],
    margin: float = 0.035,
    lgd: float = 0.60,
    usage_rate: float = 0.55,
) -> dict[str, Any]:
    income = max(float(income), 1.0)
    current_limit = max(float(current_limit), 0.0)
    max_multiple = float(preset.get("max_multiple_income", 2.0))
    max_growth = float(preset.get("max_growth", 1.6))
    score = _linear_score(p_accept, p_consumo, p_default, preset)
    growth = 1.0 + (max_growth - 1.0) * score
    raw_limit = current_limit * growth
    ceiling = income * max_multiple
    recommended = max(current_limit, min(raw_limit, ceiling))
    increase = recommended - current_limit
    expected_utilization = max(0.0, increase * usage_rate * (0.35 + 0.65 * p_accept))
    expected_loss = expected_utilization * p_default * lgd
    expected_profit = expected_utilization * margin - expected_loss
    return {
        "limite_atual": round(current_limit, 2),
        "limite_recomendado": round(recommended, 2),
        "aumento": round(increase, 2),
        "growth": round(recommended / max(current_limit, 1.0), 3),
        "score_politica": round(score, 4),
        "utilizacao_esperada": round(expected_utilization, 2),
        "perda_esperada": round(expected_loss, 2),
        "lucro_esperado": round(expected_profit, 2),
        "consumo_medio_3m": round(float(avg_spend), 2),
        "preset": preset,
    }


def simulate_limit_policy(
    frame: Any,
    preset: dict[str, float],
    margin: float = 0.035,
    lgd: float = 0.60,
    usage_rate: float = 0.55,
) -> dict[str, Any]:
    rows = []
    for _, row in frame.iterrows():
        result = recommend_limit(
            income=float(row["renda_mensal"]),
            current_limit=float(row["limite_atual"]),
            p_accept=float(row.get("p_aceita_limite", 0.5)),
            p_consumo=float(row.get("p_consumo_alto", 0.5)),
            p_default=float(row.get("p_inadimplente_90d", 0.1)),
            avg_spend=float(row.get("consumo_medio_3m", 0.0)),
            preset=preset,
            margin=margin,
            lgd=lgd,
            usage_rate=usage_rate,
        )
        rows.append(result)

    import pandas as pd

    result_frame = pd.DataFrame(rows)
    return {
        "n": int(len(result_frame)),
        "limite_medio_atual": round(float(result_frame["limite_atual"].mean()), 2),
        "limite_medio_recomendado": round(float(result_frame["limite_recomendado"].mean()), 2),
        "aumento_medio": round(float(result_frame["aumento"].mean()), 2),
        "utilizacao_esperada_total": round(float(result_frame["utilizacao_esperada"].sum()), 2),
        "perda_esperada_total": round(float(result_frame["perda_esperada"].sum()), 2),
        "lucro_esperado_total": round(float(result_frame["lucro_esperado"].sum()), 2),
        "taxa_aumento": round(float((result_frame["aumento"] > 0).mean()), 4),
    }
