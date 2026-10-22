from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from mlops.config import PROJECT_ROOT


def load_indicator_registry(path: str | Path | None = None) -> list[dict[str, Any]]:
    destination = Path(path) if path else PROJECT_ROOT / "configs" / "indicators.yaml"
    payload = yaml.safe_load(destination.read_text(encoding="utf-8")) or {}
    return list(payload.get("indicators", []))


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def compute_client_indicators(
    frame: pd.DataFrame,
    policies: dict[str, Any],
) -> pd.DataFrame:
    frame = frame.copy()
    collection = policies.get("collection", {})
    recovered_fraction = float(collection.get("recovered_fraction", 0.65))
    contact_cost = float(collection.get("contact_cost", 12.0))
    limit_config = policies.get("limit", {})
    lgd = float(limit_config.get("lgd", 0.60))

    frame["pd_90d"] = _numeric(frame, "p_inadimplente_90d")
    frame["p_pagamento_30d"] = _numeric(frame, "p_pagou_30d")
    frame["exposicao"] = _numeric(frame, "limite_atual")
    frame["perda_esperada"] = frame["pd_90d"] * frame["exposicao"] * lgd
    frame["valor_em_atraso"] = _numeric(frame, "valor_atrasado_cutoff")
    frame["ev_cobranca"] = (
        frame["p_pagamento_30d"] * frame["valor_em_atraso"] * recovered_fraction - contact_cost
    ).clip(lower=0.0)
    renda = _numeric(frame, "renda_mensal").clip(lower=1.0)
    frame["atraso_sobre_renda"] = frame["valor_em_atraso"] / renda
    frame["comprometimento_renda"] = _numeric(frame, "comprometimento_consumo")
    frame["utilizacao"] = _numeric(frame, "utilizacao_media")
    return frame


def _distribution(values: pd.Series, bins: int = 10) -> list[float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().clip(0.0, 1.0)
    if clean.empty:
        return [0.0] * bins
    counts, _ = np.histogram(clean.to_numpy(), bins=np.linspace(0.0, 1.0, bins + 1))
    total = counts.sum()
    return [round(float(count / total), 6) for count in counts] if total else [0.0] * bins


def psi_from_histograms(previous: list[float], current: list[float], epsilon: float = 1e-4) -> float:
    if not previous or not current or len(previous) != len(current):
        return 0.0
    total = 0.0
    for old, new in zip(previous, current, strict=True):
        old = max(float(old), epsilon)
        new = max(float(new), epsilon)
        total += (new - old) * np.log(new / old)
    return round(float(total), 6)


def _slice(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    if column not in frame:
        return {}
    result: dict[str, dict[str, float]] = {}
    for value, group in frame.groupby(column, dropna=False):
        result[str(value)] = {
            "n": int(len(group)),
            "pd_media": round(float(group["pd_90d"].mean()), 6),
            "limite_total": round(float(group["exposicao"].sum()), 2),
            "taxa_atraso": round(float((group["valor_em_atraso"] > 0).mean()), 6),
        }
    return result


def compute_portfolio_indicators(
    frame: pd.DataFrame,
    policies: dict[str, Any],
) -> dict[str, Any]:
    if frame.empty:
        return {"n_clientes": 0, "score_histogram": [0.0] * 10}
    approval = float(policies.get("risk", {}).get("approval_threshold", 0.18))
    exposure = frame["exposicao"].clip(lower=0.0)
    total_exposure = float(exposure.sum())
    top10 = float(exposure.nlargest(10).sum())
    return {
        "n_clientes": int(len(frame)),
        "carteira_total": round(total_exposure, 2),
        "limite_medio": round(float(exposure.mean()), 2),
        "renda_media": round(float(_numeric(frame, "renda_mensal").mean()), 2),
        "pd_media_90d": round(float(frame["pd_90d"].mean()), 6),
        "perda_esperada": round(float(frame["perda_esperada"].sum()), 2),
        "taxa_atraso": round(float((frame["valor_em_atraso"] > 0).mean()), 6),
        "valor_em_atraso": round(float(frame["valor_em_atraso"].sum()), 2),
        "utilizacao_media": round(float(frame["utilizacao"].mean()), 6),
        "comprometimento_medio": round(float(frame["comprometimento_renda"].mean()), 6),
        "taxa_aprovacao": round(float((frame["pd_90d"] < approval).mean()), 6),
        "exposicao_aprovada": round(float(exposure[frame["pd_90d"] < approval].sum()), 2),
        "ev_cobranca": round(float(frame["ev_cobranca"].sum()), 2),
        "concentracao_top10": round(top10 / total_exposure, 6) if total_exposure else 0.0,
        "score_histogram": _distribution(frame["pd_90d"]),
        "slices": {
            "segmento": _slice(frame, "segmento"),
            "regiao": _slice(frame, "regiao"),
        },
    }


def compare_indicators(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    if not previous:
        return {"psi_score": 0.0, "deltas": {}}
    psi = psi_from_histograms(previous.get("score_histogram", []), current.get("score_histogram", []))
    keys = [
        "pd_media_90d",
        "taxa_atraso",
        "taxa_aprovacao",
        "valor_em_atraso",
        "carteira_total",
    ]
    deltas = {
        key: round(float(current.get(key, 0.0)) - float(previous.get(key, 0.0)), 6)
        for key in keys
    }
    return {"psi_score": psi, "deltas": deltas}
