from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mlops.config import PROJECT_ROOT
from mlops.evaluation.limits import simulate_limit_policy


def load_policies(path: str | Path | None = None) -> dict[str, Any]:
    destination = Path(path) if path else PROJECT_ROOT / "configs" / "policies.yaml"
    return yaml.safe_load(destination.read_text(encoding="utf-8"))


def collection_expected_value(
    p_pay: float,
    debt: float,
    recovered_fraction: float = 0.65,
    contact_cost: float = 12.0,
) -> float:
    return float(p_pay * max(debt, 0.0) * recovered_fraction - contact_cost)


def prioritize_collections(
    frame: pd.DataFrame,
    top_fraction: float = 0.20,
    recovered_fraction: float = 0.65,
    contact_cost: float = 12.0,
    score_column: str = "p_pagou_30d",
    debt_column: str = "valor_atrasado_cutoff",
) -> pd.DataFrame:
    scored = frame.copy()
    scored["ev_cobranca"] = [
        collection_expected_value(
            p_pay=float(row[score_column]),
            debt=float(row[debt_column]),
            recovered_fraction=recovered_fraction,
            contact_cost=contact_cost,
        )
        for _, row in scored.iterrows()
    ]
    scored = scored.sort_values("ev_cobranca", ascending=False)
    n_contact = max(int(len(scored) * top_fraction), 1)
    scored["prioridade"] = 0
    scored.iloc[:n_contact, scored.columns.get_loc("prioridade")] = 1
    return scored


def simulate_collections(
    frame: pd.DataFrame,
    policies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policies = policies or load_policies()
    config = policies["collection"]
    scored = prioritize_collections(
        frame,
        top_fraction=float(config["top_fraction"]),
        recovered_fraction=float(config["recovered_fraction"]),
        contact_cost=float(config["contact_cost"]),
    )
    base_recovery = float(
        (scored["p_pagou_30d"] * scored["valor_atrasado_cutoff"] * config["recovered_fraction"]).sum()
    )
    base_cost = float(len(scored) * config["contact_cost"])
    selected = scored[scored["prioridade"] == 1]
    model_recovery = float(
        (selected["p_pagou_30d"] * selected["valor_atrasado_cutoff"] * config["recovered_fraction"]).sum()
    )
    model_cost = float(len(selected) * config["contact_cost"])
    return {
        "n": int(len(scored)),
        "contatos_politica_atual": int(len(scored)),
        "contatos_politica_modelo": int(len(selected)),
        "recuperacao_estimada_atual": round(base_recovery, 2),
        "recuperacao_estimada_modelo": round(model_recovery, 2),
        "custo_atual": round(base_cost, 2),
        "custo_modelo": round(model_cost, 2),
        "roi_atual": round((base_recovery - base_cost) / max(base_cost, 1.0), 4),
        "roi_modelo": round((model_recovery - model_cost) / max(model_cost, 1.0), 4),
        "reducao_contatos": round(1 - len(selected) / max(len(scored), 1), 4),
        "uplift_recuperacao_por_contato": round(
            (model_recovery / max(len(selected), 1))
            - (base_recovery / max(len(scored), 1)),
            4,
        ),
    }


def simulate_risk(
    frame: pd.DataFrame,
    threshold: float = 0.18,
    score_column: str = "p_inadimplente_90d",
    exposure_column: str = "limite_atual",
) -> dict[str, Any]:
    approved = frame[frame[score_column] < threshold]
    baseline_default = float(frame["inadimplente_90d"].mean())
    model_default = float(approved["inadimplente_90d"].mean()) if len(approved) else 0.0
    return {
        "n": int(len(frame)),
        "aprovados": int(len(approved)),
        "taxa_aprovacao": round(len(approved) / max(len(frame), 1), 4),
        "inadimplencia_base": round(baseline_default, 4),
        "inadimplencia_aprovados": round(model_default, 4),
        "exposicao_aprovada": round(float(approved[exposure_column].sum()), 2),
        "reducao_inadimplencia": round(baseline_default - model_default, 4),
        "threshold": threshold,
    }


def business_report(
    frame: pd.DataFrame,
    policies: dict[str, Any] | None = None,
    preset: str = "equilibrado",
) -> dict[str, Any]:
    policies = policies or load_policies()
    risk = policies["risk"]
    limit_config = policies["limit"]
    return {
        "collection": simulate_collections(frame, policies),
        "limit": simulate_limit_policy(
            frame,
            preset=limit_config["presets"][preset],
            margin=float(limit_config["margin"]),
            lgd=float(limit_config["lgd"]),
            usage_rate=float(limit_config["usage_rate"]),
        ),
        "risk": simulate_risk(frame, threshold=float(risk["approval_threshold"])),
    }
