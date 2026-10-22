from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from mlops.agents.llm import get_llm
from mlops.agents.toon import encode_toon


class PortfolioNarrative(BaseModel):
    headline: str = Field(description="Resumo executivo em uma frase.")
    key_findings: list[str] = Field(description="3 a 5 achados objetivos sobre a carteira.")
    risks: list[str] = Field(description="2 a 4 riscos ou anomalias que exigem atenção.")
    recommended_actions: list[str] = Field(description="2 a 5 ações operacionais recomendadas.")
    disclaimer: str = Field(description="Aviso de apoio à decisão e auditoria.")


def generate_narrative(
    portfolio: dict[str, Any],
    alerts: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> PortfolioNarrative | None:
    if os.getenv("MLOPS_USE_LLM", "0") != "1":
        return None
    llm = get_llm()
    if not llm.available:
        return None
    payload = {
        "portfolio": portfolio,
        "alerts": alerts,
        "manifest": {
            "run_id": manifest.get("run_id"),
            "model_version": manifest.get("model_version"),
            "policy_version": manifest.get("policy_version"),
            "autonomy": manifest.get("autonomy"),
        },
    }
    try:
        return llm.complete_structured(
            "Você é analista sênior de risco de crédito e cobrança. "
            "Escreva em português do Brasil, de forma objetiva e sem inventar números. "
            "Use apenas os dados fornecidos.",
            encode_toon(payload),
            PortfolioNarrative,
            temperature=0.1,
        )
    except Exception:
        return None
