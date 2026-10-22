from __future__ import annotations

import re
from typing import Any

from mlops.agents.guardrails import (
    AUTOMATIC_DECISION_PATTERNS,
    BULK_DATA_PATTERNS,
    INJECTION_PATTERNS,
    PII_PATTERN,
)
from mlops.agents.router import heuristic_intent
from mlops.judgment.compose import (
    compose_collection_decision,
    compose_risk_decision,
)
from mlops.judgment.schemas import (
    CollectionDecision,
    GuardAssessment,
    IndicatorSignals,
    IntentDecision,
    JudgmentMeta,
    RAGVerification,
    RiskDecision,
)


def fallback_risk_decision(
    client_id: str,
    model_pd: float,
    policies: dict[str, Any],
) -> RiskDecision:
    return compose_risk_decision(client_id, model_pd, None, policies)


def fallback_collection_decision(
    client_id: str,
    p_pay: float,
    debt: float,
    ev: float,
    policies: dict[str, Any],
) -> CollectionDecision:
    return compose_collection_decision(client_id, p_pay, debt, ev, None, policies)


def _get(row: Any, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
        return float(value)
    except (TypeError, ValueError):
        return default


def fallback_indicator_signals(client_id: str, row: Any) -> IndicatorSignals:
    income = max(_get(row, "renda_mensal"), 1.0)
    utilization = _get(row, "utilizacao_media")
    commitment = _get(row, "comprometimento_consumo")
    overdue = _get(row, "valor_atrasado_cutoff")
    open_finance_entries = _get(row, "entradas_open_finance_total")
    expected_entries = income * 12
    consistency = 1.0 - min(1.0, abs(open_finance_entries - expected_entries) / (expected_entries + 1.0))
    signals = {
        "income_stable": 1.0 if consistency >= 0.6 else 0.35,
        "essential_expenses_high": 0.8 if commitment >= 0.5 else 0.2,
        "debt_ratio_high": min(1.0, (overdue / income) * 2.0),
        "open_finance_consistent": round(consistency, 4),
        "employment_change": 0.0,
    }
    stress = min(3.0, utilization * 2.0 + commitment * 2.0 + signals["debt_ratio_high"])
    return IndicatorSignals(
        client_id=client_id,
        signals=signals,
        stress_score=round(stress, 2),
        stress_confidence=None,
        channel_preference="whatsapp",
        meta=JudgmentMeta(source="fallback"),
    )


def fallback_rag_verification(answer: str, citations: list[dict[str, Any]]) -> RAGVerification:
    supported = 1.0 if citations else 0.0
    complete = min(1.0, len(answer) / 200.0)
    decision = "aceitar" if citations and len(answer) >= 40 else "revisar"
    return RAGVerification(
        supported=supported,
        contradicts=0.0,
        complete=round(complete, 4),
        grounding="suportada" if supported else "nao_suportada",
        decision=decision,  # type: ignore[arg-type]
        meta=JudgmentMeta(source="fallback"),
    )


def fallback_guard_assessment(text: str) -> GuardAssessment:
    lowered = text.lower()
    hazards = {
        "jailbreak": 0.9 if any(re.search(p, lowered) for p in INJECTION_PATTERNS) else 0.02,
        "harmful": 0.0,
        "pii_request": 0.8 if PII_PATTERN.search(text) else 0.02,
        "bulk_export": 0.9 if any(re.search(p, lowered) for p in BULK_DATA_PATTERNS) else 0.02,
        "automatic_decision": 0.9
        if any(re.search(p, lowered) for p in AUTOMATIC_DECISION_PATTERNS)
        else 0.02,
    }
    reasons = [f"fallback_{key}" for key, value in hazards.items() if value >= 0.7]
    decision = "block" if any(value >= 0.7 for value in hazards.values()) else "pass"
    return GuardAssessment(
        decision=decision,  # type: ignore[arg-type]
        hazards=hazards,
        severity=2.0 if decision == "block" else 0.0,
        reasons=reasons,
        meta=JudgmentMeta(source="fallback"),
    )


def fallback_intent_decision(question: str) -> IntentDecision:
    return IntentDecision(
        intent=heuristic_intent(question),
        confidence=None,
        meta=JudgmentMeta(source="fallback"),
    )
