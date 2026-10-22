from __future__ import annotations

from typing import Any

from mlops.judgment.client import JevResponse
from mlops.judgment.schemas import (
    CollectionDecision,
    GuardAssessment,
    IndicatorSignals,
    IntentDecision,
    JudgmentMeta,
    RAGVerification,
    RiskDecision,
)


def _meta(jev: JevResponse | None) -> JudgmentMeta:
    if jev is None:
        return JudgmentMeta(source="fallback")
    return JudgmentMeta(
        source="jev",
        model=jev.model,
        request_id=jev.request_id,
        latency_ms=jev.latency_ms,
        input_tokens=jev.input_tokens,
        output_tokens=jev.output_tokens,
    )


def _noul(jev: JevResponse | None, key: str) -> float | None:
    if jev is None:
        return None
    answer = jev.answers.get(key)
    value = getattr(answer, "noul", None)
    return float(value) if value is not None else None


def _choice(jev: JevResponse | None, key: str) -> tuple[str | None, float | None]:
    if jev is None:
        return None, None
    answer = jev.answers.get(key)
    if answer is None:
        return None, None
    return getattr(answer, "choice", None), getattr(answer, "confidence", None)


def _score(jev: JevResponse | None, key: str) -> tuple[float | None, float | None]:
    if jev is None:
        return None, None
    answer = jev.answers.get(key)
    if answer is None:
        return None, None
    return getattr(answer, "score", None), getattr(answer, "confidence", None)


def _signals(jev: JevResponse | None, prefix: str) -> dict[str, float]:
    if jev is None:
        return {}
    signals: dict[str, float] = {}
    for key, answer in jev.answers.items():
        if key.startswith(prefix) and hasattr(answer, "noul"):
            signals[key.removeprefix(prefix)] = float(answer.noul)
    return signals


def compose_risk_decision(
    client_id: str,
    model_pd: float,
    jev: JevResponse | None,
    policies: dict[str, Any],
) -> RiskDecision:
    approval = float(policies.get("risk", {}).get("approval_threshold", 0.18))
    review = float(policies.get("risk", {}).get("review_threshold", 0.12))
    signals = _signals(jev, "risk_")
    hardship, _ = _score(jev, "risk_hardship")
    action_choice, action_confidence = _choice(jev, "risk_action")

    adverse = signals.get("adverse_event", 0.0)
    fraud = signals.get("fraud_signal", 0.0)
    capacity = signals.get("capacity_stressed", 0.0)
    income = signals.get("income_consistent", 1.0)
    docs = signals.get("docs_complete", 1.0)
    jev_risk = (
        0.30 * adverse
        + 0.25 * fraud
        + 0.20 * capacity
        + 0.15 * (1.0 - income)
        + 0.10 * (1.0 - docs)
    )
    composite = 0.70 * model_pd + 0.30 * jev_risk if signals else model_pd

    reasons: list[str] = []
    if model_pd >= approval:
        action = "reprovar"
        reasons.append("pd_acima_threshold_aprovacao")
    elif model_pd >= review:
        action = "revisar"
        reasons.append("pd_acima_threshold_revisao")
    else:
        action = "aprovar"

    if fraud >= 0.80:
        action = "reprovar"
        reasons.append("jev_fraude_alta")
    elif adverse >= 0.70 and action == "aprovar":
        action = "revisar"
        reasons.append("jev_evento_adverso")
    elif capacity >= 0.70 and action == "aprovar":
        action = "revisar"
        reasons.append("jev_capacidade_pressionada")
    elif (income <= 0.35 or docs <= 0.35) and action == "aprovar":
        action = "revisar"
        reasons.append("jev_evidencia_insuficiente")

    if composite >= approval and action == "aprovar":
        action = "reprovar"
        reasons.append("risco_composto_acima_threshold")
    elif composite >= review and action == "aprovar":
        action = "revisar"
        reasons.append("risco_composto_acima_revisao")

    if hardship is not None and hardship >= 2.5:
        reasons.append("jev_dificuldade_financeira_grave")
        if action == "aprovar":
            action = "revisar"

    low_confidence = action_confidence is not None and action_confidence < 0.60
    if low_confidence:
        reasons.append("jev_baixa_confianca")
    if jev is not None and action_choice and action_choice != action:
        reasons.append(f"jev_sugeriu_{action_choice}")

    if jev is None:
        reasons.append("fallback_deterministico")

    return RiskDecision(
        client_id=client_id,
        action=action,  # type: ignore[arg-type]
        confidence=action_confidence,
        composite_risk=round(float(composite), 6),
        model_pd=round(float(model_pd), 6),
        signals={key: round(value, 4) for key, value in signals.items()},
        hardship=hardship,
        reasons=reasons,
        manual_review_recommended=bool(low_confidence or action == "revisar"),
        meta=_meta(jev),
    )


def compose_collection_decision(
    client_id: str,
    p_pay: float,
    debt: float,
    ev: float,
    jev: JevResponse | None,
    policies: dict[str, Any],
) -> CollectionDecision:
    actions = policies.get("collection", {}).get("actions", {})
    if debt <= 0 or ev <= 0:
        base_action = "nenhuma"
    elif p_pay >= 0.50:
        base_action = "lembrete"
    elif p_pay >= 0.30:
        base_action = "whatsapp"
    elif p_pay >= 0.15:
        base_action = "ligacao"
    else:
        base_action = "negociacao"

    signals = _signals(jev, "collection_")
    channel, confidence = _choice(jev, "collection_channel")
    priority_choice, _ = _choice(jev, "collection_priority")
    _, urgency_confidence = _score(jev, "collection_urgency")
    action = base_action
    reasons: list[str] = []

    if channel and channel in actions and (confidence or 0.0) >= 0.55:
        action = channel
        reasons.append("jev_canal")
    if signals.get("disputes_debt", 0.0) >= 0.70:
        action = "nenhuma"
        reasons.append("jev_divida_disputada")
    if signals.get("vulnerable", 0.0) >= 0.70:
        action = "negociacao"
        reasons.append("jev_vulnerabilidade")
    if signals.get("hardship", 0.0) >= 0.70 and action in {"ligacao", "lembrete"}:
        action = "negociacao"
        reasons.append("jev_dificuldade_financeira")
    if not signals.get("contactable", 1.0) >= 0.35:
        action = "nenhuma"
        reasons.append("jev_sem_canal_contato")

    if priority_choice in {"alta", "media", "baixa"}:
        priority = priority_choice
    elif ev > 100 and p_pay < 0.40:
        priority = "alta"
    elif ev > 0:
        priority = "media"
    else:
        priority = "baixa"

    if confidence is not None and confidence < 0.55:
        reasons.append("jev_baixa_confianca")
    if urgency_confidence is not None and urgency_confidence < 0.55:
        reasons.append("jev_urgencia_incerta")
    if jev is None:
        reasons.append("fallback_deterministico")

    return CollectionDecision(
        client_id=client_id,
        action=action,
        priority=priority,  # type: ignore[arg-type]
        channel_confidence=confidence,
        ev=round(float(ev), 2),
        signals={key: round(value, 4) for key, value in signals.items()},
        reasons=reasons,
        meta=_meta(jev),
    )


def compose_indicator_signals(client_id: str, jev: JevResponse | None) -> IndicatorSignals:
    signals = _signals(jev, "indicator_")
    stress, stress_confidence = _score(jev, "indicator_stress")
    channel, _ = _choice(jev, "indicator_channel_preference")
    return IndicatorSignals(
        client_id=client_id,
        signals={key: round(value, 4) for key, value in signals.items()},
        stress_score=stress,
        stress_confidence=stress_confidence,
        channel_preference=channel,
        meta=_meta(jev),
    )


def compose_rag_verification(jev: JevResponse | None) -> RAGVerification:
    supported = _noul(jev, "rag_supported") or 0.0
    contradicts = _noul(jev, "rag_contradicts") or 0.0
    complete = _noul(jev, "rag_complete") or 0.0
    grounding, _ = _choice(jev, "rag_grounding")
    if contradicts >= 0.60 or supported < 0.30:
        decision = "rejeitar"
    elif supported >= 0.70 and contradicts < 0.30 and complete >= 0.50:
        decision = "aceitar"
    else:
        decision = "revisar"
    return RAGVerification(
        supported=supported,
        contradicts=contradicts,
        complete=complete,
        grounding=grounding or "indefinido",
        decision=decision,  # type: ignore[arg-type]
        meta=_meta(jev),
    )


def compose_guard_assessment(jev: JevResponse | None) -> GuardAssessment:
    hazards = _signals(jev, "guard_")
    hazards.pop("severity", None)
    severity, _ = _score(jev, "guard_severity")
    reasons: list[str] = []
    decision = "pass"
    for key, value in hazards.items():
        if value >= 0.70:
            reasons.append(f"jev_{key}_alto")
            decision = "block"
        elif value >= 0.35:
            reasons.append(f"jev_{key}_revisar")
            if decision != "block":
                decision = "review"
    if severity is not None and severity >= 2.0 and decision != "block":
        decision = "block"
        reasons.append("jev_severidade_alta")
    if jev is None:
        reasons.append("fallback_deterministico")
    return GuardAssessment(
        decision=decision,  # type: ignore[arg-type]
        hazards={key: round(value, 4) for key, value in hazards.items()},
        severity=severity,
        reasons=reasons,
        meta=_meta(jev),
    )


def compose_intent_decision(jev: JevResponse | None) -> IntentDecision:
    intent, confidence = _choice(jev, "intent")
    return IntentDecision(
        intent=intent or "out_of_scope",
        confidence=confidence,
        meta=_meta(jev),
    )
