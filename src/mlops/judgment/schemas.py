from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JudgmentMeta(BaseModel):
    source: Literal["jev", "fallback"]
    model: str | None = None
    request_id: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class RiskDecision(BaseModel):
    client_id: str
    action: Literal["aprovar", "manter", "reduzir", "reprovar", "revisar"]
    confidence: float | None = None
    composite_risk: float
    model_pd: float
    signals: dict[str, float] = Field(default_factory=dict)
    hardship: float | None = None
    reasons: list[str] = Field(default_factory=list)
    manual_review_recommended: bool = False
    meta: JudgmentMeta


class CollectionDecision(BaseModel):
    client_id: str
    action: str
    priority: Literal["alta", "media", "baixa"]
    channel_confidence: float | None = None
    ev: float
    signals: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    meta: JudgmentMeta


class IndicatorSignals(BaseModel):
    client_id: str
    signals: dict[str, float] = Field(default_factory=dict)
    stress_score: float | None = None
    stress_confidence: float | None = None
    channel_preference: str | None = None
    meta: JudgmentMeta


class RAGVerification(BaseModel):
    supported: float
    contradicts: float
    complete: float
    grounding: str
    decision: Literal["aceitar", "revisar", "rejeitar"]
    meta: JudgmentMeta


class GuardAssessment(BaseModel):
    decision: Literal["pass", "review", "block"]
    hazards: dict[str, float] = Field(default_factory=dict)
    severity: float | None = None
    reasons: list[str] = Field(default_factory=list)
    meta: JudgmentMeta


class IntentDecision(BaseModel):
    intent: str
    confidence: float | None = None
    meta: JudgmentMeta
