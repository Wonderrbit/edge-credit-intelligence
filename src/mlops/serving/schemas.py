from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ScoreFeaturesRequest(BaseModel):
    sequence: list[list[float]] = Field(
        ..., description="12 timesteps x 21 features comportamentais"
    )
    static: dict[str, Any] = Field(
        ..., description="Campos estáticos: renda, limite, bureau, região, segmento etc."
    )


class Scores(BaseModel):
    pagou_30d: float
    inadimplente_90d: float
    aceita_limite: float
    consumo_alto: float


class Driver(BaseModel):
    feature: str
    attribution: float | None = None
    importance: float | None = None


class ScoreResponse(BaseModel):
    client_id: str
    model_version: str | None = None
    segmento: str | None = None
    regiao: str | None = None
    renda_mensal: float | None = None
    limite_atual: float | None = None
    scores: Scores
    decision: dict[str, Any]
    top_drivers: list[Driver] = []
    disclaimer: str


class BatchScoreRequest(BaseModel):
    client_ids: list[str] | None = None
    preset: Literal["conservador", "equilibrado", "crescimento"] = "equilibrado"


class ClientSummary(BaseModel):
    client_id: str
    segmento: str
    regiao: str
    renda_mensal: float
    limite_atual: float


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    thread_id: str = Field(default="demo", max_length=100)
    role: str = Field(default="analista", max_length=50)
    client_id: str | None = None


class Citation(BaseModel):
    doc_id: str
    title: str
    section: str
    citation: str


class AgentResponse(BaseModel):
    answer: str
    intent: str
    citations: list[Citation] = []
    tool_calls: list[dict[str, Any]] = []
    human_review: bool = False
    run_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str | None
    agent_mode: str
