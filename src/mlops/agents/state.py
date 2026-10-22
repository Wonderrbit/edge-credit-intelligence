from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    role: str
    client_id: str | None
    thread_id: str
    run_id: str
    started_at: float
    intent: str
    blocked: bool
    guardrail_reason: str
    human_review: bool
    retrieval_query: str
    retrieved: list[dict[str, Any]]
    citations: list[dict[str, str]]
    tool_calls: list[dict[str, Any]]
    tool_results: dict[str, Any]
    answer: str
    latency_ms: float
    trace: Any
    errors: list[str]
