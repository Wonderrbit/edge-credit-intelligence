from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlops.agents.rag.redact import redact_pii
from mlops.config import ensure_dirs, resolve_paths


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class AgentTrace:
    run_id: str
    question: str
    role: str
    thread_id: str
    started_at: float = field(default_factory=time.perf_counter)
    events: list[dict[str, Any]] = field(default_factory=list)

    def log(self, event: str, **payload: Any) -> None:
        safe_payload = {
            key: redact_pii(value) if isinstance(value, str) else value
            for key, value in payload.items()
        }
        self.events.append(
            {
                "ts": datetime.now(UTC).isoformat(),
                "event": event,
                **safe_payload,
            }
        )

    def write(self, answer: str, intent: str, human_review: bool) -> Path:
        paths = ensure_dirs(resolve_paths())
        path = paths.logs / "agent_traces.jsonl"
        record = {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "role": self.role,
            "question": redact_pii(self.question),
            "intent": intent,
            "answer": redact_pii(answer),
            "human_review": human_review,
            "latency_ms": round((time.perf_counter() - self.started_at) * 1000, 2),
            "events": self.events,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path
