from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

logger = logging.getLogger(__name__)


@dataclass
class JevResponse:
    answers: dict[str, Any]
    model: str
    request_id: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class JevClient:
    """Small, non-blocking wrapper around the TypeSafe System One API.

    The wrapper never raises on transport/API failures. Callers receive ``None``
    and are expected to use their deterministic fallback.
    """

    def __init__(self) -> None:
        self.api_key = (os.getenv("TYPESAFE_API_KEY") or "").strip() or None
        self.model = os.getenv("TYPESAFE_MODEL", "jev-latest")
        self.timeout = float(os.getenv("MLOPS_JEV_TIMEOUT", "30"))
        self.enabled = os.getenv("MLOPS_USE_JEV", "1") == "1"
        self.client: TypeSafeClient | None = None
        self.init_error: str | None = None
        if self.enabled and self.api_key:
            try:
                base_url = (os.getenv("TYPESAFE_BASE_URL") or "").strip() or None
                self.client = TypeSafeClient(
                    api_key=self.api_key,
                    model=self.model,
                    base_url=base_url,
                    timeout=self.timeout,
                    retry=RetryPolicy(max_retries=2),
                )
            except TypeSafeError as exc:
                self.init_error = str(exc)
                logger.warning("Jev desabilitado: %s", exc)

    @property
    def available(self) -> bool:
        return self.client is not None

    @property
    def mode(self) -> str:
        return "jev" if self.available else "offline"

    def ask(self, state: Any, questions: dict[str, Any]) -> JevResponse | None:
        if not self.client:
            return None
        started = time.perf_counter()
        try:
            response = self.client.system_one(
                state=state,
                questions=questions,
                model=self.model,
            )
        except TypeSafeError as exc:
            logger.warning("Falha na chamada Jev: %s", exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning("Falha inesperada na chamada Jev: %s", exc)
            return None
        usage = getattr(response, "usage", None)
        return JevResponse(
            answers=dict(getattr(response, "answers", {}) or {}),
            model=getattr(response, "model", self.model),
            request_id=getattr(response, "request_id", None),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            raw={},
        )

    def close(self) -> None:
        client = self.client
        self.client = None
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception:  # pragma: no cover - best effort
                pass


@lru_cache(maxsize=1)
def get_jev() -> JevClient:
    return JevClient()
