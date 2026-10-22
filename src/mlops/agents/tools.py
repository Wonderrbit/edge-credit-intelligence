from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from mlops.agents.rag.redact import redact_pii
from mlops.agents.rag.retriever import (
    ContextualRetriever,
    HybridRetriever,
    get_contextual_retriever,
    get_retriever,
)
from mlops.evaluation.policy import load_policies
from mlops.serving.predictor import CreditPredictor, get_predictor

ALLOWED_TOOLS = {
    "policy_search",
    "score_client",
    "limit_recommendation",
    "collection_queue",
    "model_metrics",
    "simulate_collection",
    "client_drivers",
}


class AgentTools:
    def __init__(
        self,
        predictor: CreditPredictor | None = None,
        retriever: HybridRetriever | ContextualRetriever | None = None,
    ):
        self.predictor = predictor or get_predictor()
        if retriever is not None:
            self.retriever = retriever
        elif os.getenv("MLOPS_USE_CONTEXTUAL", "1") == "1":
            self.retriever = get_contextual_retriever()
        else:
            self.retriever = get_retriever()
        self.policies = load_policies()
        self.calls: list[dict[str, Any]] = []

    def _call(self, name: str, payload: dict[str, Any], fn: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        try:
            result = fn()
            self.calls.append(
                {
                    "tool": name,
                    "arguments": payload,
                    "status": "ok",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
            return result
        except Exception as exc:
            self.calls.append(
                {
                    "tool": name,
                    "arguments": payload,
                    "status": "error",
                    "error": redact_pii(str(exc)),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
            raise

    def policy_search(self, query: str, k: int = 5) -> dict[str, Any]:
        results = self.retriever.search(query, k=k)
        context = self.retriever.format_context(results)
        return {
            "context": context,
            "context_chars": len(context),
            "retriever": type(self.retriever).__name__,
            "citations": self.retriever.citations(results),
            "results": [result.to_dict() for result in results],
        }

    def score_client(self, client_id: str, preset: str = "equilibrado") -> dict[str, Any]:
        return self._call(
            "score_client",
            {"client_id": client_id, "preset": preset},
            lambda: self.predictor.score_client(client_id, preset=preset),
        )

    def limit_recommendation(self, client_id: str, preset: str = "equilibrado") -> dict[str, Any]:
        def run() -> dict[str, Any]:
            score = self.predictor.score_client(client_id, preset=preset)
            return {
                "client_id": client_id,
                "preset": preset,
                "scores": score["scores"],
                "limit": score["decision"]["limite"],
                "risk_decision": score["decision"]["risco"],
            }

        return self._call(
            "limit_recommendation",
            {"client_id": client_id, "preset": preset},
            run,
        )

    def collection_queue(self, top_n: int = 10) -> dict[str, Any]:
        return self._call(
            "collection_queue",
            {"top_n": top_n},
            lambda: {"queue": self.predictor.collection_queue(top_n=top_n)},
        )

    def model_metrics(self) -> dict[str, Any]:
        return self._call("model_metrics", {}, self.predictor.model_info)

    def client_drivers(self, client_id: str) -> dict[str, Any]:
        def run() -> dict[str, Any]:
            score = self.predictor.score_client(client_id)
            return {
                "client_id": client_id,
                "top_drivers": score.get("top_drivers", []),
                "scores": score["scores"],
            }

        return self._call("client_drivers", {"client_id": client_id}, run)

    def simulate_collection(self, client_id: str, action: str = "whatsapp") -> dict[str, Any]:
        def run() -> dict[str, Any]:
            score = self.predictor.score_client(client_id)
            actions = self.policies["collection"]["actions"]
            if action not in actions:
                raise ValueError(f"Ação inválida. Opções: {sorted(actions)}")
            config = actions[action]
            debt = float(score["decision"]["cobranca"]["valor_em_atraso"])
            recovered_fraction = float(self.policies["collection"]["recovered_fraction"])
            p_pay = float(score["scores"]["pagou_30d"])
            p_uplift = min(0.99, p_pay + float(config["uplift"]))
            ev_base = p_pay * debt * recovered_fraction
            ev_action = p_uplift * debt * recovered_fraction - float(config["cost"])
            return {
                "client_id": client_id,
                "action": action,
                "debt": debt,
                "p_pay": p_pay,
                "p_pay_with_action": p_uplift,
                "ev_base": round(ev_base, 2),
                "ev_with_action": round(ev_action, 2),
                "delta_ev": round(ev_action - ev_base, 2),
                "cost": float(config["cost"]),
                "uplift": float(config["uplift"]),
            }

        return self._call(
            "simulate_collection",
            {"client_id": client_id, "action": action},
            run,
        )
