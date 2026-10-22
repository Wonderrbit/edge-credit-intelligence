from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from mlops.evaluation.policy import load_policies
from mlops.judgment.batteries import CLIENT_QUESTIONS
from mlops.judgment.client import JevClient
from mlops.judgment.compose import (
    compose_collection_decision,
    compose_indicator_signals,
    compose_risk_decision,
)
from mlops.judgment.fallback import fallback_indicator_signals


def _get(row: Any, key: str, default: Any = 0.0) -> Any:
    try:
        value = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
    except (TypeError, ValueError):
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return value


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _pseudonym(client_id: str) -> str:
    return hashlib.sha256(client_id.encode("utf-8")).hexdigest()[:12]


def _inputs_hash(row: Any, policies: dict[str, Any]) -> str:
    payload = {
        "client_id": str(_get(row, "client_id", "")),
        "renda_mensal": round(_number(_get(row, "renda_mensal")), 2),
        "limite_atual": round(_number(_get(row, "limite_atual")), 2),
        "pd": round(_number(_get(row, "p_inadimplente_90d")), 6),
        "p_pagou": round(_number(_get(row, "p_pagou_30d")), 6),
        "atraso": round(_number(_get(row, "valor_atrasado_cutoff")), 2),
        "approval_threshold": policies.get("risk", {}).get("approval_threshold"),
        "review_threshold": policies.get("risk", {}).get("review_threshold"),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def build_client_state(row: Any, policies: dict[str, Any]) -> dict[str, Any]:
    client_id = str(_get(row, "client_id", ""))
    risk = policies.get("risk", {})
    collection = policies.get("collection", {})
    limit_config = policies.get("limit", {})
    return {
        "cliente": {
            "id_pseudonimo": _pseudonym(client_id),
            "segmento": str(_get(row, "segmento", "nao_informado")),
            "regiao": str(_get(row, "regiao", "nao_informado")),
            "tempo_relacionamento_meses": round(_number(_get(row, "tempo_relacionamento_meses")), 1),
            "renda_mensal": round(_number(_get(row, "renda_mensal")), 2),
            "limite_atual": round(_number(_get(row, "limite_atual")), 2),
            "utilizacao_limite": round(_number(_get(row, "utilizacao_media")), 4),
            "comprometimento_consumo": round(_number(_get(row, "comprometimento_consumo")), 4),
            "atraso_max_90d": round(_number(_get(row, "atraso_max_90d")), 1),
            "meses_com_atraso": round(_number(_get(row, "meses_com_atraso")), 1),
            "valor_em_atraso": round(_number(_get(row, "valor_atrasado_cutoff")), 2),
            "bureau_score": round(_number(_get(row, "bureau_score")), 1),
            "saldo_open_finance": round(_number(_get(row, "saldo_open_finance_medio")), 2),
            "entradas_open_finance": round(_number(_get(row, "entradas_open_finance_total")), 2),
            "saidas_open_finance": round(_number(_get(row, "saidas_open_finance_total")), 2),
        },
        "scores": {
            "inadimplencia_90d": round(_number(_get(row, "p_inadimplente_90d")), 6),
            "pagamento_30d": round(_number(_get(row, "p_pagou_30d")), 6),
            "aceite_limite": round(_number(_get(row, "p_aceita_limite")), 6),
            "consumo_alto": round(_number(_get(row, "p_consumo_alto")), 6),
        },
        "contexto": {
            "threshold_aprovacao": risk.get("approval_threshold"),
            "threshold_revisao": risk.get("review_threshold"),
            "lgd": limit_config.get("lgd"),
            "custo_contato": collection.get("contact_cost"),
            "fracao_recuperada": collection.get("recovered_fraction"),
        },
    }


@dataclass
class JudgmentBudget:
    max_calls: int
    calls: int = 0
    fallbacks: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def claim(self) -> bool:
        with self.lock:
            if self.calls >= self.max_calls:
                self.fallbacks += 1
                return False
            self.calls += 1
            return True

    def record_fallback(self) -> None:
        with self.lock:
            self.fallbacks += 1

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {"calls": self.calls, "fallbacks": self.fallbacks, "max_calls": self.max_calls}


def decide_client(
    row: Any,
    policies: dict[str, Any],
    jev: JevClient | None,
    use_jev: bool,
    policy_version: str,
    model_version: str | None,
) -> dict[str, Any]:
    client_id = str(_get(row, "client_id"))
    model_pd = _number(_get(row, "p_inadimplente_90d"))
    p_pay = _number(_get(row, "p_pagou_30d"))
    debt = _number(_get(row, "valor_atrasado_cutoff"))
    recovered = float(policies.get("collection", {}).get("recovered_fraction", 0.65))
    contact_cost = float(policies.get("collection", {}).get("contact_cost", 12.0))
    ev = p_pay * debt * recovered - contact_cost

    response = None
    if use_jev and jev is not None and jev.available:
        response = jev.ask(build_client_state(row, policies), CLIENT_QUESTIONS)

    risk = compose_risk_decision(client_id, model_pd, response, policies)
    collection = compose_collection_decision(client_id, p_pay, debt, ev, response, policies)
    if response is not None:
        indicators = compose_indicator_signals(client_id, response)
    else:
        indicators = fallback_indicator_signals(client_id, row)

    inputs_hash = _inputs_hash(row, policies)
    decision_id = hashlib.sha256(
        f"{client_id}:{inputs_hash}:{policy_version}".encode()
    ).hexdigest()[:20]
    adverse = risk.action in {"reprovar", "reduzir"}
    return {
        "decision_id": decision_id,
        "client_id": client_id,
        "inputs_hash": inputs_hash,
        "created_at": datetime.now(UTC).isoformat(),
        "model_version": model_version,
        "policy_version": policy_version,
        "mode": "jev" if response is not None else "fallback",
        "autonomy": "full",
        "adverse_action": adverse,
        "action_status": "ready_for_execution",
        "manual_review_recommended": risk.manual_review_recommended,
        "risk": risk.model_dump(),
        "collection": collection.model_dump(),
        "indicators": indicators.model_dump(),
        "jev": {
            "model": risk.meta.model,
            "request_id": risk.meta.request_id,
            "latency_ms": risk.meta.latency_ms,
            "input_tokens": risk.meta.input_tokens,
            "output_tokens": risk.meta.output_tokens,
        },
    }


def decisions_for_frame(
    frame: pd.DataFrame,
    policies: dict[str, Any] | None = None,
    jev: JevClient | None = None,
    use_jev: bool = True,
    max_workers: int = 8,
    max_calls: int = 500,
    model_version: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    policies = policies or load_policies()
    policy_version = hashlib.sha256(
        json.dumps(policies, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    budget = JudgmentBudget(max_calls=max_calls)
    records = [row for _, row in frame.iterrows()]

    def run(row: Any) -> dict[str, Any]:
        allowed = use_jev and budget.claim()
        decision = decide_client(
            row,
            policies,
            jev if allowed else None,
            allowed,
            policy_version,
            model_version,
        )
        if allowed and decision["mode"] == "fallback":
            budget.record_fallback()
        return decision

    workers = max(1, min(max_workers, len(records) or 1))
    if workers == 1:
        decisions = [run(row) for row in records]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            decisions = list(pool.map(run, records))
    decisions.sort(key=lambda item: item["client_id"])
    return decisions, budget.snapshot()
