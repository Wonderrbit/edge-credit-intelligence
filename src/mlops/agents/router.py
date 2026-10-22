from __future__ import annotations

import json
import re

from mlops.agents.llm import LLMClient
from mlops.agents.prompts import ROUTER_PROMPT
from mlops.agents.toon import encode_toon

INTENTS = {
    "policy",
    "client_score",
    "limit",
    "collection",
    "model_metrics",
    "simulation",
    "out_of_scope",
}

CLIENT_ID = re.compile(r"\bC\d{6}\b", re.IGNORECASE)


def extract_client_id(question: str) -> str | None:
    match = CLIENT_ID.search(question)
    return match.group(0).upper() if match else None


def heuristic_intent(question: str) -> str:
    text = question.lower()
    if extract_client_id(question) or any(
        term in text for term in ("score do cliente", "perfil do cliente", "probabilidade do cliente")
    ):
        return "client_score"
    if any(term in text for term in ("simular", "simulação", "cenário", "impacto de")):
        return "simulation"
    if any(term in text for term in ("limite", "aumento de limite", "reduzir limite")):
        return "limit"
    if any(term in text for term in ("cobrança", "cobranca", "recuper", "inadimpl", "atraso", "negocia")):
        return "collection"
    if any(
        term in text
        for term in ("auc", "psi", "calibra", "métrica", "metrica", "modelo", "arquitetura", "treino", "safra")
    ):
        return "model_metrics"
    if any(
        term in text
        for term in (
            "política",
            "politica",
            "threshold",
            "revisão humana",
            "revisao humana",
            "lgpd",
            "privacidade",
            "faq",
            "governança",
            "governanca",
            "critério",
            "criterio",
            "cadência",
            "cadencia",
            "canal",
        )
    ):
        return "policy"
    return "out_of_scope"


def classify_intent(question: str, llm: LLMClient | None = None) -> str:
    heuristic = heuristic_intent(question)
    if heuristic != "out_of_scope":
        return heuristic
    if llm and llm.available:
        try:
            raw = llm.complete(
                "Você é um roteador de perguntas. Responda apenas JSON.",
                ROUTER_PROMPT.format(payload=encode_toon({"pergunta": question})),
                temperature=0.0,
                json_mode=True,
            )
            intent = json.loads(raw).get("intent", "out_of_scope")
            if intent in INTENTS:
                return intent
        except Exception:
            pass
    return heuristic
