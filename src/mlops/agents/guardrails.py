from __future__ import annotations

import re
from dataclasses import dataclass

from mlops.agents.rag.redact import redact_pii

INJECTION_PATTERNS = [
    r"ignore (as|suas) instru",
    r"ignore previous",
    r"system prompt",
    r"prompt interno",
    r"revele (o|seu) prompt",
    r"desative (a|as) (revis|regra|pol)",
    r"finja que",
    r"aja como",
]

AUTOMATIC_DECISION_PATTERNS = [
    r"aprove automaticamente",
    r"negar automaticamente",
    r"reprove automaticamente",
    r"reduza o limite sem",
    r"execute a cobran",
]

BULK_DATA_PATTERNS = [
    r"todos os (clientes|cpfs|dados)",
    r"exporte (a base|os dados|todos)",
    r"liste (todos os )?(cpfs|telefones|emails)",
]

PII_PATTERN = re.compile(
    r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b|"
    r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b|"
    r"\b(?:\+55\s*)?\(?\d{2}\)?\s*9?\d{4}[-\s]?\d{4}\b"
)

ADVERSE_TERMS = (
    "reprovado",
    "negado",
    "recusado",
    "reduzir o limite",
    "negativar",
)


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""
    human_review: bool = False
    sanitized_question: str = ""


def check_input(question: str, role: str = "analista") -> GuardrailResult:
    text = question.strip()
    if len(text) < 3:
        return GuardrailResult(False, "Pergunta muito curta.", False, text)
    if len(text) > 2000:
        return GuardrailResult(False, "Pergunta acima do limite de 2000 caracteres.", False, text[:2000])
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailResult(
                False,
                "A pergunta tenta alterar instruções ou regras do agente.",
                True,
                redact_pii(text),
            )
    for pattern in AUTOMATIC_DECISION_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailResult(
                False,
                "Decisões adversas não podem ser executadas automaticamente; exige revisão humana.",
                True,
                redact_pii(text),
            )
    for pattern in BULK_DATA_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailResult(
                False,
                "Consultas em massa de dados pessoais não são permitidas.",
                True,
                redact_pii(text),
            )
    if PII_PATTERN.search(text):
        return GuardrailResult(
            False,
            "A pergunta contém dado pessoal. Remova CPF, e-mail ou telefone e tente novamente.",
            True,
            redact_pii(text),
        )
    if role == "cliente" and re.search(r"\bC\d{6}\b", text, re.IGNORECASE):
        return GuardrailResult(
            False,
            "O perfil cliente não pode consultar scores de terceiros.",
            True,
            redact_pii(text),
        )
    return GuardrailResult(True, "", False, redact_pii(text))


def check_output(answer: str, citations: list[dict], intent: str) -> tuple[str, bool, str]:
    sanitized = redact_pii(answer)
    human_review = False
    reason = ""
    if any(term in sanitized.lower() for term in ADVERSE_TERMS):
        human_review = True
        reason = "A resposta contém recomendação adversa; revisão humana obrigatória."
    if intent in {"policy", "model_metrics"} and not citations:
        human_review = True
        reason = reason or "Resposta de política sem citação; revisão recomendada."
    return sanitized, human_review, reason
