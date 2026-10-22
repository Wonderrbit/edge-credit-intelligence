from __future__ import annotations

import re
import unicodedata

LEXICON: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("calote", "inadimplente", "inadimplencia", "default", "atraso grave", "atraso", "risco de credito"),
        ("inadimplencia", "default", "risco", "atraso"),
    ),
    (
        ("analise manual", "analise humana", "revisao humana", "revisao", "aprovacao manual"),
        ("revisao", "humana", "manual"),
    ),
    (
        ("rede", "rede neural", "modelo edge", "multi task", "multitask", "cabecas", "previsoes"),
        ("modelo", "edge", "gru", "multi task", "cabecas", "neural"),
    ),
    (
        ("safra", "safras", "janela temporal"),
        ("safra", "treino", "validacao", "teste"),
    ),
    (
        ("cobranca", "recuperacao", "divida", "devedor"),
        ("cobranca", "recuperacao", "negociacao"),
    ),
    (
        ("telefonema", "ligacao", "telefone"),
        ("ligacao", "telefone", "contato"),
    ),
    (
        ("mensagem", "whatsapp", "sms", "lembrete"),
        ("whatsapp", "mensagem", "lembrete"),
    ),
    (
        ("retencao", "guarda", "prazo", "logs", "registros"),
        ("retencao", "logs", "prazo", "auditoria"),
    ),
    (
        ("comprometimento", "multiplo", "preset", "renda mensal"),
        ("limite", "renda", "preset", "multiplo"),
    ),
    (
        ("psi", "estabilidade", "drift", "distribuicao"),
        ("psi", "estabilidade", "distribuicao", "monitoramento"),
    ),
    (
        ("explicacao", "explicabilidade", "shap", "atribuicao"),
        ("shap", "gradiente", "explicabilidade"),
    ),
    (
        ("lgpd", "privacidade", "pii", "dados pessoais", "mascaramento"),
        ("lgpd", "pii", "privacidade", "mascaramento"),
    ),
    (
        ("custo", "preco", "tarifa"),
        ("custo", "preco", "tarifa"),
    ),
    (
        ("desconto", "parcelamento", "negociacao"),
        ("negociacao", "desconto", "parcelamento"),
    ),
    (
        ("injecao", "prompt", "instrucao", "ordem escondida"),
        ("injecao", "instrucoes", "prompt", "allowlist"),
    ),
    (
        ("dispositivo", "nuvem", "on device", "edge"),
        ("dispositivo", "nuvem", "edge", "local"),
    ),
    (
        ("sintetico", "dados reais", "producao"),
        ("sintetico", "reais", "producao"),
    ),
    (
        ("calibracao", "temperatura", "brier", "ece"),
        ("calibracao", "temperatura", "brier", "ece"),
    ),
    (
        ("valor esperado", "retorno esperado", "ev"),
        ("valor", "esperado", "ev", "retorno"),
    ),
    (
        ("uplift", "experimento", "controle", "teste a b"),
        ("uplift", "experimento", "controle"),
    ),
    (
        ("fairness", "justica", "discriminacao", "segmento", "regiao"),
        ("fairness", "segmento", "regiao"),
    ),
    (
        ("threshold", "limiar", "corte", "percentual"),
        ("threshold", "limiar", "corte"),
    ),
    (
        ("aprovacao", "aprovar", "negar", "reprovar"),
        ("aprovacao", "reprovar", "revisao"),
    ),
    (
        ("cadencia", "regua", "frequencia"),
        ("cadencia", "regua", "contato"),
    ),
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return decomposed.encode("ascii", "ignore").decode("ascii")


def expand_query(query: str) -> str:
    normalized = _normalize(query)
    extra: list[str] = []
    for triggers, expansions in LEXICON:
        if any(re.search(rf"\b{re.escape(trigger)}\b", normalized) for trigger in triggers):
            extra.extend(expansions)
    if not extra:
        return query
    unique = list(dict.fromkeys(extra))
    return f"{query} {' '.join(unique)}"
