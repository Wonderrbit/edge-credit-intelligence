from __future__ import annotations

import json

import toon_format

from mlops.agents import toon
from mlops.agents.prompts import ROUTER_PROMPT, SYNTHESIS_PROMPT
from mlops.agents.toon import encode_toon


def test_encode_toon_round_trip_tabular_payload() -> None:
    payload = {
        "clientes": [
            {"client_id": "C000001", "score": 0.12, "acao": "revisar"},
            {"client_id": "C000002", "score": 0.03, "acao": "aprovar"},
        ]
    }

    encoded = encode_toon(payload)

    assert "clientes[2]{client_id,score,acao}:" in encoded
    assert toon_format.loads(encoded) == payload


def test_encode_toon_falls_back_to_json(monkeypatch) -> None:
    monkeypatch.setattr(toon, "toon_format", None)

    encoded = encode_toon({"acao": "revisar", "score": 0.12})

    assert json.loads(encoded) == {"acao": "revisar", "score": 0.12}


def test_prompts_embed_toon_without_unformatted_placeholders() -> None:
    payload = encode_toon({"pergunta": "Qual a política de limite?"})
    router = ROUTER_PROMPT.format(payload=payload)
    synthesis = SYNTHESIS_PROMPT.format(
        question="Qual a política de limite?",
        intent="policy",
        context="Contexto recuperado.",
        tool_results=encode_toon({"policy": {"ok": True}}),
    )

    assert "Entrada em TOON" in router
    assert '{"intent": "..."}' in router
    assert "Resultados das ferramentas em TOON" in synthesis
    assert "{tool_results}" not in synthesis
