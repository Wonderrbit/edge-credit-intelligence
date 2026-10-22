from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.agents.llm import get_llm
from mlops.judgment import compose_intent_decision, get_jev
from mlops.judgment.batteries import INTENT_QUESTIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test das integrações OpenAI e Jev.")
    parser.add_argument("--require", action="store_true", help="Falha se alguma chave estiver ausente.")
    args = parser.parse_args()

    llm = get_llm()
    jev = get_jev()
    result = {
        "openai": {"mode": llm.mode, "model": llm.model, "available": llm.available},
        "jev": {
            "mode": jev.mode,
            "model": jev.model,
            "available": jev.available,
            "init_error": jev.init_error,
        },
    }

    if llm.available:
        try:
            result["openai"]["sample"] = llm.complete(
                "Responda apenas OK.",
                "Teste de conectividade.",
                temperature=0.0,
            )
            result["openai"]["usage"] = llm.last_usage
        except Exception as exc:
            result["openai"]["error"] = str(exc)
            result["openai"]["available"] = False

    if jev.available:
        response = jev.ask(
            {"pergunta": "Qual é a política de limite e revisão humana?"},
            INTENT_QUESTIONS,
        )
        if response:
            decision = compose_intent_decision(response)
            result["jev"]["intent"] = decision.intent
            result["jev"]["confidence"] = decision.confidence
            result["jev"]["latency_ms"] = response.latency_ms
        else:
            result["jev"]["error"] = "chamada falhou"

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.require and (not result["openai"]["available"] or not result["jev"]["available"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
