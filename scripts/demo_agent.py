from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.agents.graph import run_agent
from mlops.agents.llm import get_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo do copiloto de decisão.")
    parser.add_argument(
        "question",
        nargs="?",
        default="Qual é o threshold de aprovação e quando devo encaminhar para revisão humana?",
    )
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--role", default="analista")
    args = parser.parse_args()

    print(f"Modo do LLM: {get_llm().mode}")
    result = run_agent(args.question, role=args.role, client_id=args.client_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
