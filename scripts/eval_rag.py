from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.agents.rag.evaluate import DEFAULT_MODES, evaluate_modes
from mlops.agents.rag.retriever import get_retriever
from mlops.config import ensure_dirs, resolve_paths

SUMMARY_KEYS = ("hit_at_1", "hit_at_k", "mrr", "recall_at_k", "ndcg_at_k", "term_coverage")


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o retriever RAG com ablação por scorer.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--rerank", action="store_true", help="Avalia também os modos com cross-encoder.")
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--questions", default="data/rag_questions.jsonl")
    parser.add_argument("--output", default="artifacts/eval/rag_eval.json")
    parser.add_argument(
        "--expand",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Liga/desliga a expansão de consulta por léxico de domínio.",
    )
    args = parser.parse_args()

    if args.rerank_model:
        os.environ["MLOPS_RERANK_MODEL"] = args.rerank_model
    if args.expand is not None:
        os.environ["MLOPS_USE_EXPANSION"] = "1" if args.expand else "0"
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    paths = ensure_dirs(resolve_paths())
    questions = Path(args.questions)
    if not questions.is_absolute():
        questions = paths.project_root / questions
    retriever = get_retriever()
    report = evaluate_modes(
        retriever,
        questions,
        k=args.k,
        modes=modes,
        rerank=args.rerank,
    )
    hybrid = report.get("hybrid") or next(iter(report.values()))
    output_report = {
        **hybrid,
        "modes": {
            name: {key: metrics[key] for key in SUMMARY_KEYS}
            for name, metrics in report.items()
        },
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = paths.project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "questions": hybrid["questions"],
        "k": hybrid["k"],
        "best_mode": max(report, key=lambda name: (report[name]["ndcg_at_k"], report[name]["mrr"])),
        "modes": output_report["modes"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Relatório: {output}")


if __name__ == "__main__":
    main()
