from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.agents.evaluate import evaluate_agent
from mlops.agents.rag.evaluate import evaluate_retriever
from mlops.agents.rag.ingest import build_index
from mlops.agents.rag.retriever import get_retriever
from mlops.config import ensure_dirs, resolve_paths
from mlops.pipelines.export_tfjs import export_tfjs
from mlops.pipelines.monitoring import run_monitoring
from mlops.pipelines.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline ponta a ponta do Akropoli Decision Copilot.")
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--months", type=int, default=None)
    parser.add_argument("--fast", action="store_true", help="Treino rápido para smoke test.")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-agent-eval", action="store_true")
    parser.add_argument("--embeddings", action="store_true")
    args = parser.parse_args()

    paths = ensure_dirs(resolve_paths())
    if not args.skip_training:
        print("[1/6] Treinando baselines, Edge multi-task e políticas...")
        result = run_training(
            n_clients=args.clients,
            months=args.months,
            fast=args.fast,
            force_data=True,
        )
        print(
            json.dumps(
                {
                    "registry": result["registry"],
                    "edge_metrics": result["metrics"]["edge_calibrated"],
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    else:
        print("[1/6] Treino pulado.")

    print("[2/6] Exportando TensorFlow.js...")
    export_info = export_tfjs()
    print(json.dumps({"total_bytes": export_info["total_bytes"], "parameters": export_info["parameters"]}, indent=2))

    print("[3/6] Construindo índice RAG...")
    rag_meta = build_index(
        paths.knowledge,
        paths.rag,
        use_embeddings=args.embeddings,
    )
    print(json.dumps(rag_meta, indent=2, ensure_ascii=False))

    print("[4/6] Avaliando RAG...")
    rag_report = evaluate_retriever(
        get_retriever(use_embeddings=args.embeddings),
        paths.project_root / "data" / "rag_questions.jsonl",
        k=5,
    )
    (paths.eval / "rag_eval.json").write_text(
        json.dumps(rag_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in rag_report.items() if key != "rows"},
            indent=2,
        )
    )

    print("[5/6] Gerando relatório de monitoramento...")
    monitoring = run_monitoring()
    print(json.dumps({"status": monitoring["status"], "alerts": monitoring["alerts"]}, indent=2, ensure_ascii=False))

    if not args.skip_agent_eval:
        print("[6/6] Avaliando agente...")
        agent_report = evaluate_agent(
            paths.project_root / "data" / "agent_eval.json",
            paths.eval / "agent_eval.json",
        )
        print(
            json.dumps(
                {key: value for key, value in agent_report.items() if key != "rows"},
                indent=2,
            )
        )
    else:
        print("[6/6] Avaliação do agente pulada.")


if __name__ == "__main__":
    main()
