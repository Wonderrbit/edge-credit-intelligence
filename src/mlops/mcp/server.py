from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - compatibility with mcp<2
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[assignment]

from mlops.autopilot.decisions import decide_client
from mlops.autopilot.runner import _model_version
from mlops.autopilot.store import load_latest
from mlops.config import resolve_paths
from mlops.evaluation.policy import load_policies
from mlops.judgment import get_jev

server = MCPServer(
    name="akropoli-credit",
    instructions=(
        "Ferramentas read-only de crédito, limite, cobrança e políticas da Akropoli. "
        "Decisões adversas são auditadas; use os resultados como apoio."
    ),
)


@server.tool()
def score_client(client_id: str, preset: str = "equilibrado") -> dict[str, Any]:
    from mlops.serving.predictor import get_predictor

    return get_predictor().score_client(client_id, preset=preset)


@server.tool()
def collection_queue(top_n: int = 20) -> dict[str, Any]:
    from mlops.serving.predictor import get_predictor

    return {"queue": get_predictor().collection_queue(top_n=top_n)}


@server.tool()
def policy_search(query: str, k: int = 5) -> dict[str, Any]:
    from mlops.agents.rag.retriever import get_retriever

    retriever = get_retriever()
    results = retriever.search(query, k=k)
    return {
        "context": retriever.format_context(results),
        "citations": retriever.citations(results),
    }


@server.tool()
def get_indicators() -> dict[str, Any]:
    latest = load_latest(resolve_paths().project_root / "artifacts" / "autopilot")
    if not latest:
        return {"error": "Nenhum run do autopilot disponível."}
    path = Path(latest["run_dir"]) / "portfolio.json"
    if not path.exists():
        return {"error": "portfolio.json ausente."}
    return json.loads(path.read_text(encoding="utf-8"))


@server.tool()
def judge_case(client_id: str) -> dict[str, Any]:
    import pandas as pd

    path = resolve_paths().serving / "serving_clients.parquet"
    frame = pd.read_parquet(path)
    subset = frame[frame["client_id"].astype(str) == client_id]
    if subset.empty:
        return {"error": f"Cliente não encontrado: {client_id}"}
    return decide_client(
        subset.iloc[0],
        policies=load_policies(),
        jev=get_jev(),
        use_jev=True,
        policy_version="mcp",
        model_version=_model_version(),
    )


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
