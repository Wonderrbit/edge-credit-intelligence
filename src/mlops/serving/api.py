from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mlops.agents.llm import get_llm
from mlops.config import resolve_paths
from mlops.serving.predictor import CreditPredictor, PredictorError, get_predictor
from mlops.serving.schemas import (
    AgentRequest,
    AgentResponse,
    BatchScoreRequest,
    ClientSummary,
    HealthResponse,
    ScoreFeaturesRequest,
    ScoreResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = get_predictor()
    except Exception as exc:
        app.state.predictor_error = str(exc)
        app.state.predictor = None
    yield


app = FastAPI(
    title="Akropoli Decision Copilot",
    description=(
        "API de scoring de crédito, limite e cobrança com agente LangGraph + RAG. "
        "Projeto de demonstração com dados sintéticos."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = os.getenv("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="API key inválida.")


def predictor() -> CreditPredictor:
    instance = getattr(app.state, "predictor", None)
    if instance is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(
                app.state,
                "predictor_error",
                "Modelo não carregado. Execute o pipeline de treino.",
            ),
        )
    return instance


@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {
        "service": "Akropoli Decision Copilot",
        "mode": "machine-api",
        "docs": "/docs",
        "health": "/health",
        "autopilot": "/v1/indicators/latest",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    instance = getattr(app.state, "predictor", None)
    return HealthResponse(
        status="ok" if instance else "degraded",
        model_version=instance.registry.get("version") if instance else None,
        agent_mode=get_llm().mode,
    )


@app.get("/model/info", dependencies=[Depends(require_api_key)])
def model_info(instance: CreditPredictor = Depends(predictor)) -> dict[str, Any]:
    return instance.model_info()


@app.get("/clients", response_model=list[ClientSummary], dependencies=[Depends(require_api_key)])
def clients(
    limit: int = Query(default=50, ge=1, le=500),
    instance: CreditPredictor = Depends(predictor),
) -> list[dict[str, Any]]:
    return instance.list_clients(limit=limit)


@app.get("/score/{client_id}", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
def score_client(
    client_id: str,
    preset: str = Query(default="equilibrado", pattern="^(conservador|equilibrado|crescimento)$"),
    instance: CreditPredictor = Depends(predictor),
) -> dict[str, Any]:
    try:
        return instance.score_client(client_id, preset=preset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/score", dependencies=[Depends(require_api_key)])
def score_features(
    request: ScoreFeaturesRequest,
    instance: CreditPredictor = Depends(predictor),
) -> dict[str, Any]:
    try:
        return instance.score_features(request.sequence, request.static)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/score/batch", response_model=list[ScoreResponse], dependencies=[Depends(require_api_key)])
def batch_score(
    request: BatchScoreRequest,
    instance: CreditPredictor = Depends(predictor),
) -> list[dict[str, Any]]:
    return instance.batch_score(request.client_ids, preset=request.preset)


@app.get("/collection/queue", dependencies=[Depends(require_api_key)])
def collection_queue(
    top_n: int = Query(default=20, ge=1, le=200),
    instance: CreditPredictor = Depends(predictor),
) -> dict[str, Any]:
    return {"queue": instance.collection_queue(top_n=top_n)}


@app.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics(instance: CreditPredictor = Depends(predictor)) -> dict[str, Any]:
    return {
        "edge_metrics": instance.model_metrics.get("edge_metrics", {}),
        "calibration": instance.model_metrics.get("calibration", {}),
        "metrics": instance.metrics,
    }


@app.get("/monitoring", dependencies=[Depends(require_api_key)])
def monitoring(instance: CreditPredictor = Depends(predictor)) -> dict[str, Any]:
    return instance.monitoring_report()


@app.post("/agent/ask", response_model=AgentResponse, dependencies=[Depends(require_api_key)])
def agent_ask(request: AgentRequest) -> dict[str, Any]:
    from mlops.agents.graph import run_agent

    try:
        return run_agent(
            request.question,
            thread_id=request.thread_id,
            role=request.role,
            client_id=request.client_id,
        )
    except PredictorError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _autopilot_root() -> Path:
    configured = os.getenv("MLOPS_AUTOPILOT_OUTPUT", "artifacts/autopilot")
    path = Path(configured)
    return path if path.is_absolute() else resolve_paths().project_root / path


def _latest_run_dir() -> Path:
    pointer = _autopilot_root() / "latest.json"
    if not pointer.exists():
        raise HTTPException(status_code=404, detail="Nenhum run do autopilot encontrado.")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    raw_run_dir = payload.get("run_dir", "")
    run_dir = Path(raw_run_dir)
    if not run_dir.is_absolute():
        run_dir = _autopilot_root() / run_dir
    if not run_dir.exists():
        run_name = payload.get("run_id") or Path(raw_run_dir).name
        run_dir = _autopilot_root() / "runs" / str(run_name)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run apontado não existe mais.")
    return run_dir


@app.get("/v1/indicators/latest", dependencies=[Depends(require_api_key)])
def latest_indicators() -> dict[str, Any]:
    run_dir = _latest_run_dir()
    return json.loads((run_dir / "portfolio.json").read_text(encoding="utf-8"))


@app.get("/v1/decisions", dependencies=[Depends(require_api_key)])
def latest_decisions(
    limit: int = Query(default=100, ge=1, le=5000),
    adverse_only: bool = Query(default=False),
) -> dict[str, Any]:
    run_dir = _latest_run_dir()
    rows = []
    path = run_dir / "decisions.jsonl"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if adverse_only and not item.get("adverse_action"):
                    continue
                rows.append(item)
                if len(rows) >= limit:
                    break
    return {"run_dir": str(run_dir), "count": len(rows), "decisions": rows}


@app.get("/v1/runs", dependencies=[Depends(require_api_key)])
def list_runs(limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    root = _autopilot_root() / "runs"
    if not root.exists():
        return {"runs": []}
    runs = sorted((path for path in root.iterdir() if path.is_dir()), reverse=True)[:limit]
    return {
        "runs": [
            {
                "run_id": path.name,
                "run_dir": str(path),
                "has_manifest": (path / "manifest.json").exists(),
            }
            for path in runs
        ]
    }


@app.get("/v1/runs/{run_id}", dependencies=[Depends(require_api_key)])
def get_run(run_id: str) -> dict[str, Any]:
    path = _autopilot_root() / "runs" / run_id / "manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run não encontrado.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/v1/autopilot/run", dependencies=[Depends(require_api_key)])
async def run_autopilot_endpoint(
    limit: int | None = Query(default=None, ge=1, le=100000),
    dry_run: bool = Query(default=False),
    mode: str = Query(default="full", pattern="^(full|indicators|judge|report)$"),
) -> dict[str, Any]:
    from mlops.autopilot.runner import run_autopilot

    return await run_in_threadpool(
        run_autopilot,
        limit=limit,
        dry_run=dry_run,
        mode=mode,
    )


paths = resolve_paths()
if paths.edge.exists():
    app.mount("/edge", StaticFiles(directory=str(paths.edge), html=True), name="edge")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "mlops.serving.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()
