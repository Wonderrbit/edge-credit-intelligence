from __future__ import annotations

import runpy
from pathlib import Path

REQUIRED_ARTIFACTS = (
    "registry.json",
    "models/edge_credit_multitask.keras",
    "models/calibration.json",
    "edge/preprocessing.json",
    "serving/serving_clients.parquet",
    "serving/serving_sequences.npz",
    "serving/client_drivers.json",
    "eval/metrics.json",
    "eval/model_metrics.json",
    "rag/chunks.json",
    "rag/index_meta.json",
    "rag/bm25.joblib",
    "rag/tfidf_matrix.joblib",
    "rag/tfidf_vectorizer.joblib",
)


def pytest_sessionstart(session) -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts"
    if all((artifacts / relative).exists() for relative in REQUIRED_ARTIFACTS):
        return
    namespace = runpy.run_path(
        str(root / "scripts" / "make_demo_assets.py"), run_name="demo_assets"
    )
    namespace["make_demo_assets"]()
