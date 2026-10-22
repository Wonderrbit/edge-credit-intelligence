from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.config import resolve_paths

REQUIRED_FILES = (
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


def make_demo_assets(force: bool = False) -> dict[str, object]:
    paths = resolve_paths()
    source_root = paths.project_root / "demo_assets"
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Pacote de demo ausente: {source_root}. "
            "Restaure demo_assets/ ou execute o pipeline completo."
        )

    copied = 0
    skipped = 0
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        destination = paths.artifacts / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not force:
            skipped += 1
            continue
        shutil.copy2(source, destination)
        copied += 1

    missing = [
        str(paths.artifacts / relative)
        for relative in REQUIRED_FILES
        if not (paths.artifacts / relative).exists()
    ]
    if missing:
        raise RuntimeError(
            "Pacote de demo incompleto. Arquivos ausentes: " + ", ".join(missing)
        )

    return {
        "source": str(source_root),
        "destination": str(paths.artifacts),
        "copied": copied,
        "skipped": skipped,
        "created": copied > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hidrata artifacts/ a partir do pacote demo_assets/."
    )
    parser.add_argument("--force", action="store_true", help="Sobrescreve os arquivos de demo.")
    args = parser.parse_args()
    print(json.dumps(make_demo_assets(force=args.force), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
