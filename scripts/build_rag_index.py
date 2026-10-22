from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.agents.rag.ingest import build_index
from mlops.config import ensure_dirs, resolve_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrói o índice híbrido de políticas.")
    parser.add_argument("--embeddings", action="store_true", help="Usa embeddings multilíngues.")
    parser.add_argument("--knowledge-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    paths = ensure_dirs(resolve_paths())
    use_embeddings = args.embeddings or os.getenv("MLOPS_USE_EMBEDDINGS", "0") == "1"
    meta = build_index(
        args.knowledge_dir or paths.knowledge,
        args.output_dir or paths.rag,
        use_embeddings=use_embeddings,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
