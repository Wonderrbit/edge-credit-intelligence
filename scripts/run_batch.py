from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.pipelines.batch import run_batch_scoring


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa batch scoring.")
    parser.add_argument("--input", default="artifacts/serving/serving_clients.parquet")
    parser.add_argument("--output", default=None)
    parser.add_argument("--preset", default="equilibrado", choices=["conservador", "equilibrado", "crescimento"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = run_batch_scoring(args.input, args.output, preset=args.preset, limit=args.limit)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
