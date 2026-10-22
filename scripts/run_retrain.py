from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.pipelines.retrain import retrain_if_needed


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia monitoramento e retreina se necessário.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    decision = retrain_if_needed(force=args.force, fast=args.fast)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
