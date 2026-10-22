from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.autopilot.runner import run_autopilot


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o autopilot autônomo de crédito e cobrança.")
    parser.add_argument("--mode", default="full", choices=["full", "indicators", "judge", "report"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Não chama Jev e não envia webhook.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-jev", dest="use_jev", action="store_true", default=None)
    parser.add_argument("--no-jev", dest="use_jev", action="store_false")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    args = parser.parse_args()
    result = run_autopilot(
        limit=args.limit,
        dry_run=args.dry_run,
        mode=args.mode,
        output_dir=args.output_dir,
        use_jev=args.use_jev,
        max_workers=args.max_workers,
        max_calls=args.max_calls,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
