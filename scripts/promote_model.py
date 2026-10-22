from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.pipelines.promote import promote_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Promove uma versão do registro.")
    parser.add_argument("--version", default=None)
    parser.add_argument("--stage", default="production")
    parser.add_argument("--actor", default="mlops")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload = promote_model(
        version=args.version,
        stage=args.stage,
        actor=args.actor,
        force=args.force,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
