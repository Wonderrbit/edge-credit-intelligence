from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.pipelines.export_tfjs import export_tfjs


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta o modelo Edge para TensorFlow.js.")
    parser.add_argument("--quantization", choices=["float16", "none"], default="float16")
    args = parser.parse_args()
    info = export_tfjs(quantization=args.quantization)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
