from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.pipelines.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o pipeline de treinamento.")
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--months", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    result = run_training(
        n_clients=args.clients,
        months=args.months,
        seed=args.seed,
        fast=args.fast,
        force_data=args.force_data,
        enable_mlflow=args.mlflow,
    )
    metrics = result["metrics"]
    print("Treinamento concluído")
    for target, values in metrics["edge_calibrated"].items():
        print(
            f"  {target}: AUC={values.get('roc_auc', 0):.3f} "
            f"KS={values.get('ks', 0):.3f} Brier={values.get('brier', 0):.3f}"
        )
    print(f"  Modelo: {result['registry']['model_path']}")
    print("  Métricas: artifacts/eval/metrics.json")


if __name__ == "__main__":
    main()
