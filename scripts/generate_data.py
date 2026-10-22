from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.config import load_config, set_seed
from mlops.data.quality import data_quality_report
from mlops.data.synthetic import generate_synthetic_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera a base sintética de Open Finance.")
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--months", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    data_config = config["data"]
    n_clients = args.clients or int(data_config["n_clients"])
    months = args.months or int(data_config["months"])
    seed = args.seed if args.seed is not None else int(data_config["seed"])
    set_seed(seed)

    tables = generate_synthetic_data(
        n_clients=n_clients,
        months=months,
        seed=seed,
        feature_months=int(data_config["feature_months"]),
        performance_months=int(data_config["performance_months"]),
        safras=tuple(data_config["safras"]),
    )
    report = data_quality_report(tables, "artifacts/eval/data_quality.json")
    print(f"Clientes: {n_clients:,} | Meses: {months:,} | Safras: {data_config['safras']}")
    for name, frame in tables.items():
        print(f"  {name}: {len(frame):,} linhas x {len(frame.columns)} colunas")
    print(f"Qualidade: {'OK' if report['passed'] else 'FALHOU'}")


if __name__ == "__main__":
    main()
