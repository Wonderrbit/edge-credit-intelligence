from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.data.features import build_datasets
from mlops.data.lineage import build_lineage


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrói features tabulares e sequenciais.")
    parser.add_argument("--train-safras", default="1,2")
    parser.add_argument("--validation-safras", default="3,4")
    parser.add_argument("--test-safras", default="5,6")
    args = parser.parse_args()

    train = tuple(int(value) for value in args.train_safras.split(","))
    validation = tuple(int(value) for value in args.validation_safras.split(","))
    test = tuple(int(value) for value in args.test_safras.split(","))
    datasets = build_datasets(
        train_safras=train,
        validation_safras=validation,
        test_safras=test,
    )
    build_lineage(
        sources={
            "clientes": "data/raw/clientes.parquet",
            "painel": "data/raw/painel.parquet",
            "targets": "data/raw/targets.parquet",
        },
        transformations=[
            {"name": "sql/10_features_consumo.sql", "type": "sql"},
            {"name": "sql/30_features_cobranca.sql", "type": "sql"},
            {"name": "edge_preprocessing", "type": "standard_scaler_onehot"},
        ],
        output_path="artifacts/eval/lineage.json",
    )
    print(f"Features tabulares: {datasets['tabular'].shape}")
    print(f"Sequência Edge: {datasets['seq'].shape} | Estáticos: {datasets['static'].shape}")
    print(
        "Split:",
        json.dumps(
            {
                split: int((datasets["split"] == split).sum())
                for split in ("train", "validation", "test")
            }
        ),
    )


if __name__ == "__main__":
    main()
