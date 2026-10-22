from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd


class DataAdapter(Protocol):
    def read(self, name: str) -> pd.DataFrame: ...


class ParquetAdapter:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.base_dir / f"{name}.parquet")


class CSVAdapter:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.base_dir / f"{name}.csv")
