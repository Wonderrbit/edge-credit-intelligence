"""Adapters for real data sources (Parquet, CSV, SQL, Open Finance, bureau)."""

from mlops.data.adapters.base import CSVAdapter, ParquetAdapter

__all__ = ["CSVAdapter", "ParquetAdapter"]
