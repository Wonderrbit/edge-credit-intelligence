from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


class RunStore:
    def __init__(self, root: str | Path, run_id: str):
        self.root = Path(root)
        self.run_id = run_id
        self.run_dir = self.root / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._last_hash = ""

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        path.write_text(
            json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return path

    def write_text(self, name: str, content: str) -> Path:
        path = self.run_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def write_parquet(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.run_dir / name
        frame.to_parquet(path, index=False)
        return path

    def write_jsonl(self, name: str, rows: list[dict[str, Any]]) -> Path:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(to_jsonable(row), ensure_ascii=False, allow_nan=False))
                handle.write("\n")
        return path

    def append_audit(self, entry: dict[str, Any]) -> Path:
        path = self.run_dir / "audit.jsonl"
        payload = dict(entry)
        payload["prev_hash"] = self._last_hash
        payload["hash"] = hashlib.sha256(
            json.dumps(to_jsonable(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self._last_hash = payload["hash"]
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(payload), ensure_ascii=False, allow_nan=False))
            handle.write("\n")
        return path

    def update_latest(self, payload: dict[str, Any]) -> Path:
        path = self.root / "latest.json"
        path.write_text(
            json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return path


def load_latest(root: str | Path) -> dict[str, Any] | None:
    path = Path(root) / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_latest_portfolio(root: str | Path) -> dict[str, Any] | None:
    latest = load_latest(root)
    if not latest:
        return None
    portfolio_path = Path(latest.get("run_dir", "")) / "portfolio.json"
    if not portfolio_path.exists():
        return None
    try:
        return json.loads(portfolio_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
