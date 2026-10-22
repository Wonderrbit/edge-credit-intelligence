from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_raw: Path
    data_processed: Path
    knowledge: Path
    artifacts: Path
    models: Path
    edge: Path
    eval: Path
    rag: Path
    serving: Path
    logs: Path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def resolve_paths(config: dict[str, Any] | None = None) -> Paths:
    config = config or load_config()
    root = Path(config.get("project_root", PROJECT_ROOT)).resolve()
    artifacts = root / config.get("artifacts_dir", "artifacts")
    data = root / config.get("data_dir", "data")
    return Paths(
        project_root=root,
        data_raw=data / "raw",
        data_processed=data / "processed",
        knowledge=data / "knowledge",
        artifacts=artifacts,
        models=artifacts / "models",
        edge=artifacts / "edge",
        eval=artifacts / "eval",
        rag=artifacts / "rag",
        serving=artifacts / "serving",
        logs=artifacts / "logs",
    )


def ensure_dirs(paths: Paths | None = None) -> Paths:
    paths = paths or resolve_paths()
    for directory in (
        paths.data_raw,
        paths.data_processed,
        paths.knowledge,
        paths.artifacts,
        paths.models,
        paths.edge,
        paths.eval,
        paths.rag,
        paths.serving,
        paths.logs,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass
