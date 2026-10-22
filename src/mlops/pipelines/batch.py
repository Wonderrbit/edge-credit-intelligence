from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mlops.config import ensure_dirs, resolve_paths
from mlops.serving.predictor import get_predictor


def run_batch_scoring(
    input_path: str | Path,
    output_path: str | Path | None = None,
    preset: str = "equilibrado",
    limit: int | None = None,
) -> dict[str, Any]:
    paths = ensure_dirs(resolve_paths())
    frame = pd.read_parquet(input_path) if str(input_path).endswith(".parquet") else pd.read_csv(input_path)
    if "client_id" not in frame.columns:
        raise ValueError("A entrada precisa conter a coluna client_id.")
    client_ids = frame["client_id"].astype(str).tolist()
    if limit:
        client_ids = client_ids[:limit]
    predictor = get_predictor()
    rows = predictor.batch_score(client_ids, preset=preset)
    output = pd.DataFrame(rows)
    if "scores" in output.columns:
        for target in ("pagou_30d", "inadimplente_90d", "aceita_limite", "consumo_alto"):
            output[f"p_{target}"] = output["scores"].apply(
                lambda value, target=target: value.get(target)
            )
    destination = Path(output_path) if output_path else paths.serving / "batch_scores.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".csv":
        output.to_csv(destination, index=False)
    else:
        output.to_parquet(destination, index=False)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input": str(input_path),
        "output": str(destination),
        "n": len(rows),
        "preset": preset,
        "columns": list(output.columns),
        "mean_scores": {
            key: float(output[key].mean())
            for key in ("p_pagou_30d", "p_inadimplente_90d", "p_aceita_limite", "p_consumo_alto")
            if key in output
        },
    }
    (paths.eval / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
