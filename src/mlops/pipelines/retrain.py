from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlops.config import ensure_dirs, resolve_paths
from mlops.pipelines.monitoring import run_monitoring
from mlops.pipelines.train import run_training


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _auc_summary(registry: dict[str, Any]) -> float:
    metrics = registry.get("edge_metrics", {})
    if not metrics:
        return 0.0
    return sum(float(values.get("roc_auc", 0.0)) for values in metrics.values()) / len(metrics)


def retrain_if_needed(force: bool = False, fast: bool = False) -> dict[str, Any]:
    paths = ensure_dirs(resolve_paths())
    monitoring = run_monitoring()
    current = _read_json(paths.artifacts / "registry.json", {})
    current_auc = _auc_summary(current)
    should_retrain = force or monitoring["status"] == "retrain"

    decision: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "monitoring_status": monitoring["status"],
        "current_version": current.get("version"),
        "current_mean_auc": round(current_auc, 6),
        "retrained": False,
        "promote": False,
    }
    if not should_retrain:
        decision["reason"] = "Monitoramento não indicou necessidade de retreinamento."
        (paths.eval / "retrain_decision.json").write_text(
            json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return decision

    if current:
        (paths.artifacts / "registry_previous.json").write_text(
            json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    result = run_training(fast=fast, force_data=False)
    new_registry = result["registry"]
    new_auc = _auc_summary(new_registry)
    decision.update(
        {
            "retrained": True,
            "new_version": new_registry.get("version"),
            "new_mean_auc": round(new_auc, 6),
            "delta_auc": round(new_auc - current_auc, 6),
            "promote": bool(new_auc >= current_auc + 0.005),
            "reason": (
                "Ganho de AUC acima de 0,005; elegível para shadow deployment."
                if new_auc >= current_auc + 0.005
                else "Ganho insuficiente; manter campeão atual."
            ),
        }
    )
    (paths.eval / "retrain_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return decision
