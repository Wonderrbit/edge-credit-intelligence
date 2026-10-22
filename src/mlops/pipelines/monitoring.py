from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mlops.config import ensure_dirs, resolve_paths


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def run_monitoring() -> dict[str, Any]:
    paths = ensure_dirs(resolve_paths())
    model_metrics = _read_json(paths.eval / "model_metrics.json", {})
    metrics = _read_json(paths.eval / "metrics.json", {})
    registry = _read_json(paths.artifacts / "registry.json", {})
    quality = _read_json(paths.eval / "data_quality.json", {})

    stability = metrics.get("stability", {})
    feature_psi = stability.get("features", [])
    score_psi = stability.get("score", {})
    max_feature_psi = max((row.get("psi", 0.0) for row in feature_psi), default=0.0)
    score_psi_value = float(score_psi.get("psi", 0.0)) if isinstance(score_psi, dict) else 0.0

    edge_metrics = model_metrics.get("edge_metrics", {})
    calibration = model_metrics.get("calibration", {})
    alerts: list[dict[str, Any]] = []
    if max_feature_psi > 0.20:
        alerts.append(
            {
                "severity": "high",
                "type": "feature_drift",
                "message": f"PSI máximo de feature {max_feature_psi:.3f} acima de 0,20.",
            }
        )
    elif max_feature_psi > 0.10:
        alerts.append(
            {
                "severity": "medium",
                "type": "feature_drift",
                "message": f"PSI máximo de feature {max_feature_psi:.3f} entre 0,10 e 0,20.",
            }
        )
    if score_psi_value > 0.20:
        alerts.append(
            {
                "severity": "high",
                "type": "score_drift",
                "message": f"PSI de score {score_psi_value:.3f} acima de 0,20.",
            }
        )
    for target, values in edge_metrics.items():
        auc = float(values.get("roc_auc", 0.0))
        if auc and auc < 0.65:
            alerts.append(
                {
                    "severity": "medium",
                    "type": "performance",
                    "message": f"AUC de {target} em {auc:.3f} abaixo do piso de 0,65.",
                }
            )
        ece = float(calibration.get(target, {}).get("ece_after", 0.0))
        if ece > 0.10:
            alerts.append(
                {
                    "severity": "medium",
                    "type": "calibration",
                    "message": f"ECE de {target} em {ece:.3f} acima de 0,10.",
                }
            )
    status = "ok"
    if any(alert["severity"] == "high" for alert in alerts):
        status = "retrain"
    elif alerts:
        status = "attention"

    serving_path = paths.serving / "serving_clients.parquet"
    serving_summary: dict[str, Any] = {"n": 0}
    if serving_path.exists():
        frame = pd.read_parquet(serving_path)
        serving_summary = {
            "n": int(len(frame)),
            "mean_p_default": float(frame["p_inadimplente_90d"].mean()) if "p_inadimplente_90d" in frame else None,
            "mean_p_pay": float(frame["p_pagou_30d"].mean()) if "p_pagou_30d" in frame else None,
            "mean_limit": float(frame["limite_atual"].mean()) if "limite_atual" in frame else None,
        }

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "model_version": registry.get("version"),
        "alerts": alerts,
        "drift": {
            "max_feature_psi": round(max_feature_psi, 4),
            "score_psi": score_psi_value,
            "top_features": sorted(feature_psi, key=lambda row: row.get("psi", 0), reverse=True)[:8],
        },
        "performance": edge_metrics,
        "calibration": {
            target: {
                "ece_after": values.get("ece_after"),
                "brier_after": values.get("brier_after"),
            }
            for target, values in calibration.items()
        },
        "data_quality": {
            "status": quality.get("status"),
            "checks": quality.get("checks"),
        },
        "serving": serving_summary,
        "recommended_action": (
            "Executar retreinamento e shadow deployment."
            if status == "retrain"
            else "Manter monitoramento e investigar os alertas."
            if status == "attention"
            else "Nenhuma ação imediata."
        ),
    }
    output = paths.eval / "monitoring_report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report
