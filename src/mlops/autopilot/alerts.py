from __future__ import annotations

import json
import urllib.request
from typing import Any


def _alert(code: str, severity: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, **extra}


def evaluate_alerts(
    current: dict[str, Any],
    comparison: dict[str, Any],
    config: dict[str, Any],
    jev_stats: dict[str, int],
    cost_usd: float,
    quality: dict[str, Any],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if not quality.get("passed", True):
        alerts.append(
            _alert(
                "data_quality",
                "critical",
                "Contratos de dados falharam; decisões podem estar comprometidas.",
            )
        )
    deltas = comparison.get("deltas", {})
    delinquency_delta = float(config.get("delinquency_delta", 0.02))
    if float(deltas.get("taxa_atraso", 0.0)) >= delinquency_delta:
        alerts.append(
            _alert(
                "delinquency_delta",
                "high",
                f"Taxa de atraso subiu {deltas['taxa_atraso'] * 100:.2f} p.p. vs. run anterior.",
            )
        )
    psi = float(comparison.get("psi_score", 0.0))
    if psi >= float(config.get("psi_threshold", 0.25)):
        alerts.append(
            _alert("psi_score", "high", f"PSI da distribuição de PD em {psi:.3f} (threshold).")
        )
    approval_drop = float(deltas.get("taxa_aprovacao", 0.0))
    if approval_drop <= -float(config.get("approval_rate_drop", 0.10)):
        alerts.append(
            _alert(
                "approval_rate_drop",
                "medium",
                f"Taxa de aprovação caiu {abs(approval_drop) * 100:.2f} p.p. vs. run anterior.",
            )
        )
    calls = int(jev_stats.get("calls", 0))
    fallbacks = int(jev_stats.get("fallbacks", 0))
    total = max(calls, fallbacks)
    if total and fallbacks / total > float(config.get("jev_error_rate", 0.05)):
        alerts.append(
            _alert(
                "jev_error_rate",
                "medium",
                f"{fallbacks}/{total} decisões usaram fallback determinístico.",
            )
        )
    if cost_usd > float(config.get("max_cost_usd", 5.0)):
        alerts.append(
            _alert("cost", "medium", f"Custo estimado do run (US$ {cost_usd:.4f}) acima do budget.")
        )
    return alerts


def send_webhook(url: str | None, payload: dict[str, Any], timeout: float = 10.0) -> bool:
    if not url:
        return False
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return 200 <= response.status < 300
    except Exception:
        return False
