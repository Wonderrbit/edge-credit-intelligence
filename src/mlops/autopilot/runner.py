from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mlops.autopilot.alerts import evaluate_alerts, send_webhook
from mlops.autopilot.decisions import decisions_for_frame
from mlops.autopilot.indicators import (
    compare_indicators,
    compute_client_indicators,
    compute_portfolio_indicators,
    load_indicator_registry,
)
from mlops.autopilot.narrative import generate_narrative
from mlops.autopilot.report import build_markdown
from mlops.autopilot.sources import default_clients_source, validate_client_frame
from mlops.autopilot.storage import upload_file, upload_run
from mlops.autopilot.store import RunStore, load_latest, load_latest_portfolio, utc_now
from mlops.config import PROJECT_ROOT, ensure_dirs, resolve_paths
from mlops.evaluation.policy import load_policies
from mlops.judgment import get_jev

OUTPUT_COLUMNS = [
    "client_id",
    "segmento",
    "regiao",
    "renda_mensal",
    "limite_atual",
    "pd_90d",
    "p_pagamento_30d",
    "exposicao",
    "perda_esperada",
    "valor_em_atraso",
    "ev_cobranca",
    "utilizacao",
    "comprometimento_renda",
]


def load_autopilot_config(path: str | Path | None = None) -> dict[str, Any]:
    destination = Path(path) if path else PROJECT_ROOT / "configs" / "autopilot.yaml"
    payload = yaml.safe_load(destination.read_text(encoding="utf-8")) or {}
    config = dict(payload.get("autopilot", payload))
    if os.getenv("MLOPS_AUTOPILOT_OUTPUT"):
        config["output_dir"] = os.environ["MLOPS_AUTOPILOT_OUTPUT"]
    if os.getenv("MLOPS_AUTOPILOT_MAX_WORKERS"):
        config["max_workers"] = int(os.environ["MLOPS_AUTOPILOT_MAX_WORKERS"])
    if os.getenv("MLOPS_USE_JEV"):
        config["use_jev"] = os.environ["MLOPS_USE_JEV"] == "1"
    if os.getenv("MLOPS_JEV_MAX_CALLS_PER_RUN"):
        config.setdefault("jev", {})["max_calls_per_run"] = int(
            os.environ["MLOPS_JEV_MAX_CALLS_PER_RUN"]
        )
    return config


def _policy_version() -> str:
    paths = [PROJECT_ROOT / "configs" / "policies.yaml", PROJECT_ROOT / "configs" / "judgment.yaml"]
    digest = hashlib.sha256()
    for path in paths:
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _model_version() -> str | None:
    path = resolve_paths().artifacts / "registry.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("version")
    except json.JSONDecodeError:
        return None


def _estimate_cost(decisions: list[dict[str, Any]], config: dict[str, Any]) -> tuple[int, float]:
    jev = config.get("jev", {})
    actual_tokens = sum(
        int(decision.get("jev", {}).get("input_tokens", 0) or 0) for decision in decisions
    )
    calls = sum(1 for decision in decisions if decision.get("mode") == "jev")
    estimated_tokens = actual_tokens or calls * int(jev.get("estimated_input_tokens_per_call", 1800))
    price = float(jev.get("price_per_million_input_tokens", 0.042))
    return estimated_tokens, round(estimated_tokens / 1_000_000 * price, 6)


def run_autopilot(
    limit: int | None = None,
    dry_run: bool = False,
    mode: str = "full",
    output_dir: str | Path | None = None,
    use_jev: bool | None = None,
    max_workers: int | None = None,
    max_calls: int | None = None,
) -> dict[str, Any]:
    if mode not in {"full", "indicators", "judge", "report"}:
        raise ValueError("mode deve ser full, indicators, judge ou report")
    started = time.perf_counter()
    started_at = utc_now()
    config = load_autopilot_config()
    root = Path(output_dir) if output_dir else PROJECT_ROOT / str(
        config.get("output_dir", "artifacts/autopilot")
    )
    root = root if root.is_absolute() else PROJECT_ROOT / root
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    store = RunStore(root, run_id)

    paths = ensure_dirs(resolve_paths())
    frame = default_clients_source().fetch()
    source_rows = len(frame)
    if limit is not None:
        frame = frame.head(limit).copy()
    quality = validate_client_frame(frame)
    policies = load_policies()
    scored = compute_client_indicators(frame, policies)
    portfolio = compute_portfolio_indicators(scored, policies)
    latest = load_latest(root)
    comparable = (
        limit is None
        and latest is not None
        and latest.get("source_rows") == source_rows
        and latest.get("rows_processed") == len(scored)
    )
    previous = load_latest_portfolio(root) if comparable else None
    comparison = compare_indicators(portfolio, previous)
    portfolio["comparison"] = comparison

    decisions: list[dict[str, Any]] = []
    jev_stats = {"calls": 0, "fallbacks": 0, "max_calls": 0}
    if mode in {"full", "judge"}:
        configured_use_jev = bool(config.get("use_jev", True))
        final_use_jev = configured_use_jev if use_jev is None else use_jev
        if dry_run:
            final_use_jev = False
        final_max_calls = (
            max_calls
            if max_calls is not None
            else int(config.get("jev", {}).get("max_calls_per_run", 500))
        )
        final_workers = max_workers or int(config.get("max_workers", 8))
        jev = get_jev() if final_use_jev else None
        decisions, jev_stats = decisions_for_frame(
            scored,
            policies=policies,
            jev=jev,
            use_jev=final_use_jev,
            max_workers=final_workers,
            max_calls=final_max_calls,
            model_version=_model_version(),
        )
        if dry_run:
            jev_stats["fallbacks"] = 0

    tokens, cost = _estimate_cost(decisions, config)
    alerts = evaluate_alerts(
        current=portfolio,
        comparison=comparison,
        config=config.get("alerts", {}),
        jev_stats=jev_stats,
        cost_usd=cost,
        quality=quality.model_dump(),
    )
    review_queue = [
        {
            "decision_id": item["decision_id"],
            "client_id": item["client_id"],
            "risk_action": item["risk"]["action"],
            "risk_reasons": item["risk"]["reasons"],
            "manual_review_recommended": item["manual_review_recommended"],
            "adverse_action": item["adverse_action"],
            "gate": "informational",
        }
        for item in decisions
        if item["manual_review_recommended"] or item["adverse_action"]
    ]
    finished_at = utc_now()
    manifest = {
        "run_id": run_id,
        "run_dir": str(store.run_dir),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "mode": mode,
        "dry_run": dry_run,
        "autonomy": os.getenv("MLOPS_AUTONOMY", config.get("autonomy", "full")),
        "judgment_mode": "jev"
        if int(jev_stats.get("calls", 0)) > int(jev_stats.get("fallbacks", 0))
        else "offline",
        "source": "serving_clients.parquet",
        "source_rows": source_rows,
        "rows_processed": int(len(scored)),
        "limit": limit,
        "comparison_baseline": "previous" if previous else "none",
        "n_decisions": len(decisions),
        "n_adverse": sum(1 for item in decisions if item["adverse_action"]),
        "n_review_recommended": len(review_queue),
        "model_version": _model_version(),
        "policy_version": _policy_version(),
        "jev_calls": int(jev_stats.get("calls", 0)),
        "jev_fallbacks": int(jev_stats.get("fallbacks", 0)),
        "jev_success": max(
            int(jev_stats.get("calls", 0)) - int(jev_stats.get("fallbacks", 0)), 0
        ),
        "estimated_input_tokens": tokens,
        "estimated_cost_usd": cost,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "indicator_registry": load_indicator_registry(),
        "alerts": alerts,
        "paths": paths.artifacts.as_posix(),
    }
    narrative = generate_narrative(portfolio, alerts, manifest)
    if narrative is not None:
        manifest["narrative"] = narrative.model_dump()

    report = build_markdown(
        manifest=manifest,
        portfolio=portfolio,
        comparison=comparison,
        decisions=decisions,
        quality=quality.model_dump(),
        alerts=alerts,
        jev_stats=jev_stats,
    )

    store.write_parquet("indicators.parquet", scored[OUTPUT_COLUMNS])
    store.write_json("portfolio.json", portfolio)
    store.write_json("comparison.json", comparison)
    store.write_json("data_quality.json", quality.model_dump())
    store.write_jsonl("decisions.jsonl", decisions)
    store.write_jsonl("review_queue.jsonl", review_queue)
    store.write_json("alerts.json", alerts)
    store.write_text("report.md", report)
    store.write_json("manifest.json", manifest)
    store.append_audit(
        {
            "type": "run",
            "event": "started",
            "run_id": run_id,
            "started_at": started_at,
            "mode": mode,
            "dry_run": dry_run,
        }
    )
    for decision in decisions:
        store.append_audit(
            {
                "type": "decision",
                "event": "evaluated",
                "run_id": run_id,
                "decision_id": decision["decision_id"],
                "client_id": decision["client_id"],
                "risk_action": decision["risk"]["action"],
                "collection_action": decision["collection"]["action"],
                "adverse_action": decision["adverse_action"],
                "mode": decision["mode"],
                "policy_version": decision["policy_version"],
                "model_version": decision["model_version"],
            }
        )
    store.append_audit(
        {
            "type": "run",
            "event": "finished",
            "run_id": run_id,
            "finished_at": finished_at,
            "alerts": len(alerts),
        }
    )
    store.update_latest(
        {
            "run_id": run_id,
            "run_dir": str(store.run_dir),
            "finished_at": finished_at,
            "n_decisions": len(decisions),
            "source_rows": source_rows,
            "rows_processed": int(len(scored)),
            "limit": limit,
            "autonomy": manifest["autonomy"],
            "alerts": alerts,
        }
    )
    artifacts_uri = upload_run(store.run_dir)
    if artifacts_uri:
        manifest["artifacts_uri"] = artifacts_uri
        manifest_path = store.write_json("manifest.json", manifest)
        upload_file(manifest_path, f"{artifacts_uri}/manifest.json")

    if alerts and not dry_run:
        send_webhook(
            os.getenv(config.get("alert_webhook_env", "MLOPS_ALERT_WEBHOOK"))
            or os.getenv("MLOPS_ALERT_WEBHOOK"),
            {
                "text": f"Autopilot {run_id}: {len(alerts)} alerta(s).",
                "run_id": run_id,
                "alerts": alerts,
                "report": str(store.run_dir / "report.md"),
            },
        )

    return {
        "run_id": run_id,
        "run_dir": str(store.run_dir),
        "artifacts_uri": artifacts_uri,
        "manifest": manifest,
        "alerts": alerts,
        "decisions": len(decisions),
        "review_queue": len(review_queue),
        "report": str(store.run_dir / "report.md"),
    }
