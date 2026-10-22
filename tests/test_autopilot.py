from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlops.autopilot.decisions import decide_client, decisions_for_frame
from mlops.autopilot.indicators import (
    compare_indicators,
    compute_client_indicators,
    compute_portfolio_indicators,
)
from mlops.autopilot.runner import run_autopilot
from mlops.autopilot.sources import validate_client_frame
from mlops.autopilot.storage import upload_run
from mlops.autopilot.store import RunStore
from mlops.evaluation.policy import load_policies


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "client_id": "C000001",
                "segmento": "varejo",
                "regiao": "sudeste",
                "renda_mensal": 5000.0,
                "limite_atual": 3000.0,
                "utilizacao_media": 0.4,
                "comprometimento_consumo": 0.3,
                "p_inadimplente_90d": 0.05,
                "p_pagou_30d": 0.6,
                "p_aceita_limite": 0.5,
                "p_consumo_alto": 0.4,
                "valor_atrasado_cutoff": 200.0,
                "entradas_open_finance_total": 60000.0,
                "tempo_relacionamento_meses": 24,
                "bureau_score": 700,
                "atraso_max_90d": 5,
                "meses_com_atraso": 1,
                "saldo_open_finance_medio": 1000.0,
                "saidas_open_finance_total": 40000.0,
            },
            {
                "client_id": "C000002",
                "segmento": "pj",
                "regiao": "sul",
                "renda_mensal": 2000.0,
                "limite_atual": 5000.0,
                "utilizacao_media": 0.9,
                "comprometimento_consumo": 0.8,
                "p_inadimplente_90d": 0.55,
                "p_pagou_30d": 0.1,
                "p_aceita_limite": 0.2,
                "p_consumo_alto": 0.2,
                "valor_atrasado_cutoff": 2500.0,
                "entradas_open_finance_total": 12000.0,
                "tempo_relacionamento_meses": 8,
                "bureau_score": 350,
                "atraso_max_90d": 80,
                "meses_com_atraso": 5,
                "saldo_open_finance_medio": 0.0,
                "saidas_open_finance_total": 22000.0,
            },
        ]
    )


def test_quality_contract() -> None:
    report = validate_client_frame(_frame())
    assert report.passed is True
    assert report.rows == 2


def test_indicators_and_comparison() -> None:
    policies = load_policies()
    scored = compute_client_indicators(_frame(), policies)
    portfolio = compute_portfolio_indicators(scored, policies)
    assert portfolio["n_clientes"] == 2
    assert portfolio["carteira_total"] == 8000.0
    assert len(portfolio["score_histogram"]) == 10
    comparison = compare_indicators(portfolio, portfolio)
    assert comparison["psi_score"] == 0.0


def test_decide_client_offline() -> None:
    policies = load_policies()
    row = _frame().iloc[0]
    decision = decide_client(row, policies, jev=None, use_jev=False, policy_version="p", model_version="m")
    assert decision["mode"] == "fallback"
    assert decision["risk"]["action"] in {"aprovar", "revisar", "reprovar"}
    assert decision["action_status"] == "ready_for_execution"


def test_decisions_for_frame_parallel() -> None:
    policies = load_policies()
    decisions, stats = decisions_for_frame(
        _frame(),
        policies=policies,
        jev=None,
        use_jev=False,
        max_workers=2,
        max_calls=0,
    )
    assert len(decisions) == 2
    assert stats["calls"] == 0


def test_decisions_count_jev_fallbacks() -> None:
    policies = load_policies()
    decisions, stats = decisions_for_frame(
        _frame(),
        policies=policies,
        jev=None,
        use_jev=True,
        max_workers=2,
        max_calls=10,
    )
    assert len(decisions) == 2
    assert stats["calls"] == 2
    assert stats["fallbacks"] == 2
    assert all(item["mode"] == "fallback" for item in decisions)


def test_run_store_audit_chain(tmp_path: Path) -> None:
    store = RunStore(tmp_path, "run-test")
    store.append_audit({"type": "test", "value": 1})
    store.append_audit({"type": "test", "value": 2})
    lines = (tmp_path / "runs" / "run-test" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert second["prev_hash"] == first["hash"]


def test_run_autopilot_dry_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MLOPS_ARTIFACTS_BUCKET", raising=False)
    monkeypatch.delenv("MLOPS_ARTIFACTS_S3", raising=False)
    result = run_autopilot(limit=3, dry_run=True, mode="full", output_dir=tmp_path)
    run_dir = Path(result["run_dir"])
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "indicators.parquet").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "audit.jsonl").exists()
    assert (tmp_path / "latest.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert manifest["autonomy"] == "full"
    assert result["artifacts_uri"] is None


def test_upload_run_without_destination(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MLOPS_ARTIFACTS_BUCKET", raising=False)
    monkeypatch.delenv("MLOPS_ARTIFACTS_S3", raising=False)
    assert upload_run(tmp_path) is None
