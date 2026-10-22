from __future__ import annotations

import json
from pathlib import Path

from mlops.serving.api import app, latest_indicators, root


def test_root_is_machine_api() -> None:
    payload = root()
    assert payload["mode"] == "machine-api"
    assert payload["docs"] == "/docs"


def test_no_dashboard_route() -> None:
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/demo" not in paths
    assert "/v1/indicators/latest" in paths
    assert "/v1/decisions" in paths


def test_latest_indicators(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "portfolio.json").write_text(json.dumps({"n_clientes": 1}), encoding="utf-8")
    (tmp_path / "latest.json").write_text(
        json.dumps({"run_id": "run-1", "run_dir": str(run_dir)}), encoding="utf-8"
    )
    monkeypatch.setenv("MLOPS_AUTOPILOT_OUTPUT", str(tmp_path))
    assert latest_indicators()["n_clientes"] == 1


def test_latest_indicators_resolves_run_from_another_host(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "runs" / "run-2"
    run_dir.mkdir(parents=True)
    (run_dir / "portfolio.json").write_text(json.dumps({"n_clientes": 2}), encoding="utf-8")
    (tmp_path / "latest.json").write_text(
        json.dumps({"run_id": "run-2", "run_dir": "/app/artifacts/autopilot/runs/run-2"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MLOPS_AUTOPILOT_OUTPUT", str(tmp_path))
    assert latest_indicators()["n_clientes"] == 2
