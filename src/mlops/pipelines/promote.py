from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from mlops.config import ensure_dirs, resolve_paths


def promote_model(
    version: str | None = None,
    stage: str = "production",
    actor: str = "mlops",
    force: bool = False,
) -> dict[str, Any]:
    paths = ensure_dirs(resolve_paths())
    registry_path = paths.artifacts / "registry.json"
    if not registry_path.exists():
        raise RuntimeError("registry.json não encontrado. Execute o treino primeiro.")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if version and registry.get("version") != version:
        raise ValueError(
            f"Versão solicitada {version} difere da versão atual {registry.get('version')}."
        )
    decision_path = paths.eval / "retrain_decision.json"
    if decision_path.exists() and not force:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if decision.get("retrained") and not decision.get("promote"):
            raise RuntimeError(
                "Retreinamento não atingiu o ganho mínimo. Use --force para promover mesmo assim."
            )
    payload = {
        **registry,
        "stage": stage,
        "promoted_at": datetime.now(UTC).isoformat(),
        "promoted_by": actor,
        "governance": {
            "human_approval_required": True,
            "rollback_available": True,
            "previous_registry": str(paths.artifacts / "registry_previous.json"),
        },
    }
    destination = paths.artifacts / f"registry_{stage}.json"
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (paths.logs / "promotions.jsonl").open("a", encoding="utf-8").write(
        json.dumps(
            {
                "ts": payload["promoted_at"],
                "version": payload["version"],
                "stage": stage,
                "actor": actor,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return payload
