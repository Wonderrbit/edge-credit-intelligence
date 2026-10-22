from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlops.agents.graph import run_agent


def evaluate_agent(cases_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    rows = []
    passed = 0
    for case in cases:
        result = run_agent(case["question"])
        answer = result["answer"].lower()
        checks = {
            "intent": result["intent"] == case["expected_intent"],
            "must_include": all(term.lower() in answer for term in case.get("must_include", [])),
            "citations": (not case.get("must_cite")) or bool(result["citations"]),
            "human_review": bool(result["human_review"]) == bool(case.get("expect_human_review")),
        }
        ok = all(checks.values())
        passed += int(ok)
        rows.append(
            {
                "id": case["id"],
                "ok": ok,
                "checks": checks,
                "intent": result["intent"],
                "expected_intent": case["expected_intent"],
                "n_citations": len(result["citations"]),
                "human_review": result["human_review"],
                "answer_preview": result["answer"][:280],
            }
        )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "cases": len(cases),
        "passed": passed,
        "accuracy": round(passed / max(len(cases), 1), 4),
        "rows": rows,
    }
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
