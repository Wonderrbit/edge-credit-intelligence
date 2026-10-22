from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from mlops.autopilot.decisions import decide_client
from mlops.config import ensure_dirs, resolve_paths
from mlops.evaluation.policy import load_policies
from mlops.judgment import get_jev


def _load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def _metrics(expected: list[str], predicted: list[str]) -> dict:
    labels = sorted(set(expected) | set(predicted))
    return {
        "accuracy": round(float(accuracy_score(expected, predicted)), 4),
        "macro_f1": round(float(f1_score(expected, predicted, average="macro", zero_division=0)), 4),
        "labels": labels,
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
    }


def evaluate(cases: list[dict], use_jev: bool) -> dict:
    policies = load_policies()
    jev = get_jev() if use_jev else None
    rows = []
    for case in cases:
        decision = decide_client(
            case["row"],
            policies=policies,
            jev=jev,
            use_jev=use_jev and jev is not None and jev.available,
            policy_version="eval",
            model_version="eval",
        )
        rows.append(
            {
                "id": case["id"],
                "expected_risk": case["expected"]["risk_action"],
                "predicted_risk": decision["risk"]["action"],
                "expected_collection": case["expected"]["collection_action"],
                "predicted_collection": decision["collection"]["action"],
                "mode": decision["mode"],
                "confidence": decision["risk"]["confidence"],
                "reasons": decision["risk"]["reasons"],
            }
        )
    risk = _metrics(
        [row["expected_risk"] for row in rows],
        [row["predicted_risk"] for row in rows],
    )
    collection = _metrics(
        [row["expected_collection"] for row in rows],
        [row["predicted_collection"] for row in rows],
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "n": len(rows),
        "mode": rows[0]["mode"] if rows else "none",
        "risk": risk,
        "collection": collection,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o julgamento Jev/fallback em casos rotulados.")
    parser.add_argument("--input", default="data/judgment_eval.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--use-jev", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    paths = ensure_dirs(resolve_paths())
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = paths.project_root / input_path
    cases = _load_cases(input_path)
    if args.limit:
        cases = cases[: args.limit]
    report = evaluate(cases, use_jev=args.use_jev)
    output = Path(args.output) if args.output else paths.eval / "judgment_eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Relatório: {output}")


if __name__ == "__main__":
    main()
