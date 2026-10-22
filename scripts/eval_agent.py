from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops.agents.evaluate import evaluate_agent
from mlops.config import ensure_dirs, resolve_paths


def main() -> None:
    paths = ensure_dirs(resolve_paths())
    report = evaluate_agent(
        paths.project_root / "data" / "agent_eval.json",
        paths.eval / "agent_eval.json",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"Relatório: {paths.eval / 'agent_eval.json'}")


if __name__ == "__main__":
    main()
