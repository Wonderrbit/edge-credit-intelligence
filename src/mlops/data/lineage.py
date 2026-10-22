from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_lineage(
    sources: dict[str, str | Path],
    transformations: list[dict[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sources.items()
            if Path(path).exists()
        },
        "transformations": transformations,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    import json

    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
