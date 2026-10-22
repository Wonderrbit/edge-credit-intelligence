from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _parse_destination(destination: str) -> tuple[str, str] | None:
    if not destination.startswith("s3://"):
        return None
    bucket, _, prefix = destination[5:].partition("/")
    if not bucket:
        return None
    return bucket, prefix.strip("/")


def _client() -> Any:
    try:
        import boto3
    except ImportError:
        return None
    return boto3.client("s3")


def upload_run(run_dir: str | Path, destination: str | None = None) -> str | None:
    target = destination or os.getenv("MLOPS_ARTIFACTS_BUCKET") or os.getenv("MLOPS_ARTIFACTS_S3")
    if not target:
        return None
    parsed = _parse_destination(target)
    client = _client()
    if parsed is None or client is None:
        return None
    bucket, prefix = parsed
    run_path = Path(run_dir)
    base = "/".join(part for part in (prefix, "runs", run_path.name) if part)
    try:
        for path in sorted(run_path.rglob("*")):
            if not path.is_file():
                continue
            key = f"{base}/{path.relative_to(run_path).as_posix()}"
            client.upload_file(str(path), bucket, key)
    except Exception:
        return None
    return f"s3://{bucket}/{base}"


def upload_file(path: str | Path, s3_uri: str) -> bool:
    parsed = _parse_destination(s3_uri)
    client = _client()
    if parsed is None or client is None:
        return False
    bucket, key = parsed
    try:
        client.upload_file(str(path), bucket, key)
    except Exception:
        return False
    return True
