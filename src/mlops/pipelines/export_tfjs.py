from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

from mlops.config import ensure_dirs, resolve_paths
from mlops.data.lineage import sha256_file


def _stub_optional_tfjs_dependencies() -> None:
    # tensorflowjs imports optional modules that are not required for a Keras
    # model export. Newer TensorFlow versions removed some of their APIs, so we
    # provide empty modules to keep the Keras -> TF.js converter usable.
    for module in ("tensorflow_decision_forests", "tensorflow_hub"):
        if module not in sys.modules:
            sys.modules[module] = types.ModuleType(module)


def _portable_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def export_tfjs(
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    quantization: str = "float16",
) -> dict[str, Any]:
    paths = ensure_dirs(resolve_paths())
    model_path = Path(model_path) if model_path else paths.models / "edge_credit_multitask.keras"
    output_dir = Path(output_dir) if output_dir else paths.edge
    output_dir.mkdir(parents=True, exist_ok=True)

    _stub_optional_tfjs_dependencies()
    import tf_keras

    try:
        from tensorflowjs.converters import save_keras_model
    except ImportError as exc:
        raise SystemExit(
            "tensorflowjs nao instalado. Rode: "
            "python -m pip install -r requirements-export.txt && "
            "python -m pip install --no-deps tensorflowjs==4.22.0"
        ) from exc

    model = tf_keras.models.load_model(model_path, compile=False)
    quantization_map = {"float16": "*"} if quantization == "float16" else None
    save_keras_model(model, str(output_dir), quantization_dtype_map=quantization_map)

    schema_source = paths.data_processed / "feature_schema.json"
    if schema_source.exists():
        (output_dir / "preprocessing.json").write_text(
            schema_source.read_text(encoding="utf-8"), encoding="utf-8"
        )

    model_json = output_dir / "model.json"
    weight_files = sorted(output_dir.glob("group*.bin"))
    info = {
        "model_path": _portable_path(model_path, paths.project_root),
        "model_sha256": sha256_file(model_path),
        "tfjs_model_json": _portable_path(model_json, paths.project_root),
        "tfjs_model_sha256": sha256_file(model_json),
        "weight_files": [
            {
                "path": _portable_path(path, paths.project_root),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in weight_files
        ],
        "parameters": int(model.count_params()),
        "quantization": quantization,
        "total_bytes": int(model_json.stat().st_size + sum(path.stat().st_size for path in weight_files)),
        "inputs": [layer.name for layer in model.inputs],
        "outputs": [layer.name for layer in model.outputs],
    }
    (output_dir / "tfjs_manifest.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return info
