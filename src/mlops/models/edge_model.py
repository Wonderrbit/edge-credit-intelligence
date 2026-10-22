from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tf_keras as keras
from tf_keras import layers

from mlops.data.schema import TARGETS
from mlops.evaluation.metrics import binary_metrics


def build_edge_model(
    sequence_length: int,
    n_sequence_features: int,
    n_static_features: int,
    gru_units: int = 32,
    dense_units: int = 32,
    learning_rate: float = 0.001,
) -> keras.Model:
    sequence_input = layers.Input(
        shape=(sequence_length, n_sequence_features), name="sequence"
    )
    static_input = layers.Input(shape=(n_static_features,), name="static")

    x = layers.GRU(gru_units, return_sequences=False, name="gru")(sequence_input)
    x = layers.LayerNormalization(name="sequence_norm")(x)
    x = layers.Concatenate(name="concat")([x, static_input])
    x = layers.Dense(dense_units, activation="relu", name="shared_dense")(x)
    x = layers.Dropout(0.25, name="shared_dropout")(x)

    outputs = {
        "pagou_30d": layers.Dense(1, activation="sigmoid", name="pagou_30d")(x),
        "inadimplente_90d": layers.Dense(
            1, activation="sigmoid", name="inadimplente_90d"
        )(x),
        "aceita_limite": layers.Dense(1, activation="sigmoid", name="aceita_limite")(x),
        "consumo_alto": layers.Dense(1, activation="sigmoid", name="consumo_alto")(x),
    }

    model = keras.Model(
        inputs={"sequence": sequence_input, "static": static_input},
        outputs=outputs,
        name="edge_credit_multitask",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            target: keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=True, gamma=2.0
            )
            for target in TARGETS
        },
        metrics={
            target: [
                keras.metrics.AUC(name="auc"),
                keras.metrics.BinaryAccuracy(name="accuracy"),
            ]
            for target in TARGETS
        },
    )
    return model


def _split_arrays(dataset: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mask = dataset["split"] == split
    x_seq = dataset["X_seq"][mask]
    x_static = dataset["X_static"][mask]
    y = {target: dataset["y"][target][mask] for target in TARGETS}
    return x_seq, x_static, y


def train_edge_model(
    dataset: dict[str, Any],
    config: dict[str, Any] | None = None,
    output_dir: str | Path = "artifacts/models",
    epochs: int | None = None,
) -> dict[str, Any]:
    config = config or {}
    edge_config = config.get("edge", config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_length = int(dataset["X_seq"].shape[1])
    n_sequence_features = int(dataset["X_seq"].shape[2])
    n_static_features = int(dataset["X_static"].shape[1])

    model = build_edge_model(
        sequence_length=sequence_length,
        n_sequence_features=n_sequence_features,
        n_static_features=n_static_features,
        gru_units=int(edge_config.get("gru_units", 32)),
        dense_units=int(edge_config.get("dense_units", 32)),
        learning_rate=float(edge_config.get("learning_rate", 0.001)),
    )

    x_train, s_train, y_train = _split_arrays(dataset, "train")
    x_val, s_val, y_val = _split_arrays(dataset, "validation")
    x_test, s_test, y_test = _split_arrays(dataset, "test")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]
    history = model.fit(
        {"sequence": x_train, "static": s_train},
        y_train,
        validation_data=({"sequence": x_val, "static": s_val}, y_val),
        epochs=epochs or int(edge_config.get("epochs", 35)),
        batch_size=int(edge_config.get("batch_size", 64)),
        callbacks=callbacks,
        verbose=0,
    )

    val_probs = _predict(model, x_val, s_val)
    test_probs = _predict(model, x_test, s_test)
    test_metrics = {
        target: binary_metrics(y_test[target], test_probs[target]) for target in TARGETS
    }
    val_metrics = {
        target: binary_metrics(y_val[target], val_probs[target]) for target in TARGETS
    }

    model_path = output_dir / "edge_credit_multitask.keras"
    model.save(model_path)
    try:
        model.save(output_dir / "edge_credit_multitask.h5")
    except Exception:
        pass

    (output_dir / "edge_metrics.json").write_text(
        json.dumps(
            {
                "test": test_metrics,
                "validation": val_metrics,
                "params": int(model.count_params()),
                "history": {key: [float(value) for value in values] for key, values in history.history.items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "edge_test_predictions.npz",
        **{target: test_probs[target] for target in TARGETS},
        **{f"val_{target}": val_probs[target] for target in TARGETS},
        **{f"y_test_{target}": y_test[target] for target in TARGETS},
        **{f"y_val_{target}": y_val[target] for target in TARGETS},
    )
    return {
        "model": model,
        "history": history,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "model_path": model_path,
    }


def _predict(model: keras.Model, x_seq: np.ndarray, x_static: np.ndarray) -> dict[str, np.ndarray]:
    raw = model.predict({"sequence": x_seq, "static": x_static}, verbose=0)
    if isinstance(raw, dict):
        return {target: np.asarray(raw[target]).ravel() for target in TARGETS}
    if isinstance(raw, (list, tuple)):
        return {target: np.asarray(value).ravel() for target, value in zip(TARGETS, raw, strict=False)}
    return {"pagou_30d": np.asarray(raw).ravel()}


def load_edge_model(path: str | Path = "artifacts/models/edge_credit_multitask.keras") -> keras.Model:
    model_path = Path(path)
    try:
        return keras.models.load_model(model_path, compile=False)
    except (OSError, ValueError):
        h5_path = model_path.with_suffix(".h5")
        if h5_path == model_path or not h5_path.exists():
            raise
        return keras.models.load_model(h5_path, compile=False)


def predict_edge(
    model: keras.Model,
    x_seq: np.ndarray,
    x_static: np.ndarray,
) -> dict[str, np.ndarray]:
    return _predict(model, x_seq, x_static)
