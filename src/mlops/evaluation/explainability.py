from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops.models.baselines import feature_names


def tree_shap_importance(
    pipeline: Any,
    frame: pd.DataFrame,
    max_rows: int = 300,
) -> list[dict[str, float]]:
    try:
        import shap

        sample = frame.sample(min(max_rows, len(frame)), random_state=42)
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        transformed = preprocessor.transform(sample)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        importance = np.abs(values).mean(axis=0)
        names = feature_names(pipeline)
        order = np.argsort(-importance)[:20]
        return [
            {"feature": names[index], "mean_abs_shap": float(importance[index])}
            for index in order
        ]
    except Exception:
        return []


def permutation_importance(
    pipeline: Any,
    frame: pd.DataFrame,
    target: str,
    numeric_features: list[str],
    categorical_features: list[str],
    max_rows: int = 500,
) -> list[dict[str, float]]:
    from sklearn.inspection import permutation_importance as sklearn_permutation_importance

    sample = frame.sample(min(max_rows, len(frame)), random_state=42)
    result = sklearn_permutation_importance(
        pipeline,
        sample[numeric_features + categorical_features],
        sample[target].astype(int),
        n_repeats=4,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )
    names = numeric_features + categorical_features
    order = np.argsort(-result.importances_mean)[:20]
    return [
        {
            "feature": names[index],
            "importance": float(result.importances_mean[index]),
            "std": float(result.importances_std[index]),
        }
        for index in order
    ]


def gradient_x_input(
    model: Any,
    x_seq: np.ndarray,
    x_static: np.ndarray,
    schema: dict[str, Any],
    target: str = "inadimplente_90d",
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    import tensorflow as tf

    sequence = tf.convert_to_tensor(x_seq, dtype=tf.float32)
    static = tf.convert_to_tensor(x_static, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(sequence)
        tape.watch(static)
        output = model({"sequence": sequence, "static": static}, training=False)
        output = output[target] if isinstance(output, dict) else output
        score = tf.reduce_sum(output)
    grads_seq, grads_static = tape.gradient(score, [sequence, static])
    seq_importance = tf.reduce_mean(tf.abs(grads_seq), axis=(0, 1)).numpy()
    static_importance = tf.reduce_mean(tf.abs(grads_static), axis=0).numpy()
    seq_names = schema["seq_features"]
    static_names = schema["static_feature_names"]
    seq_order = np.argsort(-seq_importance)[:10]
    static_order = np.argsort(-static_importance)[:10]
    return (
        [
            {"feature": seq_names[index], "importance": float(seq_importance[index])}
            for index in seq_order
        ],
        [
            {"feature": static_names[index], "importance": float(static_importance[index])}
            for index in static_order
        ],
    )


def per_client_drivers(
    model: Any,
    x_seq: np.ndarray,
    x_static: np.ndarray,
    schema: dict[str, Any],
    client_ids: np.ndarray,
    target: str = "inadimplente_90d",
    top_n: int = 3,
) -> dict[str, list[dict[str, float]]]:
    import tensorflow as tf

    sequence = tf.convert_to_tensor(x_seq, dtype=tf.float32)
    static = tf.convert_to_tensor(x_static, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(sequence)
        tape.watch(static)
        output = model({"sequence": sequence, "static": static}, training=False)
        score = output[target] if isinstance(output, dict) else output
        score = tf.reduce_sum(score)
    grads_seq, grads_static = tape.gradient(score, [sequence, static])
    seq_attr = tf.reduce_mean(tf.abs(grads_seq * sequence), axis=1).numpy()
    static_attr = tf.abs(grads_static * static).numpy()
    names = list(schema["seq_features"]) + list(schema["static_feature_names"])
    attr = np.concatenate([seq_attr, static_attr], axis=1)
    drivers: dict[str, list[dict[str, float]]] = {}
    for index, client_id in enumerate(client_ids):
        order = np.argsort(-attr[index])[:top_n]
        drivers[str(client_id)] = [
            {"feature": names[position], "attribution": float(attr[index, position])}
            for position in order
            if attr[index, position] > 0
        ]
    return drivers


def save_json(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
