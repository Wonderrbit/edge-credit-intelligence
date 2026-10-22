from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from mlops.evaluation.causal import qini_coefficient, uplift_at_k
from mlops.models.baselines import build_preprocessor


class UpliftTLearner:
    """Two-model uplift learner for collection treatments.

    A separate outcome model is fitted for each action. The recommended action
    is the one with the highest predicted payment probability, and uplift is
    measured against the no-action/control group.
    """

    def __init__(self, numeric_features: list[str], categorical_features: list[str], random_state: int = 42):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.preprocessor = build_preprocessor(numeric_features, categorical_features)
        self.models: dict[int, LGBMClassifier] = {}
        self.actions: list[int] = []

    def fit(self, frame: pd.DataFrame, treatment_column: str = "acao", target: str = "pagou_30d") -> UpliftTLearner:
        features = frame[self.numeric_features + self.categorical_features]
        self.preprocessor.fit(features)
        transformed = self.preprocessor.transform(features)
        self.actions = sorted(frame[treatment_column].unique().tolist())
        for action in self.actions:
            mask = frame[treatment_column].to_numpy() == action
            if mask.sum() < 20:
                continue
            model = LGBMClassifier(
                n_estimators=160,
                learning_rate=0.05,
                num_leaves=15,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(transformed[mask], frame.loc[mask, target].astype(int))
            self.models[int(action)] = model
        return self

    def predict_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(
            frame[self.numeric_features + self.categorical_features]
        )
        columns = []
        for action in self.actions:
            model = self.models.get(int(action))
            if model is None:
                columns.append(np.full(len(frame), 0.0))
            else:
                columns.append(model.predict_proba(transformed)[:, 1])
        return np.column_stack(columns)

    def recommend(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        matrix = self.predict_matrix(frame)
        best_index = matrix.argmax(axis=1)
        actions = np.array(self.actions)[best_index]
        control_index = self.actions.index(0) if 0 in self.actions else 0
        uplift = matrix.max(axis=1) - matrix[:, control_index]
        return {
            "acao_recomendada": actions,
            "uplift_estimado": uplift,
            "p_pagamento_estimada": matrix.max(axis=1),
        }


def train_uplift(
    tabular: pd.DataFrame,
    treatments: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    output_dir: str | Path = "artifacts/models",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = tabular.merge(treatments, on="client_id", how="inner", suffixes=("", "_trat"))
    train = frame[frame["safra"].isin((1, 2, 3, 4))].copy()
    test = frame[frame["safra"].isin((5, 6))].copy()
    learner = UpliftTLearner(numeric_features, categorical_features).fit(train)
    recommendation = learner.recommend(test)
    control_mask = test["acao"].to_numpy() == 0
    any_mask = test["acao"].to_numpy() != 0
    uplift = recommendation["uplift_estimado"]
    metrics = {
        "n_test": int(len(test)),
        "uplift_at_20": uplift_at_k(
            test["pagou_30d"].to_numpy(), uplift, test["acao"].to_numpy(), k=0.20
        ),
        "qini": qini_coefficient(
            test["pagou_30d"].to_numpy(), uplift, test["acao"].to_numpy()
        ),
        "control_rate": float(test.loc[control_mask, "pagou_30d"].mean()) if control_mask.any() else 0.0,
        "treated_rate": float(test.loc[any_mask, "pagou_30d"].mean()) if any_mask.any() else 0.0,
        "recommended_actions": pd.Series(recommendation["acao_recomendada"]).value_counts().to_dict(),
    }
    joblib.dump(learner, output_dir / "uplift_collections.joblib")
    (output_dir / "uplift_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"learner": learner, "metrics": metrics, "test_recommendation": recommendation}
