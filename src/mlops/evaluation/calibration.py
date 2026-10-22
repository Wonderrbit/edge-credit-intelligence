from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from mlops.evaluation.metrics import expected_calibration_error


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> TemperatureScaler:
        logits = _logit(np.asarray(y_prob, dtype=float))
        y_true = np.asarray(y_true, dtype=float)

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            calibrated = _sigmoid(logits / temperature)
            return float(log_loss(y_true, np.clip(calibrated, 1e-6, 1 - 1e-6)))

        result = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded")
        self.temperature = float(np.exp(result.x))
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        return _sigmoid(_logit(np.asarray(y_prob, dtype=float)) / self.temperature)

    def to_dict(self) -> dict[str, float]:
        return {"temperature": self.temperature}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> TemperatureScaler:
        return cls(temperature=float(payload.get("temperature", 1.0)))


def calibrate_heads(
    y_val: dict[str, np.ndarray],
    probabilities_val: dict[str, np.ndarray],
    y_test: dict[str, np.ndarray],
    probabilities_test: dict[str, np.ndarray],
) -> tuple[dict[str, TemperatureScaler], dict[str, dict[str, float]], dict[str, np.ndarray]]:
    scalers: dict[str, TemperatureScaler] = {}
    report: dict[str, dict[str, float]] = {}
    calibrated: dict[str, np.ndarray] = {}
    for target in y_val:
        scaler = TemperatureScaler().fit(y_val[target], probabilities_val[target])
        scalers[target] = scaler
        before = probabilities_test[target]
        after = scaler.transform(before)
        report[target] = {
            "temperature": scaler.temperature,
            "ece_before": expected_calibration_error(y_test[target], before),
            "ece_after": expected_calibration_error(y_test[target], after),
            "brier_before": float(np.mean((before - y_test[target]) ** 2)),
            "brier_after": float(np.mean((after - y_test[target]) ** 2)),
        }
        calibrated[target] = after
    return scalers, report, calibrated


def save_calibration(scalers: dict[str, TemperatureScaler], path: str | Path) -> None:
    payload = {target: scaler.to_dict() for target, scaler in scalers.items()}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(path: str | Path) -> dict[str, TemperatureScaler]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {target: TemperatureScaler.from_dict(values) for target, values in payload.items()}
