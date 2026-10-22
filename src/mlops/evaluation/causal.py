from __future__ import annotations

import numpy as np
from scipy.stats import norm


def uplift_at_k(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
    k: float = 0.20,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    uplift_score = np.asarray(uplift_score, dtype=float)
    treatment = np.asarray(treatment)
    n = max(int(len(y_true) * k), 1)
    order = np.argsort(-uplift_score)[:n]
    treated = order[treatment[order] != 0]
    control = order[treatment[order] == 0]
    treated_rate = y_true[treated].mean() if len(treated) else 0.0
    control_rate = y_true[control].mean() if len(control) else 0.0
    return float(treated_rate - control_rate)


def qini_curve(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    uplift_score = np.asarray(uplift_score, dtype=float)
    treatment = np.asarray(treatment)
    order = np.argsort(-uplift_score)
    y_sorted = y_true[order]
    t_sorted = treatment[order] != 0
    n_treated = max(t_sorted.sum(), 1)
    n_control = max((~t_sorted).sum(), 1)
    treated_cum = np.cumsum(y_sorted * t_sorted) / n_treated
    control_cum = np.cumsum(y_sorted * (~t_sorted)) / n_control
    qini = treated_cum - control_cum
    x = np.arange(1, len(y_true) + 1) / max(len(y_true), 1)
    return x, qini


def qini_coefficient(
    y_true: np.ndarray,
    uplift_score: np.ndarray,
    treatment: np.ndarray,
) -> float:
    x, qini = qini_curve(y_true, uplift_score, treatment)
    return float(np.trapezoid(qini, x))


def ab_test_sample_size(
    baseline_rate: float,
    expected_lift: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    p1 = float(baseline_rate)
    p2 = min(0.999, p1 + float(expected_lift))
    p_bar = (p1 + p2) / 2
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    numerator = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(numerator / max((p2 - p1) ** 2, 1e-9)))


def difference_in_differences(
    treated_pre: float,
    treated_post: float,
    control_pre: float,
    control_post: float,
) -> float:
    return float((treated_post - treated_pre) - (control_post - control_pre))
