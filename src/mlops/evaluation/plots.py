from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlops.evaluation.metrics import calibration_table, decile_table


def plot_calibration(
    y_true: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    target: str,
    path: str | Path,
) -> None:
    table_before = calibration_table(y_true, before)
    table_after = calibration_table(y_true, after)
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfeita")
    plt.plot(table_before["mean_predicted"], table_before["observed_rate"], marker="o", label="Antes")
    plt.plot(table_after["mean_predicted"], table_after["observed_rate"], marker="o", label="Depois")
    plt.xlabel("Probabilidade prevista")
    plt.ylabel("Taxa observada")
    plt.title(f"Calibração — {target}")
    plt.legend()
    plt.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=140)
    plt.close()


def plot_lift(y_true: np.ndarray, y_prob: np.ndarray, target: str, path: str | Path) -> None:
    table = decile_table(y_true, y_prob)
    plt.figure(figsize=(7, 4))
    plt.bar(table["decile"].astype(str), table["lift"], color="#1f77b4")
    plt.axhline(1.0, color="gray", linestyle="--")
    plt.xlabel("Decil de score (1 = maior risco)")
    plt.ylabel("Lift")
    plt.title(f"Lift por decil — {target}")
    plt.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=140)
    plt.close()


def plot_score_distribution(
    reference: np.ndarray,
    current: np.ndarray,
    target: str,
    path: str | Path,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(reference, bins=30, alpha=0.55, label="Treino", density=True)
    plt.hist(current, bins=30, alpha=0.55, label="Teste", density=True)
    plt.xlabel("Score")
    plt.ylabel("Densidade")
    plt.title(f"Distribuição de score — {target}")
    plt.legend()
    plt.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=140)
    plt.close()


def plot_importance(
    importance: list[dict[str, float]],
    title: str,
    path: str | Path,
) -> None:
    if not importance:
        return
    values = [row.get("importance", row.get("mean_abs_shap", 0.0)) for row in importance[:15]]
    names = [row["feature"] for row in importance[:15]]
    plt.figure(figsize=(7, 5))
    plt.barh(names[::-1], values[::-1], color="#2ca02c")
    plt.xlabel("Importância")
    plt.title(title)
    plt.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=140)
    plt.close()
