from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops.config import ensure_dirs, resolve_paths
from mlops.data.schema import STATIC_CATEGORICAL, STATIC_NUMERIC, TARGETS
from mlops.evaluation.calibration import load_calibration
from mlops.evaluation.limits import recommend_limit
from mlops.evaluation.policy import (
    collection_expected_value,
    load_policies,
    prioritize_collections,
)
from mlops.models.edge_model import load_edge_model, predict_edge


class PredictorError(RuntimeError):
    pass


class CreditPredictor:
    def __init__(self, artifacts_root: str | Path | None = None):
        paths = ensure_dirs(resolve_paths())
        self.paths = paths
        self.artifacts_root = Path(artifacts_root) if artifacts_root else paths.artifacts
        self.registry = self._read_json(self.artifacts_root / "registry.json")
        self.schema = self._read_json(self.artifacts_root / "edge" / "preprocessing.json")
        self.calibration = load_calibration(self.artifacts_root / "models" / "calibration.json")
        self.policies = load_policies()
        self.model = load_edge_model(self.artifacts_root / "models" / "edge_credit_multitask.keras")
        self.serving_clients = pd.read_parquet(self.artifacts_root / "serving" / "serving_clients.parquet")
        self.serving_sequences = np.load(
            self.artifacts_root / "serving" / "serving_sequences.npz", allow_pickle=True
        )
        self.client_index = {
            str(client_id): index
            for index, client_id in enumerate(self.serving_sequences["client_id"].astype(str))
        }
        drivers_path = self.artifacts_root / "serving" / "client_drivers.json"
        self.client_drivers = self._read_json(drivers_path) if drivers_path.exists() else {}
        self.metrics = self._read_json(self.artifacts_root / "eval" / "metrics.json")
        self.model_metrics = self._read_json(self.artifacts_root / "eval" / "model_metrics.json")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise PredictorError(
                f"Artefato ausente: {path}. Execute `python scripts/run_pipeline.py` primeiro."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def list_clients(self, limit: int = 50) -> list[dict[str, Any]]:
        columns = ["client_id", "segmento", "regiao", "renda_mensal", "limite_atual"]
        subset = self.serving_clients[columns].head(limit)
        return subset.to_dict(orient="records")

    def score_client(self, client_id: str, preset: str = "equilibrado") -> dict[str, Any]:
        if client_id not in self.client_index:
            raise KeyError(f"Cliente não encontrado no conjunto de serving: {client_id}")
        row = self.serving_clients.loc[
            self.serving_clients["client_id"] == client_id
        ].iloc[0]
        scores = {
            "pagou_30d": float(row["p_pagou_30d"]),
            "inadimplente_90d": float(row["p_inadimplente_90d"]),
            "aceita_limite": float(row["p_aceita_limite"]),
            "consumo_alto": float(row["p_consumo_alto"]),
        }
        limit_config = self.policies["limit"]
        limit_result = recommend_limit(
            income=float(row["renda_mensal"]),
            current_limit=float(row["limite_atual"]),
            p_accept=scores["aceita_limite"],
            p_consumo=scores["consumo_alto"],
            p_default=scores["inadimplente_90d"],
            avg_spend=float(row["consumo_medio_3m"]),
            preset=limit_config["presets"].get(preset, limit_config["presets"]["equilibrado"]),
            margin=float(limit_config["margin"]),
            lgd=float(limit_config["lgd"]),
            usage_rate=float(limit_config["usage_rate"]),
        )
        debt = float(row.get("valor_atrasado_cutoff", 0.0))
        ev = collection_expected_value(
            p_pay=scores["pagou_30d"],
            debt=debt,
            recovered_fraction=float(self.policies["collection"]["recovered_fraction"]),
            contact_cost=float(self.policies["collection"]["contact_cost"]),
        )
        risk_threshold = float(self.policies["risk"]["approval_threshold"])
        review_threshold = float(self.policies["risk"]["review_threshold"])
        if scores["inadimplente_90d"] >= risk_threshold:
            risk_decision = "reprovar"
        elif scores["inadimplente_90d"] >= review_threshold:
            risk_decision = "revisar"
        else:
            risk_decision = "aprovar"
        return {
            "client_id": client_id,
            "model_version": self.registry.get("version"),
            "segmento": row.get("segmento"),
            "regiao": row.get("regiao"),
            "renda_mensal": float(row["renda_mensal"]),
            "limite_atual": float(row["limite_atual"]),
            "scores": scores,
            "decision": {
                "risco": risk_decision,
                "limite": limit_result,
                "cobranca": {
                    "valor_em_atraso": debt,
                    "ev_contato": round(ev, 2),
                    "prioridade": "alta" if ev > 0 else "baixa",
                    "acao_sugerida": self._collection_action(scores["pagou_30d"]),
                },
            },
            "top_drivers": self.client_drivers.get(client_id, []),
            "disclaimer": "Saída de apoio à decisão. Decisões adversas exigem revisão humana.",
        }

    def _collection_action(self, p_pay: float) -> str:
        if p_pay >= 0.70:
            return "lembrete"
        if p_pay >= 0.50:
            return "whatsapp"
        if p_pay >= 0.30:
            return "ligacao"
        return "negociacao"

    def score_features(self, sequence: list[list[float]], static: dict[str, Any]) -> dict[str, Any]:
        seq = np.asarray(sequence, dtype=np.float32)
        expected_shape = (
            int(self.schema["feature_window"]),
            int(self.schema["n_sequence_features"]),
        )
        if seq.shape != expected_shape:
            raise ValueError(f"sequence deve ter shape {expected_shape}, recebido {seq.shape}")
        means = np.asarray(self.schema["seq_mean"], dtype=np.float32)
        scales = np.asarray(self.schema["seq_scale"], dtype=np.float32)
        seq = (seq - means) / np.where(scales == 0, 1, scales)

        numeric = []
        for column in STATIC_NUMERIC:
            if column not in static:
                raise ValueError(f"Campo estático ausente: {column}")
            numeric.append(float(static[column]))
        numeric = np.asarray(numeric, dtype=np.float32)
        static_mean = np.asarray(self.schema["static_mean"], dtype=np.float32)
        static_scale = np.asarray(self.schema["static_scale"], dtype=np.float32)
        numeric = (numeric - static_mean) / np.where(static_scale == 0, 1, static_scale)

        encoded = []
        for column, categories in zip(
            STATIC_CATEGORICAL, self.schema["categories"], strict=False
        ):
            value = str(static.get(column, categories[0] if categories else ""))
            encoded.extend([1.0 if value == category else 0.0 for category in categories])
        static_vector = np.concatenate([numeric, np.asarray(encoded, dtype=np.float32)])
        raw = predict_edge(
            self.model,
            seq[None, :, :],
            static_vector[None, :],
        )
        calibrated = {
            target: float(self.calibration[target].transform(raw[target])[0]) for target in TARGETS
        }
        return {
            "scores": calibrated,
            "raw_scores": {target: float(raw[target][0]) for target in TARGETS},
            "model_version": self.registry.get("version"),
            "disclaimer": "Saída de apoio à decisão. Decisões adversas exigem revisão humana.",
        }

    def batch_score(self, client_ids: list[str] | None = None, preset: str = "equilibrado") -> list[dict[str, Any]]:
        ids = client_ids or self.serving_clients["client_id"].head(100).tolist()
        return [self.score_client(client_id, preset=preset) for client_id in ids]

    def collection_queue(self, top_n: int = 20) -> list[dict[str, Any]]:
        frame = self.serving_clients.copy()
        ranked = prioritize_collections(
            frame,
            top_fraction=min(1.0, max(top_n / max(len(frame), 1), 0.01)),
            recovered_fraction=float(self.policies["collection"]["recovered_fraction"]),
            contact_cost=float(self.policies["collection"]["contact_cost"]),
        ).head(top_n)
        return ranked[
            [
                "client_id",
                "valor_atrasado_cutoff",
                "p_pagou_30d",
                "ev_cobranca",
                "prioridade",
            ]
        ].to_dict(orient="records")

    def model_info(self) -> dict[str, Any]:
        return {
            "registry": self.registry,
            "edge_metrics": self.model_metrics.get("edge_metrics", {}),
            "calibration": self.model_metrics.get("calibration", {}),
            "thresholds": self.policies["risk"],
            "edge_artifacts": {
                "model_json": str(self.paths.edge / "model.json"),
                "preprocessing": str(self.paths.edge / "preprocessing.json"),
            },
        }

    def monitoring_report(self) -> dict[str, Any]:
        report_path = self.artifacts_root / "eval" / "monitoring_report.json"
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "status": "not_generated",
            "message": "Execute `python scripts/run_monitoring.py` para gerar o relatório.",
        }


@lru_cache(maxsize=1)
def get_predictor() -> CreditPredictor:
    return CreditPredictor()
