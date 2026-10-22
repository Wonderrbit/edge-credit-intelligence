from __future__ import annotations

from mlops.serving.predictor import CreditPredictor


def test_predictor_scores_bundled_demo_client() -> None:
    predictor = CreditPredictor()
    result = predictor.score_client("C000001")

    assert result["client_id"] == "C000001"
    assert set(result["scores"]) == {
        "pagou_30d",
        "inadimplente_90d",
        "aceita_limite",
        "consumo_alto",
    }
    assert result["decision"]["risco"] in {"aprovar", "revisar", "reprovar"}
