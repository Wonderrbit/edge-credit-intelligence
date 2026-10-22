from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops.config import ensure_dirs, resolve_paths, set_seed
from mlops.data.schema import (
    CUTOFF_COL,
    ID_COL,
    SAFRA_COL,
    TARGET_END_COL,
    TARGETS,
    TARGET_COLUMNS,
)

PERSONAS = ("saudavel", "alavancado", "recuperavel", "cronico", "novo")
PERSONA_PROBS = (0.38, 0.22, 0.18, 0.10, 0.12)

PERSONA_PARAMS: dict[str, dict[str, float]] = {
    "saudavel": {
        "compras": 8.0,
        "ticket": 190.0,
        "parcelas": 2.2,
        "categorias": 5.5,
        "utilizacao": 0.32,
        "sessoes": 18.0,
        "atraso_prob": 0.04,
        "atraso_dias": 8.0,
        "p_pay_base": 0.86,
        "limite_factor": 0.75,
        "bureau": 760.0,
    },
    "alavancado": {
        "compras": 6.5,
        "ticket": 245.0,
        "parcelas": 5.0,
        "categorias": 5.0,
        "utilizacao": 0.78,
        "sessoes": 13.0,
        "atraso_prob": 0.23,
        "atraso_dias": 24.0,
        "p_pay_base": 0.58,
        "limite_factor": 1.35,
        "bureau": 640.0,
    },
    "recuperavel": {
        "compras": 4.0,
        "ticket": 150.0,
        "parcelas": 3.0,
        "categorias": 3.8,
        "utilizacao": 0.92,
        "sessoes": 8.0,
        "atraso_prob": 0.44,
        "atraso_dias": 52.0,
        "p_pay_base": 0.44,
        "limite_factor": 0.95,
        "bureau": 560.0,
    },
    "cronico": {
        "compras": 2.8,
        "ticket": 110.0,
        "parcelas": 4.0,
        "categorias": 2.8,
        "utilizacao": 1.08,
        "sessoes": 5.0,
        "atraso_prob": 0.67,
        "atraso_dias": 78.0,
        "p_pay_base": 0.20,
        "limite_factor": 0.85,
        "bureau": 470.0,
    },
    "novo": {
        "compras": 5.0,
        "ticket": 155.0,
        "parcelas": 2.5,
        "categorias": 4.0,
        "utilizacao": 0.42,
        "sessoes": 16.0,
        "atraso_prob": 0.10,
        "atraso_dias": 14.0,
        "p_pay_base": 0.60,
        "limite_factor": 0.45,
        "bureau": 680.0,
    },
}

ACTION_COSTS = np.array([0.0, 2.0, 5.0, 18.0, 35.0])
ACTION_UPLIFT = np.array([0.00, 0.04, 0.09, 0.16, 0.25])


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def _month_end_dates(months: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-31", periods=months, freq="ME")


def generate_synthetic_data(
    output_dir: str | Path | None = None,
    n_clients: int = 6000,
    months: int = 24,
    seed: int = 42,
    feature_months: int = 12,
    performance_months: int = 6,
    safras: tuple[int, ...] | list[int] = (1, 2, 3, 4, 5, 6),
) -> dict[str, pd.DataFrame]:
    """Generate a realistic synthetic Open Finance / retail credit panel.

    The panel is a client-month table. Each client has a cohort (safra) that
    defines a 12-month feature window and a 6-month performance window. Every
    feature is computed only from the feature window, while targets come from
    the performance window, which makes temporal leakage tests meaningful.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    paths = ensure_dirs(resolve_paths())
    output_path = Path(output_dir) if output_dir else paths.data_raw
    output_path.mkdir(parents=True, exist_ok=True)

    max_safra = max(safras)
    required_months = max_safra + feature_months + performance_months - 1
    if months < required_months:
        raise ValueError(f"months={months} insuficiente; use pelo menos {required_months}")

    client_ids = np.array([f"C{i:06d}" for i in range(1, n_clients + 1)])
    personas = rng.choice(PERSONAS, size=n_clients, p=PERSONA_PROBS)
    safra = rng.choice(np.asarray(safras), size=n_clients)
    cutoff = safra + feature_months - 1
    target_end = cutoff + performance_months
    if target_end.max() > months:
        raise ValueError("A janela de performance excede o painel gerado")

    age = rng.integers(18, 76, size=n_clients)
    income = rng.lognormal(mean=np.log(4200), sigma=0.48, size=n_clients).clip(900, 45000)
    region = rng.choice(
        ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"],
        size=n_clients,
        p=[0.44, 0.20, 0.20, 0.09, 0.07],
    )
    segment = rng.choice(
        ["Varejo", "Alto Valor", "Novo", "Recuperacao"],
        size=n_clients,
        p=[0.58, 0.14, 0.16, 0.12],
    )
    tenure = rng.integers(1, 181, size=n_clients)
    open_finance_accounts = rng.integers(0, 6, size=n_clients)
    products = rng.integers(1, 6, size=n_clients)

    persona_ticket = np.array([PERSONA_PARAMS[p]["ticket"] for p in personas])
    persona_limit = np.array([PERSONA_PARAMS[p]["limite_factor"] for p in personas])
    persona_bureau = np.array([PERSONA_PARAMS[p]["bureau"] for p in personas])
    bureau_score = np.clip(rng.normal(persona_bureau, 55), 250, 1000)
    current_limit = np.clip(
        income * persona_limit * rng.uniform(0.35, 1.45, size=n_clients),
        300,
        income * 3.2,
    )

    clients = pd.DataFrame(
        {
            ID_COL: client_ids,
            "idade": age,
            "renda_mensal": income.round(2),
            "regiao": region,
            "segmento": segment,
            "tempo_relacionamento_meses": tenure,
            "limite_atual": current_limit.round(2),
            "bureau_score": bureau_score.round(0),
            "contas_open_finance": open_finance_accounts,
            "produtos_ativos": products,
            "persona": personas,
            SAFRA_COL: safra,
            CUTOFF_COL: cutoff,
            TARGET_END_COL: target_end,
        }
    )

    month_index = np.arange(months)
    seasonality = 1.0 + 0.16 * np.sin(2 * np.pi * month_index / 12)
    seasonality = seasonality + np.where((month_index + 1) == 12, 0.18, 0.0)
    drift = np.where(month_index >= 18, 1.0 + 0.045 * (month_index - 17), 1.0)
    income_shock = np.where(month_index >= 20, 0.92, 1.0)

    persona_compras = np.array([PERSONA_PARAMS[p]["compras"] for p in personas])[:, None]
    persona_ticket_arr = persona_ticket[:, None]
    persona_parcelas = np.array([PERSONA_PARAMS[p]["parcelas"] for p in personas])[:, None]
    persona_categorias = np.array([PERSONA_PARAMS[p]["categorias"] for p in personas])[:, None]
    persona_util = np.array([PERSONA_PARAMS[p]["utilizacao"] for p in personas])[:, None]
    persona_sessoes = np.array([PERSONA_PARAMS[p]["sessoes"] for p in personas])[:, None]
    persona_late_prob = np.array([PERSONA_PARAMS[p]["atraso_prob"] for p in personas])[:, None]
    persona_late_days = np.array([PERSONA_PARAMS[p]["atraso_dias"] for p in personas])[:, None]
    persona_limits = np.array([PERSONA_PARAMS[p]["limite_factor"] for p in personas])[:, None]

    client_factor = rng.lognormal(mean=0.0, sigma=0.20, size=(n_clients, 1))

    compras_qtd = rng.poisson(
        np.clip(persona_compras * seasonality * drift * client_factor, 0.05, 40)
    ).astype(int)
    ticket_medio = rng.lognormal(
        mean=np.log(persona_ticket_arr), sigma=0.30, size=(n_clients, months)
    ).clip(8, 2500)
    valor_total = compras_qtd * ticket_medio
    parcelas_medias = np.clip(
        np.round(rng.normal(persona_parcelas, 1.1, size=(n_clients, months))), 1, 12
    )
    categoria_diversidade = np.clip(
        np.round(rng.normal(persona_categorias, 1.5, size=(n_clients, months))), 1, 10
    )

    p_credito = np.clip(
        rng.normal(0.42, 0.10, size=(n_clients, months)) + (persona_util - 0.5) * 0.12,
        0.05,
        0.80,
    )
    p_pix = np.clip(rng.normal(0.27, 0.08, size=(n_clients, months)), 0.03, 0.65)
    p_boleto = np.clip(rng.normal(0.13, 0.05, size=(n_clients, months)), 0.01, 0.35)
    p_debito = np.clip(1.0 - p_credito - p_pix - p_boleto, 0.02, 0.55)
    scale = p_credito + p_debito + p_pix + p_boleto
    p_credito, p_debito, p_pix, p_boleto = (
        p_credito / scale,
        p_debito / scale,
        p_pix / scale,
        p_boleto / scale,
    )

    late_prob = np.clip(persona_late_prob * drift * (1.0 + 0.20 * (persona_util > 0.8)), 0, 0.95)
    late_event = rng.random(size=(n_clients, months)) < late_prob
    late_draw = rng.gamma(
        shape=2.0,
        scale=persona_late_days / 2.0,
        size=(n_clients, months),
    )
    atraso_dias = np.where(late_event, np.clip(late_draw, 1, 120), 0.0)
    atraso_dias = np.round(atraso_dias).astype(int)

    dias_sem_transacao = np.clip(
        np.round(rng.normal(5.5, 3.2, size=(n_clients, months)) + atraso_dias * 0.08), 0, 30
    )
    utilizacao_limite = np.clip(
        rng.normal(persona_util, 0.14, size=(n_clients, months))
        + drift * 0.035
        + atraso_dias * 0.001,
        0,
        1.55,
    )
    sessoes_app = rng.poisson(
        np.clip(persona_sessoes * seasonality * client_factor, 0.2, 80)
    ).astype(int)
    tempo_medio_sessao = np.clip(
        rng.normal(190, 55, size=(n_clients, months)), 10, 700
    )
    velocidade_scroll = np.clip(
        rng.normal(45, 18, size=(n_clients, months)), 5, 150
    )
    cliques = rng.poisson(np.clip(sessoes_app * 0.8 + 3, 0.2, 80)).astype(int)

    contatos_cobranca = rng.poisson(
        np.where(atraso_dias > 15, 1.2 + persona_late_prob * 2.0, 0.03)
    ).astype(int)
    promessas_pagamento = rng.binomial(contatos_cobranca, 0.42)

    p_pay_history = np.clip(
        _sigmoid(
            0.9
            - atraso_dias / 45.0
            - utilizacao_limite * 0.8
            + (persona_util < 0.5) * 0.35
        ),
        0.02,
        0.98,
    )
    pagamentos_realizados = rng.binomial(
        np.clip(compras_qtd + 1, 1, 50), p_pay_history
    ).astype(int)
    valor_pago = np.where(
        pagamentos_realizados > 0,
        valor_total * rng.uniform(0.15, 1.0, size=(n_clients, months)),
        0.0,
    )
    valor_atrasado = np.where(
        atraso_dias > 0,
        valor_total * rng.uniform(0.20, 1.0, size=(n_clients, months)),
        0.0,
    )

    saldo_open_finance = np.clip(
        rng.normal(income[:, None] * 0.55, income[:, None] * 0.35), 0, None
    ) * income_shock
    entradas_open_finance = np.clip(
        rng.normal(income[:, None], income[:, None] * 0.18), 0, None
    ) * income_shock
    saidas_open_finance = np.clip(
        rng.normal(income[:, None] * 0.92, income[:, None] * 0.22)
        + valor_atrasado * 0.02,
        0,
        None,
    )
    score_bureau_mes = np.clip(
        bureau_score[:, None] + rng.normal(0, 25, size=(n_clients, months)) - atraso_dias * 0.7,
        0,
        1000,
    )

    dates = _month_end_dates(months)
    panel_records: list[pd.DataFrame] = []
    for idx, client_id in enumerate(client_ids):
        frame = pd.DataFrame(
            {
                ID_COL: client_id,
                "mes": month_index + 1,
                "data_ref": dates,
                "compras_qtd": compras_qtd[idx],
                "ticket_medio": ticket_medio[idx].round(2),
                "valor_total": valor_total[idx].round(2),
                "parcelas_medias": parcelas_medias[idx],
                "categoria_diversidade": categoria_diversidade[idx],
                "pct_credito": p_credito[idx].round(4),
                "pct_debito": p_debito[idx].round(4),
                "pct_pix": p_pix[idx].round(4),
                "pct_boleto": p_boleto[idx].round(4),
                "atraso_dias": atraso_dias[idx],
                "valor_atrasado": valor_atrasado[idx].round(2),
                "dias_sem_transacao": dias_sem_transacao[idx],
                "sessoes_app": sessoes_app[idx],
                "tempo_medio_sessao": tempo_medio_sessao[idx].round(1),
                "velocidade_scroll": velocidade_scroll[idx].round(1),
                "cliques": cliques[idx],
                "contatos_cobranca": contatos_cobranca[idx],
                "promessas_pagamento": promessas_pagamento[idx],
                "pagamentos_realizados": pagamentos_realizados[idx],
                "valor_pago": valor_pago[idx].round(2),
                "utilizacao_limite": utilizacao_limite[idx].round(4),
                "saldo_open_finance": saldo_open_finance[idx].round(2),
                "entradas_open_finance": entradas_open_finance[idx].round(2),
                "saidas_open_finance": saidas_open_finance[idx].round(2),
                "score_bureau_mes": score_bureau_mes[idx].round(0),
                SAFRA_COL: safra[idx],
                CUTOFF_COL: cutoff[idx],
                TARGET_END_COL: target_end[idx],
            }
        )
        panel_records.append(frame)

    panel = pd.concat(panel_records, ignore_index=True)

    row_idx = np.arange(n_clients)[:, None]
    feature_idx = (safra[:, None] + np.arange(feature_months)[None, :]) - 1
    performance_idx = (cutoff[:, None] + np.arange(performance_months)[None, :]) - 1

    perf_atraso = atraso_dias[row_idx, performance_idx]
    perf_valor = valor_total[row_idx, performance_idx]
    cutoff_atraso = atraso_dias[row_idx, (cutoff - 1)[:, None]].ravel()
    cutoff_valor_atrasado = valor_atrasado[row_idx, (cutoff - 1)[:, None]].ravel()
    consumo_30d = perf_valor[:, 0]

    inadimplente_90d = (perf_atraso.max(axis=1) >= 90).astype(int)
    consumo_performance_medio = perf_valor.mean(axis=1)
    consumo_alto = (consumo_performance_medio > income * 0.45).astype(int)

    eligible_limit = (current_limit < income * 2.0).astype(int)
    cutoff_utilization = utilizacao_limite[row_idx, (cutoff - 1)[:, None]].ravel()
    p_accept = _sigmoid(
        -0.10
        + 1.35 * (consumo_performance_medio / np.maximum(income, 1))
        - 1.75 * cutoff_utilization
        - 2.10 * inadimplente_90d
        + 0.80 * (personas == "novo")
        + 0.35 * eligible_limit
        - 0.45 * (segment == "Recuperacao")
        + rng.normal(0, 0.18, size=n_clients)
    )
    aceita_limite = (rng.random(n_clients) < p_accept) & (eligible_limit == 1)
    aceita_limite = aceita_limite.astype(int)

    p_pay_base = np.array([PERSONA_PARAMS[p]["p_pay_base"] for p in personas])
    action = rng.integers(0, len(ACTION_COSTS), size=n_clients)
    p_pay = _sigmoid(
        np.log(np.clip(p_pay_base, 0.02, 0.98) / (1 - np.clip(p_pay_base, 0.02, 0.98)))
        - 0.70 * (cutoff_atraso > 30)
        - 0.45 * cutoff_utilization
        + ACTION_UPLIFT[action]
        + rng.normal(0, 0.16, size=n_clients)
    )
    p_pay = np.clip(p_pay, 0.01, 0.98)
    pagou_30d = rng.binomial(1, p_pay)
    valor_recuperado = np.where(
        pagou_30d == 1,
        cutoff_valor_atrasado * rng.uniform(0.45, 0.85, size=n_clients),
        0.0,
    )

    targets = pd.DataFrame(
        {
            ID_COL: client_ids,
            SAFRA_COL: safra,
            CUTOFF_COL: cutoff,
            TARGET_END_COL: target_end,
            "pagou_30d": pagou_30d.astype(int),
            "inadimplente_90d": inadimplente_90d.astype(int),
            "aceita_limite": aceita_limite.astype(int),
            "consumo_alto": consumo_alto.astype(int),
            "consumo_30d": consumo_30d.round(2),
            "consumo_performance_medio": consumo_performance_medio.round(2),
            "elegivel_limite": eligible_limit.astype(int),
            "valor_atrasado_cutoff": cutoff_valor_atrasado.round(2),
        }
    )
    targets = targets[TARGET_COLUMNS]

    tratamentos = pd.DataFrame(
        {
            ID_COL: client_ids,
            "acao": action,
            "pagou_30d": pagou_30d.astype(int),
            "valor_recuperado": valor_recuperado.round(2),
            "custo_acao": ACTION_COSTS[action],
            "p_pay_simulado": p_pay.round(4),
        }
    )

    transacoes = panel[
        [
            ID_COL,
            "mes",
            "data_ref",
            "compras_qtd",
            "ticket_medio",
            "valor_total",
            "parcelas_medias",
            "categoria_diversidade",
            "pct_credito",
            "pct_debito",
            "pct_pix",
            "pct_boleto",
            "atraso_dias",
            "dias_sem_transacao",
            "sessoes_app",
            "tempo_medio_sessao",
            "velocidade_scroll",
            "cliques",
        ]
    ].copy()
    cobrancas = panel[
        [
            ID_COL,
            "mes",
            "data_ref",
            "atraso_dias",
            "valor_atrasado",
            "contatos_cobranca",
            "promessas_pagamento",
            "pagamentos_realizados",
            "valor_pago",
        ]
    ].copy()
    open_finance = panel[
        [
            ID_COL,
            "mes",
            "data_ref",
            "saldo_open_finance",
            "entradas_open_finance",
            "saidas_open_finance",
            "score_bureau_mes",
        ]
    ].copy()

    datasets: dict[str, pd.DataFrame] = {
        "clientes": clients,
        "painel": panel,
        "transacoes": transacoes,
        "cobrancas": cobrancas,
        "open_finance": open_finance,
        "targets": targets,
        "tratamentos": tratamentos,
    }
    for name, frame in datasets.items():
        frame.to_parquet(output_path / f"{name}.parquet", index=False)

    metadata: dict[str, Any] = {
        "n_clients": n_clients,
        "months": months,
        "seed": seed,
        "feature_months": feature_months,
        "performance_months": performance_months,
        "safras": list(safras),
        "targets": list(TARGETS),
        "window_rule": "features: safra..safra+11; targets: cutoff+1..cutoff+6",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    pd.Series(metadata).to_json(output_path / "metadata.json", force_ascii=False, indent=2)
    return datasets
