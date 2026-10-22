from __future__ import annotations

from dataclasses import dataclass

ID_COL = "client_id"
SAFRA_COL = "safra"
CUTOFF_COL = "cutoff_month"
TARGET_END_COL = "target_end_month"

TARGETS = ("pagou_30d", "inadimplente_90d", "aceita_limite", "consumo_alto")
TARGET_LABELS = {
    "pagou_30d": "Propensão de pagamento em 30 dias",
    "inadimplente_90d": "Risco de inadimplência em 90 dias",
    "aceita_limite": "Propensão a aceitar/usar aumento de limite",
    "consumo_alto": "Propensão de consumo acima do perfil",
}

SEQ_FEATURES = [
    "compras_qtd",
    "ticket_medio",
    "valor_total",
    "parcelas_medias",
    "categoria_diversidade",
    "pct_credito",
    "pct_pix",
    "pct_boleto",
    "atraso_dias",
    "utilizacao_limite",
    "dias_sem_transacao",
    "sessoes_app",
    "tempo_medio_sessao",
    "velocidade_scroll",
    "cliques",
    "contatos_cobranca",
    "promessas_pagamento",
    "valor_pago",
    "saldo_open_finance",
    "entradas_open_finance",
    "saidas_open_finance",
]

STATIC_NUMERIC = [
    "idade",
    "renda_mensal",
    "tempo_relacionamento_meses",
    "limite_atual",
    "bureau_score",
    "contas_open_finance",
    "produtos_ativos",
    "consumo_sobre_renda",
    "utilizacao_media",
    "utilizacao_max",
    "tendencia_consumo",
]

STATIC_CATEGORICAL = ["regiao", "segmento"]

CLIENT_COLUMNS = [
    ID_COL,
    "idade",
    "renda_mensal",
    "regiao",
    "segmento",
    "tempo_relacionamento_meses",
    "limite_atual",
    "bureau_score",
    "contas_open_finance",
    "produtos_ativos",
    "persona",
    SAFRA_COL,
    CUTOFF_COL,
    TARGET_END_COL,
]

PANEL_COLUMNS = [
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
    "valor_atrasado",
    "dias_sem_transacao",
    "sessoes_app",
    "tempo_medio_sessao",
    "velocidade_scroll",
    "cliques",
    "contatos_cobranca",
    "promessas_pagamento",
    "pagamentos_realizados",
    "valor_pago",
    "utilizacao_limite",
    "saldo_open_finance",
    "entradas_open_finance",
    "saidas_open_finance",
    "score_bureau_mes",
    SAFRA_COL,
    CUTOFF_COL,
    TARGET_END_COL,
]

TARGET_COLUMNS = [
    ID_COL,
    SAFRA_COL,
    CUTOFF_COL,
    TARGET_END_COL,
    *TARGETS,
    "consumo_30d",
    "consumo_performance_medio",
    "elegivel_limite",
    "valor_atrasado_cutoff",
]

REQUIRED_RAW_TABLES = {
    "clientes": CLIENT_COLUMNS,
    "painel": PANEL_COLUMNS,
    "tratamentos": [
        ID_COL,
        "acao",
        "pagou_30d",
        "valor_recuperado",
        "custo_acao",
    ],
    "targets": TARGET_COLUMNS,
}


@dataclass(frozen=True)
class SplitConfig:
    train_safras: tuple[int, ...] = (1, 2)
    validation_safras: tuple[int, ...] = (3, 4)
    test_safras: tuple[int, ...] = (5, 6)

    @property
    def all_safras(self) -> tuple[int, ...]:
        return self.train_safras + self.validation_safras + self.test_safras


def split_name_for_safra(safra: int, config: SplitConfig | None = None) -> str:
    config = config or SplitConfig()
    if safra in config.train_safras:
        return "train"
    if safra in config.validation_safras:
        return "validation"
    if safra in config.test_safras:
        return "test"
    raise ValueError(f"Safra fora do desenho experimental: {safra}")
