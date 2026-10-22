WITH base AS (
    SELECT
        p.*,
        c.renda_mensal,
        c.idade,
        c.tempo_relacionamento_meses,
        c.limite_atual,
        c.bureau_score,
        c.contas_open_finance,
        c.produtos_ativos,
        c.regiao,
        c.segmento
    FROM painel p
    JOIN clientes c USING (client_id)
    WHERE p.mes BETWEEN p.safra AND p.safra + 11
),
agg AS (
    SELECT
        client_id,
        safra,
        cutoff_month,
        target_end_month,
        count(*) AS meses_observados,
        sum(valor_total) AS consumo_total_12m,
        avg(valor_total) AS consumo_medio_mensal,
        stddev_samp(valor_total) AS consumo_volatilidade,
        avg(ticket_medio) AS ticket_medio_12m,
        sum(compras_qtd) AS compras_total_12m,
        avg(compras_qtd) AS compras_media_mes,
        cutoff_month - max(mes) FILTER (WHERE compras_qtd > 0) AS recencia_compra_meses,
        avg(categoria_diversidade) AS categoria_diversidade_media,
        avg(parcelas_medias) AS parcelas_medias_12m,
        avg(pct_credito) AS pct_credito_medio,
        avg(pct_debito) AS pct_debito_medio,
        avg(pct_pix) AS pct_pix_medio,
        avg(pct_boleto) AS pct_boleto_medio,
        avg(utilizacao_limite) AS utilizacao_media,
        max(utilizacao_limite) AS utilizacao_max,
        avg(atraso_dias) AS atraso_medio,
        max(atraso_dias) AS atraso_max,
        avg(dias_sem_transacao) AS dias_sem_transacao_media,
        avg(sessoes_app) AS sessoes_media,
        avg(tempo_medio_sessao) AS tempo_sessao_medio,
        avg(velocidade_scroll) AS velocidade_scroll_media,
        sum(cliques) AS cliques_total,
        sum(contatos_cobranca) AS contatos_total,
        sum(promessas_pagamento) AS promessas_total,
        sum(pagamentos_realizados) AS pagamentos_total,
        sum(valor_pago) AS valor_pago_total,
        sum(valor_atrasado) AS valor_atrasado_total,
        sum(valor_pago) / NULLIF(sum(valor_atrasado + valor_pago), 0) AS taxa_pagamento,
        avg(saldo_open_finance) AS saldo_open_finance_medio,
        sum(entradas_open_finance) AS entradas_open_finance_total,
        sum(saidas_open_finance) AS saidas_open_finance_total,
        avg(score_bureau_mes) AS score_bureau_medio,
        avg(valor_total) FILTER (WHERE mes > cutoff_month - 3)
            - avg(valor_total) FILTER (WHERE mes <= safra + 2) AS tendencia_consumo,
        avg(utilizacao_limite) FILTER (WHERE mes > cutoff_month - 3)
            - avg(utilizacao_limite) FILTER (WHERE mes <= safra + 2) AS tendencia_utilizacao,
        avg(valor_total) FILTER (WHERE mes > cutoff_month - 3) AS consumo_medio_3m,
        avg(sessoes_app) FILTER (WHERE mes > cutoff_month - 3) AS sessoes_media_3m
    FROM base
    GROUP BY client_id, safra, cutoff_month, target_end_month
)
SELECT
    a.*,
    b.renda_mensal,
    b.idade,
    b.tempo_relacionamento_meses,
    b.limite_atual,
    b.bureau_score,
    b.contas_open_finance,
    b.produtos_ativos,
    b.regiao,
    b.segmento,
    a.consumo_medio_mensal / NULLIF(b.renda_mensal, 0) AS comprometimento_consumo,
    a.utilizacao_media * b.limite_atual AS exposicao_media,
    a.consumo_total_12m / NULLIF(b.renda_mensal * 12, 0) AS consumo_sobre_renda
FROM agg a
JOIN clientes b USING (client_id)
