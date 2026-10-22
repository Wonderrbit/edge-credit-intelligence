WITH base AS (
    SELECT
        p.*,
        c.renda_mensal,
        c.limite_atual
    FROM painel p
    JOIN clientes c USING (client_id)
    WHERE p.mes BETWEEN p.safra AND p.safra + 11
),
agg AS (
    SELECT
        client_id,
        max(atraso_dias) AS atraso_max_12m,
        sum(CASE WHEN atraso_dias > 0 THEN 1 ELSE 0 END) AS meses_com_atraso,
        max(atraso_dias) FILTER (WHERE mes > cutoff_month - 3) AS atraso_max_90d,
        sum(contatos_cobranca) FILTER (WHERE mes > cutoff_month - 3) AS contatos_90d,
        sum(promessas_pagamento) FILTER (WHERE mes > cutoff_month - 3) AS promessas_90d,
        sum(valor_pago) FILTER (WHERE mes > cutoff_month - 3)
            / NULLIF(sum(valor_atrasado) FILTER (WHERE mes > cutoff_month - 3), 0) AS taxa_pagamento_90d,
        cutoff_month - max(mes) FILTER (WHERE pagamentos_realizados > 0) AS meses_desde_ultimo_pagamento,
        sum(valor_atrasado) AS valor_em_atraso_12m,
        max(valor_atrasado) AS maior_valor_em_atraso,
        avg(contatos_cobranca) AS contatos_medio_mes,
        avg(promessas_pagamento) AS promessas_medio_mes,
        sum(promessas_pagamento) / NULLIF(sum(contatos_cobranca), 0) AS taxa_promessa,
        sum(pagamentos_realizados) / NULLIF(sum(compras_qtd + 1), 0) AS taxa_pagamento_historica
    FROM base
    GROUP BY client_id, cutoff_month
)
SELECT
    a.*,
    a.meses_com_atraso / 12.0 AS frequencia_atraso,
    a.valor_em_atraso_12m / NULLIF(b.renda_mensal * 12, 0) AS atraso_sobre_renda
FROM agg a
JOIN clientes b USING (client_id)
