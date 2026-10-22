SELECT
    safra,
    cutoff_month,
    target_end_month,
    count(*) AS clientes,
    avg(pagou_30d) AS taxa_pagamento,
    avg(inadimplente_90d) AS taxa_inadimplencia,
    avg(aceita_limite) AS taxa_aceite_limite,
    avg(consumo_alto) AS taxa_consumo_alto
FROM targets
GROUP BY safra, cutoff_month, target_end_month
ORDER BY safra
