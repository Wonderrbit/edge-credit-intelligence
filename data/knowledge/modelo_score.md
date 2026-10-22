# Modelo de Score e Governança

**Documento:** modelo_001  
**Versão:** 2026.1  
**Vigência:** 2026-01-01  
**Área responsável:** Data Science

## 1. Arquitetura

O modelo Edge é uma rede neural multi-task com tronco GRU e quatro cabeças: pagamento em 30 dias, inadimplência em 90 dias, aceite de limite e consumo alto. O modelo compartilha representação entre as tarefas, reduz custo e permite inferência no dispositivo.

## 2. Janelas

- Features: 12 meses da safra.
- Performance: 6 meses seguintes ao corte.
- Treino: safras 1 e 2.
- Validação: safras 3 e 4.
- Teste: safras 5 e 6.

## 3. Métricas

- ROC-AUC e PR-AUC para discriminação.
- KS para separação em crédito.
- Brier, log loss e ECE para calibração.
- PSI para estabilidade.
- SHAP e gradiente para explicabilidade.
- Fairness por segmento e região.

## 4. Calibração

A calibração por temperatura é ajustada na validação e aplicada no serving e no Edge. O objetivo é alinhar probabilidade prevista e frequência observada.

## 5. Governança

- Model card obrigatório.
- Registro de versão, hash e métricas.
- Revisão independente antes de produção.
- Monitoramento mensal.
- Plano de rollback.
- Revisão humana para decisões adversas.

## 6. Limitações

O ambiente atual usa dados sintéticos. Métricas não devem ser interpretadas como evidência de desempenho em produção. A validação com dados reais e a análise de safras são obrigatórias antes de qualquer uso comercial.
