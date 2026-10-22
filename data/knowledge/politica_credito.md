# Política de Crédito e Risco

**Documento:** credito_001  
**Versão:** 2026.1  
**Vigência:** 2026-01-01  
**Área responsável:** Risco de Crédito

## 1. Objetivo

Estabelecer critérios para concessão, revisão e monitoramento de crédito ao consumidor, combinando dados de consumo, relacionamento e Open Finance. A política orienta analistas e sistemas de decisão, mas não substitui a análise humana em casos sensíveis.

## 2. Princípios

- Decisões devem ser baseadas em dados disponíveis na data da análise, sem uso de informação futura.
- O score de risco é um insumo, não a única evidência.
- Toda decisão adversa deve ser explicável, auditável e revisável.
- Dados sensíveis devem ser minimizados e protegidos conforme a LGPD.
- Modelos devem ser calibrados para que a probabilidade prevista reflita a frequência observada.

## 3. Faixas de risco

| Probabilidade de inadimplência em 90 dias | Classificação | Decisão padrão |
|---|---|---|
| Até 12% | Baixo risco | Aprovar |
| 12% a 18% | Médio risco | Aprovar com limite conservador ou revisar |
| Acima de 18% | Alto risco | Reprovar ou encaminhar para revisão humana |

O threshold de aprovação padrão é 18%. A faixa de revisão começa em 12%. Ajustes de política exigem aprovação da área de Risco.

## 4. Critérios de elegibilidade

- Renda mensal comprovada ou estimada com confiança suficiente.
- Limite atual e comprometimento compatíveis com a renda.
- Ausência de atraso grave não regularizado.
- Relacionamento mínimo de 1 mês, salvo exceções aprovadas.
- Contas Open Finance com consentimento válido.

## 5. Uso de Open Finance

Dados de Open Finance podem enriquecer a análise com saldo, entradas, saídas e recorrência. O uso deve respeitar o consentimento, a finalidade e o prazo de retenção. Sinais de fluxo de caixa não podem ser usados para discriminação ilegal.

## 6. Revisão humana

Casos de reprovação, redução de limite, cobrança escalada ou divergência relevante entre score e evidências devem ser encaminhados para revisão humana. O revisor registra a justificativa e o resultado.

## 7. Monitoramento

O modelo deve ser monitorado mensalmente com PSI de features e score, calibração, AUC, KS e taxa de revisão humana. PSI acima de 0,20 exige investigação. Queda de AUC acima de 5 pontos percentuais em relação à referência exige retreinamento ou champion/challenger.

## 8. Explicabilidade

Toda decisão automatizada deve permitir identificar os principais fatores. Para modelos de árvore, usar SHAP. Para a rede Edge, usar atribuição por gradiente. As explicações devem ser compreensíveis para o analista e para o cliente, sem revelar segredos do modelo.

## 9. Limitações

O modelo é treinado com dados sintéticos neste ambiente de demonstração. Em produção, a política deve ser revalidada com dados reais, análise de safras e validação independente de risco.
