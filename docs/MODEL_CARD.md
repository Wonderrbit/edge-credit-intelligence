# Model Card — edge_credit_multitask

## Visão

Modelo multitarefa de crédito ao consumo para uma carteira sintética. Prevê pagamento em 30 dias, inadimplência em 90 dias, aceite de limite e consumo alto. A saída alimenta políticas de crédito e cobrança; não é um score de crédito regulado.

## Dados

- Fonte: `data/raw/{clientes,painel,targets}.parquet` (sintéticos).
- Features: 20 estáticas + 21 sequenciais.
- Split: treino safras 1–2, validação 3–4, teste 5–6.
- Clientes no registry: 600; base de serving: 201.
- Não há PII real. Em produção, a base deve passar por minimização, retenção e controle de acesso.

## Arquitetura

Rede Keras multitarefa com torres compartilhadas e cabeças por alvo; ~7.172 parâmetros. Calibração por temperatura por alvo. Export TF.js para inferência no edge e serving via API.

## Métricas de teste (registry 2026.10.22.1457)

| Alvo | ROC-AUC | PR-AUC | KS | Brier | ECE pós-calibração |
| --- | --- | --- | --- | --- | --- |
| `pagou_30d` | 0.732 | 0.655 | 0.444 | 0.214 | 0.108 |
| `inadimplente_90d` | 0.858 | 0.536 | 0.576 | 0.105 | 0.039 |
| `aceita_limite` | 0.648 | 0.540 | 0.297 | 0.248 | 0.109 |
| `consumo_alto` | 0.630 | 0.373 | 0.228 | 0.218 | 0.143 |

`pagou_30d` teve ECE pior após calibração (0.088 → 0.108); a temperatura foi mantida para consistência do pipeline, mas o alvo deve ser recalibrado antes de uso em produção.

## Uso pretendido

- Priorização de cobrança e decisão de limite em ambiente controlado.
- Geração de indicadores de carteira e simulações.
- Não usar como único critério para negar crédito sem explicabilidade e canal de contestação.

## Limitações

- Dados sintéticos: métricas não representam performance real.
- Base de teste pequena (201 clientes), intervals de confiança amplos.
- Drift e fairness não foram medidos em dados reais.
- Jev adiciona sinais qualitativos, mas não substitui validação estatística.
- O modelo não foi submetido a validação regulatória.

## Monitoramento

- PSI da distribuição de score e ECE da PD 90d.
- Taxa de aprovação, delinquência e valor em atraso versus execução anterior.
- Alertas em `configs/autopilot.yaml`; relatório por execução.

## Retreinamento

Executar `make pipeline`, comparar métricas com o registry e promover apenas se não houver degradação material. Toda promoção deve registrar versão, hash, métricas e decisão.
