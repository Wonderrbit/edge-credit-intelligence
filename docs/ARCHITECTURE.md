# Arquitetura

## Visão geral

O sistema tem cinco camadas:

1. **Dados e features**: `data/raw/*.parquet`, `sql/` e `src/mlops/data/` geram a base de serving (`artifacts/serving/serving_clients.parquet`).
2. **Modelo**: `edge_credit_multitask` (Keras) prevê quatro alvos: `pagou_30d`, `inadimplente_90d`, `aceita_limite`, `consumo_alto`. Calibração por temperatura e thresholds ficam em `artifacts/registry.json`.
3. **Decisão**: `src/mlops/autopilot/` calcula indicadores por cliente, aplica políticas (`configs/policies.yaml`) e produz decisões de risco e cobrança.
4. **Julgamento**: `src/mlops/judgment/` envia estados estruturados ao Jev e compõe sinais qualitativos (risco, vulnerabilidade, disputa, evidência, grounding). Se Jev falhar ou estiver desligado, o fallback determinístico assume.
5. **Entrega**: `serving/api.py` (API), `mcp/server.py` (MCP), `report.py`/`narrative.py` (relatório) e `storage.py` (S3).

## Fluxo do autopilot

```text
fetch clients
  -> validate schema/qualidade
  -> compute_client_indicators
  -> compute_portfolio_indicators
  -> compare com execução anterior
  -> (opcional) Jev por cliente em paralelo com orçamento de chamadas
  -> compose risk/collection decisions
  -> review_queue + alerts + custo estimado
  -> report.md + manifest.json + audit.jsonl
  -> upload S3 (opcional)
```

### Determinismo e versionamento

- `policy_version` = SHA-256 de `configs/policies.yaml` + `configs/judgment.yaml`.
- `model_version` vem de `artifacts/registry.json`.
- Cada decisão carrega `decision_id` (hash de cliente + inputs + política), `mode` (`offline`/`jev`) e versões.
- `audit.jsonl` é encadeado: cada linha inclui `prev_hash` e `hash` do conteúdo.

### Falhas e degradação

| Falha | Comportamento |
| --- | --- |
| Jev indisponível/timeout | fallback determinístico; `jev_fallbacks` no manifest; alerta se taxa > threshold |
| OpenAI indisponível | relatório sem narrativa; decisões não são afetadas |
| S3 indisponível | execução local preservada; `artifacts_uri` ausente |
| Fonte ausente | erro explícito antes de decidir |
| Schema inválido | `data_quality.json` + alerta; decisão segue apenas se as colunas obrigatórias existirem |

## Autonomia

`MLOPS_AUTONOMY=full` (default) permite que decisões adversas sejam executadas e apenas registradas na fila de auditoria. O campo `manual_review_recommended` é informativo. Para um modo assistido, basta mudar a configuração e o consumidor passar a exigir aprovação antes de executar a ação.

## Observabilidade

- `manifest.json`: contagens, latência, modo de julgamento, custo estimado, alertas, versões.
- `alerts.json`: delinquência, PSI, taxa de aprovação, erro Jev e custo.
- `/metrics` e `/monitoring`: métricas do modelo e PSI/ECE.
- Logs estruturados via CloudWatch em produção.

## Escalabilidade

- O cálculo por cliente é vetorizado em pandas; o gargalo é Jev (paralelizado com `max_workers` e orçamento por execução).
- Para carteiras maiores, particionar por safra/segmento e rodar workers em paralelo, mantendo um `run_id` pai.
- A API é stateless; o autopilot é um job batch. Em AWS, a API roda em ECS Service e o job em ECS Task agendada por EventBridge.

## Segurança

- API key opcional (`API_KEY`) via header `X-API-Key`.
- Segredos em `.env` local ou Secrets Manager em produção.
- S3 com versionamento, criptografia AES256 e bloqueio de acesso público.
- Dados sintéticos no repositório; nenhuma PII real deve ser commitada.
