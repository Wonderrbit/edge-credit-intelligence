# Runbook

## Pré-requisitos

- Python 3.12 e `.venv` com `requirements-dev.txt`.
- `artifacts/serving/serving_clients.parquet` e `artifacts/registry.json` presentes.
- `.env` com `OPENAI_API_KEY` e `TYPESAFE_API_KEY` rotacionadas (ou modo offline).

## Operação diária

1. Verificar conectividade:
   ```bash
   .venv\Scripts\python scripts\smoke_ai.py
   ```
2. Rodar dry-run e conferir o relatório:
   ```bash
   .venv\Scripts\python scripts\run_autopilot.py --mode full --dry-run --limit 50
   ```
3. Rodar produção:
   ```bash
   .venv\Scripts\python scripts\run_autopilot.py --mode full
   ```
4. Validar `artifacts/autopilot/runs/<run_id>/manifest.json`:
   - `jev_calls`/`jev_fallbacks`
   - `estimated_cost_usd`
   - `alerts`
   - `policy_version` e `model_version`
5. Se houver alerta `high`, abrir o `report.md` e a `review_queue.jsonl` antes de comunicar a operação.

## Agendamento

- Windows: `powershell -ExecutionPolicy Bypass -File scripts/schedule_autopilot.ps1 -Time 09:00`
- Linux/cron: `0 9 * * * cd /app && python scripts/run_autopilot.py --mode full`
- GitHub Actions: `.github/workflows/autopilot.yml` (diário, usa secrets do repositório).
- AWS: EventBridge `cron(0 9 * * ? *)` provisionado no Terraform.

## Deploy AWS

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform apply
```

Depois do apply:

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t <ecr_repository_url>:latest .
docker push <ecr_repository_url>:latest
aws secretsmanager put-secret-value --secret-id akropoli-credit/openai --secret-string '{"OPENAI_API_KEY":"..."}'
aws secretsmanager put-secret-value --secret-id akropoli-credit/typesafe --secret-string '{"TYPESAFE_API_KEY":"..."}'
aws ecs update-service --cluster akropoli-credit --service akropoli-credit-api --force-new-deployment
```

O task definition espera os segredos como JSON com as chaves `OPENAI_API_KEY` e `TYPESAFE_API_KEY`; o ECS injeta o valor bruto. Se preferir string simples, ajuste o `valueFrom`/`secrets` ou use `aws secretsmanager get-secret-value` no entrypoint.

## Rotação de chaves

1. Gerar novas chaves no provedor.
2. Atualizar `.env` local e/ou Secrets Manager.
3. Rodar `scripts/smoke_ai.py --require`.
4. Forçar novo deployment da API e do autopilot.
5. Revogar as chaves antigas.

## Limitação de armazenamento no ECS

As tasks Fargate são efêmeras. A API lê `artifacts/` local (baseline embutida na imagem) e o autopilot grava em S3. Para a API enxergar o último run em produção, monte um EFS compartilhado em `/app/artifacts` nas duas tasks ou implemente leitura read-through do S3. O Terraform de exemplo não provisiona EFS.

## Incidentes

| Sintoma | Ação |
| --- | --- |
| `jev_error_rate` alto | verificar chave/timeout; o fallback já manteve a execução; investigar antes de aumentar autonomia |
| Custo acima do limite | reduzir `MLOPS_JEV_MAX_CALLS_PER_RUN` ou `max_workers`; rodar `--dry-run` |
| PSI > 0.25 | congelar promoção de modelo; investigar drift; comparar safras |
| Queda de aprovação > 10% | revisar políticas e dados de entrada; abrir auditoria |
| API 5xx | checar logs CloudWatch, saúde da task e secrets |
| Execução sem artefatos | validar volume/S3 e permissões da task role |

## Rollback

- API: `aws ecs update-service --task-definition <revisão-anterior>`.
- Modelo: restaurar `artifacts/registry.json` e `artifacts/models/` da versão anterior; reiniciar API.
- Política: reverter `configs/policies.yaml`/`judgment.yaml`; o `policy_version` muda automaticamente.

## Custos

`estimated_cost_usd` no manifest usa tokens de entrada estimados e preço configurado em `configs/autopilot.yaml`. Para controle rígido, defina orçamento no provedor e alerte em `max_cost_usd`.
