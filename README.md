# Edge Credit Intelligence
 
 Transforma dados de clientes em decisões de crédito, limite e cobrança, com modelo calibrado, políticas, agente RAG, revisão humana e autopilot auditável via API e Docker.

[![CI](https://github.com/Wonderrbit/edge-credit-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Wonderrbit/edge-credit-intelligence/actions/workflows/ci.yml)

Protótipo de motor de decisão para crédito e cobrança. O projeto combina um modelo multitarefa de risco, políticas de negócio, indicadores de carteira e um agente com RAG e julgamento estruturado. A saída principal é uma API para consumo por outros sistemas, com artefatos de auditoria e um autopilot executável por linha de comando.

O repositório é destinado a avaliação técnica e testes controlados. Ele não é um sistema pronto para decisão de crédito em produção. Os dados são sintéticos e as métricas não substituem validação em uma carteira real.

## Escopo

O sistema:

- pontua clientes com o modelo `edge_credit_multitask`, implementado em TensorFlow/Keras;
- calcula PD 90d, probabilidade de pagamento em 30d, aceite de limite e propensão a consumo;
- converte scores em decisões de risco, limite e cobrança;
- usa Jev/TypeSafe para sinais qualitativos e OpenAI para narrativa executiva, com fallback determinístico;
- executa um autopilot com trilha de auditoria encadeada por hash;
- expõe API, servidor MCP read-only e artefatos versionados por execução;
- inclui Docker, CI e infraestrutura Terraform para uma implantação de laboratório.

Fora do escopo atual:

- decisão de crédito real sem revisão humana;
- tratamento de PII de produção;
- validação regulatória, fairness e monitoramento contínuo em dados reais;
- streaming, feature store e processamento distribuído.

## Arquitetura

```text
serving_clients.parquet / data/raw
        |
        v
  fontes e indicadores  ->  políticas de crédito e cobrança
        |                              |
        +--> Jev (sinais qualitativos) |
        |                              v
        +--> OpenAI (narrativa)  ->  decisão + auditoria
                                       |
                          API /v1/* <--+--> S3 / MCP
```

O fluxo detalhado está em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Dados e modelo

O pacote `demo_assets/` contém uma base sintética de 600 clientes e os artefatos necessários para executar a API sem treinar o pipeline. O split de treino, validação e teste usa as safras definidas em `configs/config.yaml`.

O modelo é uma rede pequena com entrada sequencial e estática. A saída tem quatro cabeças: pagamento em 30 dias, inadimplência em 90 dias, aceite de limite e consumo alto. As probabilidades passam por calibração de temperatura.

Métricas do conjunto de teste sintético:

| Tarefa | ROC-AUC | PR-AUC | KS |
| --- | ---: | ---: | ---: |
| Pagamento 30d | 0,732 | 0,655 | 0,444 |
| Inadimplência 90d | 0,858 | 0,536 | 0,576 |
| Aceite de limite | 0,648 | 0,540 | 0,297 |
| Consumo alto | 0,630 | 0,373 | 0,228 |

Esses números servem para verificar o funcionamento do pipeline, não para inferir desempenho em produção. O model card está em [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Decisão e políticas

O score não é a decisão final. A camada de política aplica thresholds, calcula limite recomendado e estima o valor esperado de cobrança. As regras ficam em `configs/` e são carregadas pela API e pelo autopilot.

O autopilot registra, para cada cliente:

- scores calibrados;
- decisão de risco, limite e cobrança;
- razões e flags de revisão;
- valor esperado e prioridade;
- hashes de política e artefatos.

Decisões adversas entram na fila de revisão. Quando `MLOPS_AUTONOMY=full`, a revisão é informativa e não bloqueia a execução.

## Agente, RAG e TOON

O agente usa LangGraph, recuperação híbrida e um `ContextualRetriever` que inclui chunks vizinhos do mesmo documento. As citações continuam apontando para o chunk foco. O índice lexical fica em `demo_assets/rag/` e pode ser reconstruído a partir de `data/knowledge/`.

Os payloads de julgamento podem ser serializados em TOON, com fallback automático para JSON. O pacote `toon-format` é opcional em runtime: se a importação ou a codificação falhar, o agente continua com JSON.

Variáveis úteis:

```env
MLOPS_USE_EMBEDDINGS=0
MLOPS_USE_RERANK=0
MLOPS_USE_EXPANSION=1
MLOPS_USE_CONTEXTUAL=1
MLOPS_CONTEXT_WINDOW=1
MLOPS_CONTEXT_MAX_CHARS=2400
```

## API

| Método | Rota | Descrição |
| --- | --- | --- |
| GET | `/health` | liveness |
| GET | `/model/info` | versão, features, métricas e thresholds |
| GET | `/clients` | clientes da base de serving |
| GET | `/score/{client_id}` | score individual |
| POST | `/score` | score por features |
| POST | `/score/batch` | score em lote |
| GET | `/collection/queue` | fila de cobrança |
| POST | `/agent/ask` | agente com RAG e fallback offline |
| GET | `/metrics` | métricas de serving |
| GET | `/monitoring` | PSI e ECE |
| GET | `/v1/indicators/latest` | último portfólio processado |
| GET | `/v1/decisions` | decisões da última execução |
| GET | `/v1/runs` | execuções disponíveis |
| GET | `/v1/runs/{run_id}` | manifest de uma execução |
| POST | `/v1/autopilot/run` | dispara o autopilot |
| GET | `/edge` | artefatos estáticos do modelo TF.js |

Swagger em `/docs`. Se `API_KEY` estiver definida, as rotas protegidas exigem o header `X-API-Key`.

## Instalação e teste offline

Pré-requisitos: Python 3.12, Docker opcional e `make` opcional.

Windows:

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python scripts\make_demo_assets.py
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\run_autopilot.py --mode full --dry-run --limit 50
.venv\Scripts\python scripts\run_api.py
```

Linux ou macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/make_demo_assets.py
python -m pytest -q
python scripts/run_autopilot.py --mode full --dry-run --limit 50
python scripts/run_api.py
```

A demo já inclui os artefatos TF.js, então o fluxo acima não exige `tensorflowjs`. Para reexportar o modelo:

```bash
python -m pip install -r requirements-export.txt
python -m pip install --no-deps tensorflowjs==4.22.0
python scripts/run_export.py
```

A API sobe em `http://127.0.0.1:8000`. Em outro terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/score/C000001
```

O comando `make assets-demo` executa a mesma hidratação de artefatos no Windows. O script não sobrescreve artefatos existentes; use `--force` apenas quando quiser restaurar o pacote de demonstração.

## Teste com OpenAI e Jev

Sem chaves, o sistema opera em modo offline e usa o fallback determinístico. Para validar a integração online:

1. Crie `.env` a partir de `.env.example`.

```bat
copy .env.example .env
```

2. Preencha `OPENAI_API_KEY` e `TYPESAFE_API_KEY`. Ative os provedores:

```env
MLOPS_USE_LLM=1
MLOPS_USE_JEV=1
```

3. Valide a conectividade:

```bat
.venv\Scripts\python scripts\smoke_ai.py --require
```

4. Rode uma execução pequena:

```bat
.venv\Scripts\python scripts\run_autopilot.py --mode full --limit 25
```

5. Confira o resultado em `artifacts/autopilot/latest.json` e no `manifest.json` da execução. Os campos `jev_calls`, `jev_fallbacks` e `estimated_cost_usd` mostram chamadas, falhas e custo estimado.

As chaves nunca devem ser commitadas. O `.env` está no `.gitignore` e a imagem Docker não recebe credenciais durante o build. Em Docker, use `--env-file .env` em tempo de execução. Se você recebeu um pacote de ambiente criptografado, decifre-o apenas no ambiente de teste e mantenha o arquivo resultante fora do controle de versão.

## Autopilot

```bat
.venv\Scripts\python scripts\run_autopilot.py --mode full --dry-run --limit 50
.venv\Scripts\python scripts\run_autopilot.py --mode full
```

Modos: `full`, `indicators`, `judge` e `report`. `--dry-run` desliga chamadas externas e não envia webhook. O limite de chamadas Jev por execução é controlado por `MLOPS_JEV_MAX_CALLS_PER_RUN`.

Artefatos por execução:

| Arquivo | Conteúdo |
| --- | --- |
| `manifest.json` | versões, contagens, custo estimado, alertas e hash de política |
| `indicators.parquet` | carteira enriquecida por cliente |
| `portfolio.json` | indicadores agregados e comparação com a execução anterior |
| `decisions.jsonl` | decisão de risco e cobrança, razões, EV e flags |
| `review_queue.jsonl` | itens para auditoria humana |
| `alerts.json` | alertas operacionais |
| `report.md` | relatório executivo |
| `audit.jsonl` | trilha encadeada por `prev_hash` e `hash` |

Se `MLOPS_ARTIFACTS_BUCKET=s3://bucket/prefixo` estiver definido, a execução também é enviada ao S3.

## Docker

O Dockerfile hidrata `artifacts/` a partir de `demo_assets/` durante o build. Um clone limpo não precisa treinar o modelo para testar a API.

```bat
docker build -t edge-credit-intelligence:local .
docker run --rm -p 8000:8000 edge-credit-intelligence:local
```

Com IA online:

```bat
docker run --rm -p 8000:8000 --env-file .env edge-credit-intelligence:local
```

Imagens publicadas:

```bat
docker login ghcr.io
docker pull ghcr.io/wonderrbit/edge-credit-intelligence:latest
docker pull ghcr.io/wonderrbit/edge-credit-intelligence:slim
```

Se o pacote GHCR estiver privado, use o build local acima ou ajuste a visibilidade em **Package settings**. A imagem completa tem cerca de 4,03 GB. A variante slim tem cerca de 2,83 GB e foi validada com health, score, agente e autopilot offline. Para reconstruir a slim, consulte [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Testes e CI

```bat
.venv\Scripts\python -m ruff check src scripts tests
.venv\Scripts\python -m pytest -q
```

A suíte cobre API, autopilot, julgamento, MCP, RAG contextual, TOON e carregamento do pacote de demonstração. O CI executa lint, testes, autopilot offline e smoke do container. Os resultados da última validação estão em [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Segurança e LGPD

- Todos os dados do repositório são sintéticos.
- Chaves devem ficar em `.env` local, Secrets Manager ou secrets de CI, nunca no Git.
- A decisão automatizada de crédito exige base legal, explicabilidade, canal de contestação e revisão humana.
- O sistema registra decisões e evidências, mas não substitui governança de crédito.
- O modelo foi treinado em dados sintéticos e precisa de validação em dados reais antes de qualquer uso produtivo.

Mais detalhes em [`docs/LGPD.md`](docs/LGPD.md) e [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Limitações conhecidas

- A base de teste é curta e sintética.
- O modelo atual é pequeno e não passou por validação de fairness.
- O autopilot não tem orquestração distribuída nem exactly-once.
- A API lê artefatos locais; em Fargate, o compartilhamento entre API e autopilot precisa de EFS ou leitura via S3.
- O Terraform é um exemplo de laboratório e não foi aplicado em uma conta AWS real.
- A imagem slim depende de profiling com o slim; novos imports dinâmicos exigem revisão dos `--include-path`.

## Estrutura

```text
src/mlops/
  agents/       grafo LangGraph, RAG, agente e TOON
  autopilot/    fontes, indicadores, decisões, store e relatórios
  judgment/     cliente Jev, batteries, composição e fallback
  mcp/          servidor MCP read-only
  serving/      API FastAPI e predictor
configs/        políticas, julgamento, indicadores e autopilot
demo_assets/    artefatos sintéticos para execução imediata
scripts/        CLIs de treino, avaliação, autopilot, MCP e demo
infra/terraform/ infraestrutura AWS de laboratório
docs/           arquitetura, model card, runbook, LGPD e validação
```

## Autor

Luiz Dranka, engenheiro de ML/AI.

- GitHub: https://github.com/Wonderrbit
- LinkedIn: https://www.linkedin.com/in/luiz-dranka-ba637620b/

Este projeto foi concebido e implementado por mim para exercitar o ciclo completo de um produto de decisão: modelagem, calibração, política, serving, agentes, auditoria, CI e implantação.

O que o repositório demonstra:

| Área | Evidência |
| --- | --- |
| Modelagem e avaliação | Modelo multitarefa, calibração e métricas de ranking, calibração e estabilidade |
| Decisão | Separação entre score, política, limite, cobrança e revisão humana |
| Agentes | LangGraph, RAG contextual, TOON e fallback determinístico |
| MLOps | API, autopilot, registry, artefatos versionados, auditoria e monitoramento |
| Infraestrutura | Docker, CI, Terraform, ECS Fargate, EventBridge e redução da imagem de 4,03 GB para 2,83 GB |
| Segurança | Dados sintéticos, minimização de PII, LGPD e limites explícitos |

## Licença

Proprietário. O código é público apenas para avaliação técnica; uso comercial exige autorização. Consulte [`LICENSE`](LICENSE).
