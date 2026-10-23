# Validação

Execução local em 2026-10-22 (Windows 11, Python 3.12, Docker 29.6.1, dados sintéticos).

## Qualidade

| Check | Resultado |
| --- | --- |
| `ruff check src scripts tests` | passou |
| `pytest -q` | 28 testes passaram |
| Clone limpo | hidratação com 21 arquivos e 28 testes passaram |
| `scripts/make_demo_assets.py` | 21 arquivos no pacote, hidratação sem sobrescrever artefatos |
| `scripts/run_autopilot.py --dry-run` | 201 clientes, 0 alertas, artefatos completos |
| `scripts/run_autopilot.py` sem chave Jev | fallback determinístico e alerta de erro de Jev |
| RAG contextual | chunks vizinhos no contexto e citações no chunk foco |
| TOON | encode/decode e fallback JSON cobertos por teste |
| API `/health`, `/model/info`, `/score/C000001` | ok |
| API `/agent/ask`, `/v1/runs`, `/v1/indicators/latest`, `/v1/decisions` | ok |
| Preditor com `demo_assets/` | carrega modelo, serving, sequências e calibração |
| HCL Terraform | parse ok, não aplicado |

## IA online

| Check | Resultado |
| --- | --- |
| OpenAI | `gpt-4.1-mini` conectado |
| Jev/TypeSafe | `jev-latest` conectado; `policy` com confiança 0.93 e 397,7 ms |
| Autopilot online | run `20261022T190334Z-d1bf8d`: 201 decisões, 201 chamadas Jev, 0 fallbacks, custo estimado US$ 0,025306, 0 alertas |

As chaves usadas nessa validação não fazem parte do repositório e não são incluídas na imagem Docker.

## Docker

| Imagem | Tamanho | Validação |
| --- | --- | --- |
| `edge-credit-intelligence:local` | 4,03 GB | health, `/model/info`, `/score/C000001`, `/agent/ask`, autopilot offline |
| `edge-credit-intelligence:slim` | 2,83 GB (~29,8% menor) | health, `/model/info`, `/v1/runs`, score, agente e autopilot offline |

O Dockerfile hidrata `artifacts/` a partir de `demo_assets/` durante o build. Um clone limpo não precisa treinar o modelo para subir a API. O contexto Docker exclui reranker, runs, logs e dados brutos.

A imagem slim foi gerada com o binário atual do slim, target na rede `bridge`, `--include-path` para `src/mlops/agents` e scripts de agente/autopilot, probes HTTP e execução de agente e autopilot. O exec do slim usa `MLOPS_AUTOPILOT_OUTPUT=/tmp/artifacts/autopilot` para não deixar diretórios de runtime com dono `root` na imagem.

No Docker 29, o slim encerra o relatório com `unsupported value: NaN` e ID mismatch após gerar a imagem. A imagem resultante foi inspecionada, executada como usuário `app` e validada. O `slim` antigo (`dslim/slim:latest`) não é compatível com a API mínima 1.40 e foi descartado.

As imagens full e slim foram varridas para `.env`, caminhos `C:\Users` e bytecode com caminho local. Nenhum arquivo suspeito foi encontrado. O pacote de ambiente cifrado foi decifrado em arquivo temporário e comparado por SHA-256 com o `.env` original antes de ser apagado.

## Publicação

| Item | Resultado |
| --- | --- |
| Repositório público | `Wonderrbit/edge-credit-intelligence` publicado |
| GitHub Actions | qualidade e Docker passaram no run `39874939972` |
| GHCR | `latest` e `slim` publicados; pacote privado, com build local documentado |
| Clone limpo | validado localmente a partir do commit `0caf1a2` |
| Pacote de ambiente | `.env` cifrado fora do repositório; hash de decifragem conferido |

## Pendências para produção

1. Rotacionar as chaves usadas nos testes online e manter os novos valores fora do Git.
2. Aplicar o Terraform e validar ECS, EventBridge, S3 e Secrets Manager em uma conta AWS de laboratório.
3. Ampliar `data/judgment_eval.jsonl` para 100+ casos e validar thresholds em dados reais.
4. Avaliar fairness, drift e estabilidade em uma carteira real antes de qualquer decisão automatizada.
5. Revisar os `--include-path` do slim ao adicionar novos módulos com import dinâmico.
