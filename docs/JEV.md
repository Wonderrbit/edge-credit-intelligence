# Julgamento com Jev (TypeSafe)

## Papel no sistema

O Jev não substitui o modelo estatístico. Ele adiciona **sinais qualitativos estruturados** sobre um estado já calculado:

- risco (evento adverso, fraude, capacidade, consistência de renda, documentação);
- cobrança (vulnerabilidade, disputa, hardship, canal preferido);
- grounding de RAG (suporte, contradição, completude);
- guardrails de escopo e revisão;
- intenção do usuário no agente.

Cada sinal é composto com pesos em `configs/judgment.yaml`; se a chamada falhar, o fallback determinístico em `src/mlops/judgment/fallback.py` assume.

## Configuração

```env
TYPESAFE_API_KEY=...
TYPESAFE_MODEL=jev-latest
TYPESAFE_BASE_URL=            # opcional
MLOPS_USE_JEV=1
MLOPS_JEV_TIMEOUT=30
MLOPS_JEV_MAX_CALLS_PER_RUN=500
```

- `MLOPS_USE_JEV=0` força modo offline.
- `--dry-run` no autopilot também força offline.
- O `JevClient` usa `RetryPolicy(max_retries=2)` e nunca levanta erro de transporte; retorna `None`.

## Chamada

```python
from mlops.judgment import get_jev
from mlops.judgment.batteries import CLIENT_QUESTIONS

response = get_jev().ask(state, CLIENT_QUESTIONS)
```

`state` é um dicionário com `cliente`, `scores` e `contexto` (ver `build_client_state` em `src/mlops/autopilot/decisions.py`). O ID do cliente é pseudonimizado com SHA-256.

Várias perguntas vão em uma única chamada; o SDK responde em paralelo. Formatos suportados:

- `Noul`: probabilidade de uma afirmação (`noul`).
- `Choice`: escolha entre opções com `confidence` e `probabilities`.
- `Score`: nota ordinal com `confidence` e `probabilities`.

## Composição

| Arquivo | Função |
| --- | --- |
| `batteries.py` | perguntas e opções por domínio |
| `compose.py` | converte respostas em sinais, decisões e verificações |
| `fallback.py` | regras determinísticas equivalentes |
| `schemas.py` | modelos Pydantic de decisão e julgamento |
| `client.py` | wrapper de transporte |

O `mode` da decisão (`offline`/`jev`) e a confiança ficam registrados em cada linha de `decisions.jsonl`.

## Avaliação

```bash
.venv\Scripts\python scripts\eval_judgment.py            # fallback
.venv\Scripts\python scripts\eval_judgment.py --use-jev  # requer chave
```

O relatório vai para `artifacts/eval/judgment_eval.json` com acurácia, macro-F1 e matriz de confusão para risco e cobrança. O dataset rotulado está em `data/judgment_eval.jsonl` (10 casos; expandir para 100+ antes de promover mudanças de threshold).

## Boas práticas

- Nunca enviar PII desnecessária; o estado já é minimizado e pseudonimizado.
- Limitar chamadas por execução para controlar custo e latência.
- Monitorar `jev_fallbacks` e `jev_error_rate`.
- Tratar a saída do Jev como sinal, não como verdade: o modelo e a política continuam sendo a base auditável.
- Revisar thresholds com dados reais antes de aumentar autonomia.
