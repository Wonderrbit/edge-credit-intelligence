# LGPD e decisões automatizadas

## Escopo

O repositório usa dados **sintéticos**. Em produção, o sistema processaria dados financeiros e cadastrais de pessoas naturais, incluindo renda, atraso, limite e transações. Isso exige tratamento como dados pessoais, potencialmente sensíveis no contexto financeiro.

## Bases legais e finalidade

- Execução de contrato e legítimo interesse para gestão de crédito e cobrança, com avaliação de impacto e balanceamento documentados.
- Consentimento quando exigido para canais específicos (ex.: WhatsApp) e para finalidades secundárias.
- Minimização: enviar ao Jev apenas o estado necessário, com `id_pseudonimo` (SHA-256) em vez do identificador direto.
- Retenção: definir prazo por artefato (decisões, auditoria, relatórios) e expurgo automático.

## Decisões automatizadas

O modo `MLOPS_AUTONOMY=full` executa decisões adversas sem aprovação prévia. Para conformidade com o Art. 20 da LGPD, o desenho deve garantir:

1. **Transparência**: registrar critérios, versão de política/modelo e fatores relevantes.
2. **Revisão**: canal para o titular solicitar revisão humana de decisão automatizada.
3. **Contestação**: fluxo de reclamação com prazo e responsável.
4. **Explicabilidade**: motivos legíveis por decisão (`risk.reasons`, `collection.reasons`) e SHAP/feature importance quando aplicável.
5. **Auditoria**: `audit.jsonl` encadeado por hash, retenção e acesso restrito.
6. **Supervisão**: responsável designado, métricas de erro e revisão amostral periódica.

A fila `review_queue.jsonl` é um gate **informativo**; ela não bloqueia a execução. Se a política da empresa exigir bloqueio, o consumidor da decisão deve implementar aprovação antes da ação.

## Segurança

- Chaves em `.env` local ou Secrets Manager; nunca no git.
- S3 com criptografia, versionamento e bloqueio público.
- API key e, em produção, mTLS/OIDC e least privilege.
- Logs sem PII e sem payload completo de Jev/OpenAI.
- Fornecedores (OpenAI/TypeSafe) devem ter DPA, região adequada e retenção configurada.

## Riscos e mitigação

| Risco | Mitigação |
| --- | --- |
| Viés e discriminação | monitorar fairness por segmento; revisar features; auditoria amostral |
| Vazamento de PII | pseudonimização, minimização, logs redigidos |
| Decisão opaca | razões estruturadas, model card, revisão humana |
| Drift | PSI/ECE, alertas, retreino com validação |
| Uso indevido de autonomia | kill-switch `--dry-run`, orçamento Jev, alertas e runbook |
| Retenção excessiva | política de expurgo por tipo de artefato |

## Checklist antes de produção

- [ ] Base legal e DPIA aprovadas.
- [ ] Contrato/DPA com OpenAI e TypeSafe.
- [ ] Canal de revisão e contestação publicado.
- [ ] Retenção e expurgo definidos.
- [ ] Teste de segurança e revisão de acessos.
- [ ] Métricas de fairness e drift em dados reais.
- [ ] Runbook e responsáveis de plantão.
