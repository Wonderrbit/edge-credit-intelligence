# Privacidade, LGPD e Segurança da Informação

**Documento:** lgpd_001  
**Versão:** 2026.1  
**Vigência:** 2026-01-01  
**Área responsável:** Segurança e Privacidade

## 1. Princípios

- Minimização de dados: coletar apenas o necessário para a finalidade informada.
- Finalidade: usar dados de consumo e Open Finance para crédito, limite e cobrança.
- Transparência: informar o cliente sobre o uso de dados e decisões automatizadas.
- Segurança: proteger dados em repouso e em trânsito.
- Prestação de contas: manter trilhas de auditoria.

## 2. Edge AI e privacidade

O scoring pode ser executado no dispositivo do cliente ou em ambiente controlado, reduzindo a exposição de dados sensíveis. Quando o modelo roda no navegador, os dados de comportamento não precisam ser enviados para a nuvem. Apenas o resultado agregado pode ser compartilhado, conforme a finalidade.

## 3. Dados pessoais e mascaramento

- CPF, CNPJ, e-mail, telefone e cartão devem ser mascarados em logs e prompts.
- O agente não deve receber PII desnecessária.
- Dados de outros clientes nunca podem aparecer em respostas.
- Consultas de política não devem incluir dados pessoais.

## 4. Retenção

- Dados brutos: conforme prazo regulatório e contrato.
- Features e scores: pelo tempo necessário à auditoria e ao monitoramento.
- Logs de agente: 90 dias, salvo obrigação legal.
- Artefatos de modelo: versionados e mantidos enquanto o modelo estiver em produção.

## 5. Direitos do titular

O titular pode solicitar confirmação de tratamento, acesso, correção, anonimização, portabilidade e revisão de decisões automatizadas. O atendimento deve registrar a solicitação e o prazo.

## 6. Segurança do agente

- Allowlist de ferramentas.
- Validação de entrada e saída.
- Proteção contra injeção de instruções.
- Limite de passos e timeout.
- Revisão humana para decisões adversas.
- Auditoria de execuções com run_id.

## 7. Incidentes

Qualquer suspeita de vazamento deve ser registrada e escalada para Segurança. O agente não deve tentar corrigir ou ocultar incidentes.
