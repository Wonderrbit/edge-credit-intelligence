# Política de Cobrança e Recuperação

**Documento:** cobranca_001  
**Versão:** 2026.1  
**Vigência:** 2026-01-01  
**Área responsável:** Cobrança

## 1. Objetivo

Definir a priorização de clientes, os canais de contato e as regras de negociação para maximizar a recuperação de crédito com custo controlado e experiência adequada.

## 2. Priorização por valor esperado

A priorização deve considerar a probabilidade de pagamento, o valor em atraso, a fração recuperável e o custo do contato.

Fórmula de valor esperado:

`EV = p_pagamento × valor_em_atraso × fração_recuperável − custo_do_contato`

A fração recuperável padrão é 65%. O custo de contato padrão é R$ 12,00. Clientes com EV negativo não devem ser acionados por telefone.

## 3. Canais e custos

| Canal | Custo unitário | Uplift esperado |
|---|---|---|
| Nenhum | R$ 0,00 | 0% |
| Lembrete | R$ 2,00 | 4% |
| WhatsApp | R$ 5,00 | 9% |
| Ligação | R$ 18,00 | 16% |
| Negociação | R$ 35,00 | 25% |

O canal deve ser escolhido pelo maior retorno esperado, não apenas pelo maior uplift.

## 4. Cadência

- Atraso de 1 a 7 dias: lembrete automatizado.
- Atraso de 8 a 30 dias: WhatsApp e oferta de parcelamento.
- Atraso de 31 a 60 dias: ligação e negociação.
- Acima de 60 dias: negociação estruturada, com aprovação para descontos.
- Clientes crônicos: revisão humana antes de nova campanha.

## 5. Negociação

Descontos e parcelamentos devem respeitar o valor presente e a capacidade de pagamento. Promessas de pagamento devem ser registradas com data, valor e canal. Quebra de promessa reduz a prioridade de novos benefícios.

## 6. Uplift e experimentação

Sempre que possível, comparar o resultado do canal escolhido com um grupo de controle. Medir uplift, recuperação, custo por real recuperado e efeito no relacionamento. A política deve ser reavaliada a cada safra.

## 7. Controles

- Não realizar contato fora do horário permitido.
- Respeitar pedidos de não perturbação.
- Não revelar dívida a terceiros.
- Registrar toda ação em auditoria.
- Encaminhar casos de vulnerabilidade para atendimento especializado.

## 8. Métricas

- Taxa de recuperação em 30 dias.
- Custo por real recuperado.
- Uplift por canal.
- Quebra de promessa.
- Reclamações e reclamações procedentes.
- Taxa de revisão humana.
