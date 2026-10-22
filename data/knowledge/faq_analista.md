# FAQ do Analista

**Documento:** faq_001  
**Versão:** 2026.1  
**Vigência:** 2026-01-01  
**Área responsável:** Analytics

## 1. O que significa cada score?

- **Pagou 30d:** probabilidade de o cliente pagar o valor em atraso em até 30 dias.
- **Inadimplente 90d:** probabilidade de atraso grave em 90 dias. Quanto menor, melhor.
- **Aceita limite:** probabilidade de aceitar ou usar um aumento de limite.
- **Consumo alto:** probabilidade de consumir acima do perfil esperado.

## 2. Como interpretar o limite recomendado?

O limite recomendado é uma sugestão da política, não uma aprovação automática. Ele considera aceite, consumo, risco, renda e limite atual. Reduções e casos de alto risco exigem revisão humana.

## 3. Como priorizar cobrança?

Priorize por valor esperado, não apenas por valor devido. Um cliente com dívida alta e probabilidade baixa de pagamento pode ter EV menor que um cliente com dívida média e alta probabilidade.

## 4. O que é PSI?

PSI mede a mudança na distribuição de uma variável ou score entre dois períodos. Valores abaixo de 0,10 são estáveis; entre 0,10 e 0,20 merecem atenção; acima de 0,20 exigem investigação.

## 5. O que é calibração?

Calibração significa que, se o modelo prevê 20% para um grupo, aproximadamente 20% desse grupo deve apresentar o evento. Medimos com Brier score, ECE e curva de confiabilidade.

## 6. Por que usar validação temporal?

Porque o comportamento muda ao longo do tempo. Um split aleatório pode vazar informação e superestimar o desempenho. Usamos safras: treino, validação e teste em períodos distintos.

## 7. O agente pode negar crédito?

Não. O agente recomenda, explica e simula. Decisões adversas exigem revisão humana. O agente não executa ações irreversíveis.

## 8. O que fazer se o modelo divergir da política?

Escalar para revisão humana, registrar a divergência e verificar se a política está atualizada. Não alterar thresholds sem aprovação.
