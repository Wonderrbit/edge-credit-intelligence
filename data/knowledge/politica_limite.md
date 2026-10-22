# Política de Limite de Crédito

**Documento:** limite_001  
**Versão:** 2026.1  
**Vigência:** 2026-01-01  
**Área responsável:** Produtos de Crédito

## 1. Objetivo

Definir critérios para recomendar aumento, manutenção ou redução de limite, equilibrando utilização, risco e rentabilidade.

## 2. Presets de política

### Conservador

- Limite máximo: 1,5 vez a renda mensal.
- Crescimento máximo: 25%.
- Peso de risco: 35%.
- Peso de aceite: 25%.
- Peso de consumo: 15%.

### Equilibrado

- Limite máximo: 2,0 vezes a renda mensal.
- Crescimento máximo: 60%.
- Peso de risco: 25%.
- Peso de aceite: 35%.
- Peso de consumo: 25%.

### Crescimento

- Limite máximo: 2,8 vezes a renda mensal.
- Crescimento máximo: 110%.
- Peso de risco: 15%.
- Peso de aceite: 45%.
- Peso de consumo: 35%.

## 3. Fórmula de recomendação

A recomendação combina propensão a aceitar o aumento, propensão de consumo e risco de inadimplência.

`score_politica = peso_aceite × p_aceite + peso_consumo × p_consumo − peso_risco × p_default + 0,35`

`limite_recomendado = min(limite_atual × (1 + (crescimento_máximo − 1) × score), renda × múltiplo_máximo)`

O limite nunca deve ser inferior ao limite atual em uma recomendação de aumento. Reduções exigem revisão humana.

## 4. Avaliação econômica

- Receita esperada: utilização esperada × margem.
- Perda esperada: utilização esperada × probabilidade de default × LGD.
- Lucro esperado: receita esperada − perda esperada.
- Margem padrão: 3,5% ao mês.
- LGD padrão: 60%.
- Taxa de uso: 55% do aumento.

## 5. Restrições

- Não recomendar aumento para clientes com atraso grave ativo.
- Não recomendar aumento acima de 2,0 vezes a renda no preset equilibrado.
- Clientes com utilização acima de 90% e risco alto devem ser revisados.
- Recomendações de redução devem ser explicadas e revisadas por humano.

## 6. Monitoramento

Acompanhar utilização do limite, aceite, inadimplência, receita e perda por safra. Comparar a política com um grupo de controle quando houver experimento. Ajustes de preset exigem aprovação de Produto e Risco.
