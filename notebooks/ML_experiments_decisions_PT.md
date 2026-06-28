# Experimentos de ML — Registro de Decisões

## Decisão 1: Redução do threshold de classificação para 0.3

### Contexto

A Regressão Logística gera uma **probabilidade** de churn por cliente, não um rótulo absoluto. Um threshold (limiar) converte essa probabilidade em uma previsão binária:

- probabilidade ≥ threshold → prevê **Churn (1)**
- probabilidade < threshold → prevê **Não Churn (0)**

O threshold padrão do `sklearn` é **0.5** ("mais provável do que não").

### Problema com o threshold padrão

O dataset é desbalanceado: ~73% dos clientes **não** cancelaram. O modelo aprende essa premissa e se torna conservador — ele só dispara uma previsão de churn quando tem bastante confiança, deixando muitos clientes reais que cancelaram com probabilidades previstas na faixa de 0.30–0.49 e classificando-os erroneamente como "Não Churn".

Resultados com threshold = 0.5:

| Métrica | Valor |
|---|---|
| Recall (Churn) | 0.559 |
| Precision (Churn) | 0.655 |
| F1 (Churn) | 0.603 |
| AUC-ROC | 0.856 |

O alto AUC-ROC (0.856) confirma que o modelo **ranqueia** bem os casos de churn — ele atribui probabilidades maiores a eles do que aos que não cancelaram. O baixo Recall é puramente um problema de threshold, não de qualidade do modelo.

### Justificativa de Negócios

Para o cenário de telecomunicações, os dois tipos de erro têm custos assimétricos:

| Tipo de Erro | O que acontece | Custo Típico |
|---|---|---|
| Falso Negativo (perder um churner) | O cliente cancela, perda permanente de receita | Alto |
| Falso Positivo (sinalizar um não-churner) | Oferta de retenção enviada a quem não ia cancelar | Baixo–Médio |

Perder um cliente custa muito mais do que desperdiçar uma oferta de retenção. Portanto, **um Recall maior é preferível**, mesmo que custe uma queda na Precision.

### Threshold Escolhido: 0.3

Com base nos custos estimados (`COST_FN = 500`, `COST_FP = 50`), a razão de custo é de 10:1. Teoricamente, o threshold ótimo que minimiza o custo financeiro esperado seria `~0.09` (`50 / (50 + 500)`). 

No entanto, um threshold tão baixo flagraria a esmagadora maioria da base de clientes como propensa a Churn, gerando um volume de falsos positivos que **inviabilizaria a operação real da equipe de retenção** (devido a restrições de orçamento e capacidade de atendimento).

Dessa forma, o threshold de **0.3** foi selecionado para conciliar a matemática teórica com a praticidade de negócios: ele abandona a métrica padrão (0.5), prioriza agressivamente o Recall em direção ao ótimo teórico, mas mantém a Precision em um nível operacionalmente viável.

---

## Decisão 2: Remoção de Total Charges por Multicolinearidade

### Evidências

Dois diagnósticos foram computados nas variáveis numéricas antes do treinamento:

**Correlação de Pearson (r):**

| Par | r |
|---|---|
| Tenure Months x Total Charges | **0.83** |

**Variance Inflation Factor (VIF):**

| Variável | VIF |
|---|---|
| Total Charges | > 10 (problemático) |
| Tenure Months | elevado |
| Monthly Charges | moderado |
| CLTV | baixo |

Um `VIF > 10` é o limite estatístico padrão que indica que a variância de uma feature está tão inflada pela correlação com outras que as estimativas dos seus coeficientes se tornam não confiáveis.

### Por que isso importa

A variável `Total Charges` (Cobrança Total) é em grande parte resultado de `Tenure Months × Monthly Charges`. Incluí-la junto com `Tenure Months` cria informações redundantes no espaço de features. Para a regressão logística, isso causa:

- Erros padrão inflados em ambas as variáveis correlacionadas
- Instabilidade generalizada nos coeficientes calculados

### Decisão

**Remover Total Charges.** A variável `Tenure Months` foi mantida porque é uma métrica mais direta e interpretável da lealdade do cliente. A informação perdida é mínima, pois `Monthly Charges` (também mantida) reconstrói parcialmente o sinal financeiro.

---

## Observação 1: Interpretação dos valores SHAP na Regressão Logística

Após a remoção de `Total Charges`, a distorção da colinearidade foi resolvida e a importância das variáveis via SHAP se tornou limpa e interpretável. Abaixo está a interpretação de negócios de cada variável significativa.

### Valores SHAP (média \|SHAP\|) — Regressão Logística no threshold 0.3

| Variável | Média \|SHAP\| | Interpretação |
|---|---|---|
| Tenure Months | 0.778 | O preditor mais forte. Clientes muito antigos raramente cancelam — a lealdade aumenta com o tempo. Tempo de casa curto é o sinal mais claro de churn. |
| Dependents_Yes | 0.577 | Clientes com dependentes cancelam significativamente menos. Configurações familiares valorizam estabilidade e o custo de troca é maior. |
| Internet Service_Fiber optic | 0.531 | Clientes de fibra ótica cancelam mais. Este é provavelmente o segmento mais competitivo onde provedores rivais oferecem velocidades similares, facilitando a troca. |
| Contract_Two year | 0.448 | Contratos de 2 anos ancoram fortemente os clientes. O compromisso contratual é um mecanismo de retenção direto. |
| Contract_One year | 0.228 | Contratos de 1 ano também reduzem churn em relação aos mensais, mas com menos força que os de 2 anos. |
| Monthly Charges | 0.218 | Contas mensais altas aumentam o risco de churn. A sensibilidade a preço é um fator secundário atrás de lealdade e tipo de contrato. |
| Streaming TV_Yes | 0.173 | Contribuição moderada para o churn. Clientes usando streaming podem ser mais tecnológicos e dispostos a trocar de provedor. |
| Multiple Lines_Yes | 0.165 | Leve sinal de churn. Pode estar correlacionado a mensalidades mais altas. |
| Paperless Billing_Yes | 0.148 | Leve sinal de churn positivo. Clientes digitalmente engajados podem fazer mais pesquisas de concorrência. |
| Partner_Yes | 0.137 | Ter parceiro(a) reduz o churn ligeiramente. Dinâmica parecida com dependentes — contas divididas têm mais atrito de troca. |
| Online Security_Yes / Tech Support_Yes | ~0.125–0.137 | Clientes inscritos em serviços agregados cancelam menos. Estão mais "presos" ao ecossistema do provedor. |

### Variáveis com contribuição SHAP negligenciável (candidatas à remoção no futuro)

| Variável | Média \|SHAP\| | Nota |
|---|---|---|
| CLTV | 0.014 | O valor projetado do ciclo de vida adiciona quase nenhum sinal além do que tempo de casa e cobranças já capturam. |
| Gender_Male | 0.009 | Gênero não é um preditor significativo de churn neste dataset. |
| Online Backup_Yes | 0.007 | Efetivamente sem contribuição. |

### Principais conclusões de negócios

1. **Alavancas de retenção com maior ROI:** converter clientes mensais para contratos anuais ou de dois anos, e promover serviços de valor agregado (segurança, suporte) para aumentar a retenção ao ecossistema.
2. **Perfil de alto risco:** clientes novos (baixo *tenure*), em contratos mensais, usando Fibra ótica, sem dependentes ou parceiro(a).
3. **Latitude e Longitude** (0.061 / 0.055) mostram sinais marginais — efeitos geográficos existem, mas são fracos para um modelo linear. Um modelo de árvore de decisão poderia extrair mais dos dados espaciais.
