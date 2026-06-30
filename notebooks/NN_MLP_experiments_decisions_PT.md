# Experimentos com NN MLP — Registro de Decisões

## Decisão 1: Arquitetura da MLP — tamanho das camadas (Input → 64 → 32 → 1)

### Tamanho da camada de entrada (Input size)

A dimensão de entrada é determinada automaticamente a partir da matriz de features após o pré-processamento: um neurônio por feature (colunas numéricas + dummies via one-hot encoding). Nenhum ajuste manual é necessário.

### Camadas ocultas (Hidden layers)

Foram escolhidas duas camadas ocultas: **64 neurônios → 32 neurônios**.

| Opção | Raciocínio |
|---|---|
| Uma única camada oculta | Suficiente para dados linearmente separáveis, mas limita a capacidade para efeitos de interação |
| **64 → 32 (escolhido)** | Duas camadas permitem que a rede aprenda uma combinação de features na primeira camada e uma representação compactada na segunda |
| Mais profunda (ex. 128 → 64 → 32) | Exagero para ~7.000 linhas; aumenta o risco de overfitting e instabilidade no treinamento |

O formato de pirâmide (largo → estreito) é uma heurística padrão para dados tabulares: a primeira camada extrai combinações de features, a segunda camada as destila em uma representação compacta antes da saída.

Essa arquitetura também espelha o MLP existente no projeto em `notebooks/02_neural_network.ipynb` (`64 → 32`), mantendo os experimentos comparáveis.

### Camada de saída (Output layer)

É utilizado um único neurônio de saída **sem função de ativação**. O logit bruto é passado para a `BCEWithLogitsLoss` durante o treinamento (numericamente mais estável do que aplicar a sigmóide primeiro), e a `torch.sigmoid()` é aplicada no momento da inferência para obter as probabilidades.

---

## Decisão 2: Função de Ativação — ReLU

**ReLU (Rectified Linear Unit)** foi escolhida em detrimento das alternativas:

| Ativação | Motivos a favor/contra |
|---|---|
| Sigmoid / Tanh | Saturam para inputs muito grandes → dissipação do gradiente (vanishing gradients) em redes mais profundas |
| **ReLU (escolhido)** | Sem saturação para valores positivos, computacionalmente barata, empiricamente uma configuração padrão forte para MLPs tabulares |
| LeakyReLU | Resolve o problema da ReLU inativa (dying ReLU); benefício marginal para redes rasas como esta |
| ELU / GELU | Mais sofisticadas, mas adicionam complexidade desnecessária para apenas duas camadas ocultas |

ReLU é o padrão da indústria para camadas ocultas em redes *feedforward* em dados tabulares e não requer otimização de hiperparâmetros.

---

## Decisão 3: Taxa de Dropout — 0.3

O Dropout zera aleatoriamente as ativações durante o treinamento com probabilidade `p`, agindo como um regularizador que previne a coadaptação dos neurônios.

| Taxa | Raciocínio |
|---|---|
| 0.0 (sem dropout) | Maior risco de overfitting em ~5.600 amostras de treinamento |
| **0.3 (escolhido)** | Regularização moderada; ponto de partida padrão para dados tabulares |
| 0.5 | Comum em Visão Computacional/NLP, mas muitas vezes agressivo demais para datasets tabulares pequenos, prejudicando a convergência |

O Dropout é aplicado após a ativação ReLU de cada camada oculta, e não antes da camada de saída (o que distorceria a escala final do logit).

---

## Decisão 4: Função de Perda (Loss function) — BCEWithLogitsLoss

`BCEWithLogitsLoss` combina uma camada sigmóide e a Entropia Cruzada Binária (Binary Cross-Entropy) em uma única operação numericamente estável. É a escolha padrão para classificação binária no PyTorch.

Utilizar `BCELoss` com uma ativação sigmóide no neurônio de saída é matematicamente equivalente, mas numericamente inferior — o truque log-sum-exp dentro do `BCEWithLogitsLoss` evita overflow/underflow para valores extremos de logits.

---

## Decisão 5: Otimizador — Adam com lr=1e-3

**Adam** (Adaptive Moment Estimation) foi escolhido em relação ao SGD e suas variantes:

| Otimizador | Raciocínio |
|---|---|
| SGD | Requer ajuste cuidadoso da taxa de aprendizado e *momentum*; convergência mais lenta |
| SGD + Momentum | Melhor generalização em alguns cenários, mas requer mais ajustes |
| **Adam lr=1e-3 (escolhido)** | Taxas de aprendizado adaptativas por parâmetro; converge de forma confiável com as configurações padrão; escolha comum para MLPs tabulares |
| AdamW | Adiciona decaimento de peso (weight decay) desacoplado da atualização do gradiente; um excelente próximo passo caso seja observado overfitting |

A taxa de aprendizado (learning rate) `1e-3` é o padrão do Adam e se mostra empiricamente robusta em uma ampla gama de problemas tabulares.

---

## Decisão 6: Tamanho do Lote (Batch size) — 64

| Batch size | Raciocínio |
|---|---|
| Full batch | Atualizações determinísticas, sem estocasticidade; convergência lenta e tende a mínimos acentuados |
| 16–32 | Alto ruído no gradiente; convergência mais lenta |
| **64 (escolhido)** | Padrão convencional; equilibra a qualidade do gradiente e a velocidade de treinamento; cabe confortavelmente na memória |
| 256+ | Treinamento mais rápido por época, mas com sinal de validação mais ruidoso em um dataset pequeno |

Com ~5.600 amostras de treinamento, um batch size de 64 gera ~88 atualizações de gradiente por época — o suficiente para um aprendizado estável.

---

## Decisão 7: Máximo de épocas e Early Stopping (150 épocas, patience=10)

### Máximo de épocas — 150

Um teto de 150 épocas foi estabelecido para limitar o pior cenário de tempo de treinamento. Na prática, o *early stopping* (parada antecipada) é acionado bem antes desse limite.

### Early stopping — patience=10

O *Early stopping* monitora a **perda de validação (validation loss)** após cada época e interrompe o treinamento quando ela não apresenta melhorias por `patience` épocas consecutivas, restaurando em seguida os melhores pesos encontrados até então.

Isso tem dois propósitos:
1. **Evitar overfitting** — o modelo não continua aprendendo o ruído do treinamento após o sinal de validação estagnar.
2. **Remover a necessidade de pré-especificar as épocas** — o treinamento termina quando os dados dizem para parar, não em um número arbitrário fixo.

A paciência (patience) de 10 foi escolhida para:
- Permitir que o otimizador escape de platôs locais (uma paciência curta como 3–5 pode parar o treinamento cedo demais durante um solavanco temporário).
- Não esperar tanto tempo a ponto de acumular um overfitting significativo (patience > 20 neste tamanho de dataset corre esse risco).

---

## Decisão 8: Limiar de Classificação (Threshold) — 0.3

O mesmo racional documentado em `ML_experiments_decisions.md` (Decisão 1). Repetido aqui para manter o documento independente:

O dataset é desbalanceado (~73% Não Churn). O limiar (threshold) padrão de 0.5 faz com que o modelo deixe passar ~44% dos churners reais. Reduzir para 0.3 troca Precisão por Recall, o que é a decisão correta quando o custo de perder um churner (perda de receita permanente) é maior do que o custo de uma oferta de retenção desperdiçada.

O Threshold = 0.3 é aplicado de forma consistente em todos os modelos deste projeto para uma comparação justa.

---

## Decisão 9: Método SHAP — DeepExplainer

O `shap.GradientExplainer` foi inicialmente utilizado, mas causou o travamento (crash) do kernel neste hardware devido à pressão na memória (200 amostras de fundo × 300 amostras de teste × cálculo de gradientes).

O `shap.DeepExplainer` foi escolhido como substituto:
- Desenvolvido especificamente para modelos de *deep learning*.
- Utiliza um dataset de fundo para aproximar os valores SHAP por meio de um algoritmo de retropropagação modificado.
- Pegada de memória significativamente menor do que o GradientExplainer para as mesmas quantidades de amostras.
- O dataset de fundo foi reduzido para **50 amostras**, e as amostras de teste para **100**, de modo a permanecer dentro dos limites de memória.

A redução no número de amostras significa que os valores do SHAP são aproximações de um subconjunto dos testes, não da distribuição completa. Devem ser tratados como indicadores direcionais, e não como atribuições exatas.

---

## Observação 1: Comparação de Modelos — Dummy Classifier vs Regressão Logística vs MLP

Todos os três modelos foram avaliados no mesmo conjunto de teste separado (hold-out de 20%) com threshold = 0.3.

### Tabela de Resumo

| Métrica | Dummy Classifier | Regressão Logística | Rede Neural MLP |
|---|---|---|---|
| Acurácia (Accuracy) | ~0.73 | ~0.77 | ~0.76 |
| Precisão Churn (Precision) | 0.00 | ~0.54 | ~0.54 |
| Recall Churn | 0.00 | ~0.78 | ~0.76 |
| F1-Score (Churn) | 0.00 | ~0.64 | ~0.63 |
| AUC-ROC | 0.50 | ~0.856 | ~0.858 |
| PR-AUC | ~0.27 | ~0.690 | ~0.682 |

*(Os valores exatos dependem da rodada; consulte a saída do notebook para ver os números atuais)*

### Principais descobertas

1. **O Dummy Classifier é inútil para retenção** — 0 de recall nos churners significa que ele nunca sinaliza ninguém para intervenção. Sua precisão de 73% é totalmente explicada por sempre prever a classe majoritária. Ele serve apenas como um teste de sanidade de limite inferior (baseline).

2. **A Regressão Logística e a MLP têm desempenhos quase idênticos** — as diferenças estão dentro da margem de ruído (< 0.01 na maioria das métricas). Isso é esperado para um dataset tabular de ~7.000 linhas com features em sua maioria binárias (após One-Hot Encoding). A capacidade não linear da MLP adiciona um valor marginal sobre um separador linear.

3. **A vantagem de AUC-ROC para a MLP é insignificante** — o AUC-ROC da MLP é cerca de ~0.002 superior, bem abaixo do que seria operacionalmente relevante no negócio.

4. **Recomendação para este dataset**: A Regressão Logística é o modelo preferido — desempenho equivalente, mais rápida de treinar, totalmente interpretável via `shap.LinearExplainer`, e não requer otimização de hiperparâmetros.

---

## Observação 2: Análise de Trade-off de Custos — Falsos Positivos vs Falsos Negativos

### Tipos de erros no contexto de churn

| Erro | Definição | Consequência para o negócio |
|---|---|---|
| **Falso Negativo (FN)** | Churner previsto como "Não Churn" | Cliente vai embora sem ser contatado — perda total da receita |
| **Falso Positivo (FP)** | Cliente leal previsto como "Churn" | Oferta de retenção enviada para cliente leal — custo da oferta desperdiçado, mas o cliente fica |

### Custos Assumidos (valores ilustrativos — devem ser substituídos por valores reais de negócio)

| Parâmetro | Valor | Raciocínio |
|---|---|---|
| `COST_FN` | $500 | Receita mensal média aproximada × meses esperados de perda de receita |
| `COST_FP` | $50 | Custo aproximado de uma oferta padrão de retenção (desconto, voucher, call center) |

Estes valores são constantes configuráveis no notebook (`COST_FN`, `COST_FP`).

### O que a tabela de custos mostra

A tabela de custos detalha as contagens de VP / VN / FP / FN para cada modelo e calcula:

- **Custo de FN** = número de churners perdidos × `COST_FN`
- **Custo de FP** = número de alarmes falsos × `COST_FP`
- **Custo Total** = Custo FN + Custo FP

Considerando a proporção de custo assumida de 10:1 (FN/FP), os modelos que maximizam o recall (LR e MLP com threshold=0.3) incorrem em um custo total significativamente menor do que o Dummy Classifier, apesar de gerarem mais falsos positivos.

### O que o gráfico de sensibilidade mostra

A curva de sensibilidade plota o custo total estimado em função do `COST_FN` (de $100 a $1.500), mantendo o `COST_FP` fixo em $50. Principais conclusões:

- **Para qualquer custo de FN superior a ~$200**, tanto a LR quanto a MLP são substancialmente mais baratas que o Dummy Classifier.
- **As curvas da LR e da MLP estão quase sobrepostas** — a pequena diferença nas contagens de FN/FP entre elas não cria uma discrepância de custo significativa na escala do dataset de testes.
- **A diferença entre o modelo Dummy e os modelos reais aumenta linearmente** de acordo com o custo do FN — quanto mais valioso for cada cliente, mais importante se torna o Recall.

### Threshold e Custo

O Threshold de 0.3 foi escolhido para melhorar o recall (menos FNs) à custa de mais FPs. O gráfico de sensibilidade torna esse *trade-off* claro e justificado: sob a premissa de um custo de 10:1, aceitar mais FPs para capturar mais FNs é a escolha economicamente racional. Se o `COST_FP` subir significativamente (ex: ligações de outbound extremamente caras), o limiar de ponto de equilíbrio se deslocaria para cima.
