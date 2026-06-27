# 🎬 Roteiro Detalhado de Apresentação Técnica (Vídeo STAR - 5 Minutos)

**Público-alvo:** Avaliadores do Tech Challenge (Foco profundo na técnica, engenharia e matemática por trás das decisões).
**Dica para leitura:** Um ritmo normal de fala atinge cerca de 130 a 150 palavras por minuto. Este roteiro tem aproximadamente 700 palavras, perfeito para 5 minutos. Treine lendo em voz alta.

---

## ⏱️ [0:00 - 0:45] S - Situação (O Contexto e o Problema)
*Abra a gravação mostrando o ML Canvas rapidamente e depois mude para o Dashboard Streamlit.*

**Fala sugerida:**
"Olá a todos. Sou o [Seu Nome] e vou apresentar a solução arquitetada pelo nosso grupo para o desafio de Churn da operadora de telecomunicações.
O problema de negócios aqui é claro: a evasão de clientes. Na nossa base histórica, cerca de 27% dos clientes cancelaram seus serviços. O custo financeiro de um 'Falso Negativo' — ou seja, o modelo errar e deixar um cliente cancelar sem enviarmos uma oferta — é gigantesco, estimado em quase 10 vezes o custo de uma campanha de retenção (que seria o nosso Falso Positivo). Portanto, o mandato técnico para este pipeline de Machine Learning era claro desde o início: nós precisávamos **otimizar o Recall**, minimizando ao máximo os Falsos Negativos, e entregar isso através de uma arquitetura pronta para produção, escalável e monitorável."

## ⏱️ [0:45 - 1:45] T - Tarefa (A Engenharia de Dados e Baseline)
*Mostre rapidamente a matriz de correlação ou a tabela de VIF (Variance Inflation Factor).*

**Fala sugerida:**
"Para construir essa solução, começamos pela Engenharia de Dados. Usamos o `pandera` para estabelecer um contrato de dados rígido e evitar *Data Drift* logo na entrada. 
Na análise exploratória, detectamos um problema severo de **Multicolinearidade**. O teste estatístico de *Variance Inflation Factor* (VIF) revelou que a variável `Total Charges` (Cobrança Total) tinha um VIF muito acima de 10 e uma correlação de Pearson de 0.83 com `Tenure Months` (Meses de Contrato). Para evitar que os coeficientes do nosso modelo ficassem instáveis e distorcessem a nossa interpretabilidade, tomamos a decisão técnica de remover `Total Charges`.

Com o dataset limpo e transformado através de um `ColumnTransformer` do scikit-learn — usando *OneHotEncoding* e *StandardScaler* dentro de um Pipeline unificado para evitar *Data Leakage* — rodamos o nosso baseline: um Dummy Classifier. Ele obteve 73% de acurácia, mas zero de Recall. A acurácia alta era uma ilusão causada pelo desbalanceamento das classes. Precisávamos de modelos reais."

## ⏱️ [1:45 - 3:30] A - Ação (Modelagem, PyTorch e Otimização do Threshold)
*Mostre um trecho de código da Rede Neural (PyTorch) ou a UI do MLflow.*

**Fala sugerida:**
"Nós desenvolvemos dois modelos principais e usamos o **MLflow** para rastrear todos os hiperparâmetros, métricas e artefatos, usando um backend SQLite.

**Primeiro, a Rede Neural Artificial (PyTorch):**
Construímos um *Multi-Layer Perceptron* (MLP). Nossa arquitetura tinha a camada de entrada baseada nas features escalonadas, passando por duas camadas ocultas, reduzindo de 64 para 32 neurônios. Usamos a função de ativação **ReLU**, com *Dropout* de 0.3 para regularização. Para o otimizador, escolhemos o **Adam** com *learning rate* de 1e-3, e compilamos a perda usando `BCEWithLogitsLoss`, que é numericamente mais estável do que aplicar uma Sigmoid separada antes da *Binary Cross Entropy*. Para evitar *overfitting*, implementamos *Early Stopping* monitorando a perda de validação com paciência de 10 épocas.

**A Decisão do Threshold (Limiar):**
Aqui entra a principal sacada do projeto. Modelos como a Regressão Logística usam um *threshold* padrão de 0.5. Nós notamos que com 0.5, estávamos perdendo 44% dos canceladores reais. Por isso, reduzimos cirurgicamente o nosso *threshold* para **0.3**. Trocamos um pouco de *Precision* por um salto drástico no *Recall*, porque matematicamente provamos que aceitar mais Falsos Positivos saía mais barato para o negócio.

**O Empate (MLP vs Regressão Logística):**
Quando comparamos a Rede Neural com a nossa Regressão Logística, vimos um empate técnico: ambos bateram cerca de 0.85 de AUC-ROC e 78% de Recall no limiar de 0.3. 
Nossa decisão arquitetural foi: **Colocar a Regressão Logística em produção.** Para dados tabulares com essas características, a Regressão Logística foi igualmente performática, porém muito mais leve de processar e 100% interpretável. Nós conseguimos usar o SHAP (*Shapley Additive exPlanations*) Linear para entender que `Tenure Months` e `Contratos de 2 Anos` eram as features de maior peso na retenção, sem precisar lidar com os estouros de memória que o *DeepExplainer* da Rede Neural estava causando."

## ⏱️ [3:30 - 5:00] R - Resultado (O Produto em Produção e MLOps)
*Abra o Dashboard Streamlit, faça uma inferência e mostre a latência na tela.*

**Fala sugerida:**
"O resultado foi um produto totalmente *end-to-end*. 
Para servir o modelo, criamos uma API REST usando **FastAPI**. Injetamos esquemas do **Pydantic** para validar tipagens estritas em tempo real e criamos um *middleware customizado* que captura a latência de cada requisição, salvando logs estruturados em JSON de forma assíncrona.
Além da predição em lote (*Batch*) feita por scripts Python (prontos para orquestração no Airflow), nós construímos um frontend em **Streamlit** (este que vocês estão vendo). Ele consome a API e plota a probabilidade de Churn e a velocidade de resposta.

Para finalizar, não esquecemos do Dia 2 do modelo (MLOps). Entregamos um repositório que contém um **Model Card** (detalhando vieses e limitações técnicas) e um **Playbook de Monitoramento**, que estabelece as regras de observabilidade para rastrear *Data Drift* via teste de Kolmogorov-Smirnov.

Construímos mais do que um Jupyter Notebook: entregamos inteligência artificial aplicada, validada estatisticamente, monitorável e, acima de tudo, altamente rentável para o negócio. Muito obrigado!"
