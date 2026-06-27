# 🎬 Roteiro de Vídeo (Método STAR) - Pitch do Projeto Telco Churn
**Duração Estimada:** 5 Minutos
**Público-alvo:** Professor / Avaliadores (Foco Técnico e de Negócios)

---

## ⏱️ [0:00 - 1:00] S - Situação (Contexto)
**Objetivo:** Explicar o problema de negócio e a dor da empresa.

* **Fala sugerida:** 
  > "Olá! Nosso projeto aborda um problema crítico para operadoras de telecomunicações: a alta taxa de evasão de clientes, o famoso *Churn*. A perda de um cliente gera um prejuízo enorme em receita recorrente, muito maior do que o custo de oferecer um desconto para tentar retê-lo. Identificamos que atuar reativamente não era suficiente. Nós precisávamos prever quem iria cancelar o plano antes que o cancelamento de fato ocorresse, permitindo que a equipe de retenção atuasse de forma cirúrgica e proativa."

## ⏱️ [1:00 - 1:30] T - Tarefa (O que precisava ser feito)
**Objetivo:** Mostrar o escopo do projeto.

* **Fala sugerida:**
  > "A nossa tarefa era construir um pipeline de Machine Learning de ponta a ponta. Não queríamos entregar apenas um Jupyter Notebook estático, mas sim um produto de dados real. Precisávamos de um modelo altamente focado em **Recall**, pois no nosso contexto de negócios, um *Falso Negativo* (deixar um cliente cancelar sem avisar) custa cerca de 10 vezes mais do que um *Falso Positivo* (dar um desconto para alguém que já ia ficar). O objetivo final era servir essa inteligência em uma API robusta e um Dashboard para as áreas de negócio."

## ⏱️ [1:30 - 3:30] A - Ação (O que fizemos e nossas Decisões)
**Objetivo:** Brilhar tecnicamente, explicar a Rede Neural vs Regressão Logística e justificar todas as escolhas.

* **Fala sugerida:**
  > "Para atacar o problema, desenvolvemos todo o pré-processamento de dados e treinamos diferentes arquiteturas rastreando tudo via MLflow.
  >
  > **1. A Escolha do Modelo (Rede Neural vs Regressão Logística):**
  > Desenvolvemos uma Rede Neural Multi-Layer Perceptron (MLP) em PyTorch com duas camadas ocultas usando ativação ReLU e Early Stopping. Além disso, criamos um baseline sólido com Regressão Logística. 
  > *Nossa Decisão:* Avaliando as métricas de teste, ambos empataram com um AUC-ROC em torno de 0.85 e um Recall de 78%. Optamos por colocar a **Regressão Logística em produção**. Por que? Porque para dados tabulares simples, ela entregou a mesma performance da Rede Neural, mas sendo muito mais leve e, o mais importante: **100% interpretável**.
  > 
  > **2. Ajuste do Threshold (Limiar de Decisão):**
  > Por padrão, modelos usam 0.5 de probabilidade para classificar um Churn. Nós percebemos que isso estava nos fazendo perder quase metade dos canceladores reais. *Nossa Decisão:* Reduzimos o threshold para **0.3**. Assumimos que o custo de perder um cliente é altíssimo, então preferimos aceitar mais Falsos Positivos para garantir que a grande maioria dos potenciais cancelamentos fossem capturados.
  >
  > **3. Tratamento de Colinearidade:**
  > Identificamos via matriz de correlação e teste VIF que a variável *Total Charges* era altamente correlacionada com os meses de contrato (*Tenure*). *Nossa Decisão:* Removemos *Total Charges* para evitar distorções nos coeficientes e nas explicações do modelo, mantendo apenas o tempo de contrato que é um sinal direto de lealdade.
  >
  > **4. Engenharia e Deploy:**
  > Construímos uma API com FastAPI, validada rigorosamente com Pydantic para evitar que dados incorretos quebrem a inferência. Adicionamos um middleware de latência no backend e criamos um Dashboard interativo em Streamlit."

## ⏱️ [3:30 - 5:00] R - Resultado (O que entregamos)
**Objetivo:** Fechar com chave de ouro mostrando os artefatos de governança e a conclusão.

* **Fala sugerida:**
  > "Como resultado, entregamos uma solução completa. Hoje, a área de negócios acessa o Dashboard, insere os dados e tem a resposta de risco com a latência computacional em tempo real na tela.
  > 
  > Mais do que código, nós entregamos **Governança de IA**. Documentamos o projeto conforme as melhores práticas de MLOps: criamos um ML Canvas conectando o modelo ao negócio, um Model Card detalhando as limitações da IA, e um Playbook de Monitoramento preparado para tratar cenários futuros.
  > 
  > Com essas decisões — desde o ajuste de threshold focado no negócio até a escolha pragmática da Regressão Logística — conseguimos maximizar o retorno financeiro da operadora e criar uma infraestrutura de produção sólida. Muito obrigado!"

---

## 💡 Dicas Adicionais para a Gravação:
1. **Compartilhamento de Tela:** Durante a parte da "Ação/Resultado", deixe o Dashboard em Streamlit rodando na tela.
2. **Mostre Documentos:** Ao citar a Rede Neural e a Governança, mencione os arquivos de decisões (Decision Logs) ou o Model Card.
3. **Naturalidade:** Entenda os "porquês" (Threshold 0.3, VIF/Colinearidade, PyTorch vs Sklearn) para falar de forma convicta, demonstrando total domínio do que foi construído.
