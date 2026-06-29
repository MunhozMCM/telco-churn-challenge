# Playbook de Monitoramento e Manutenção

Este playbook descreve o procedimento padrão de MLOps para monitorar e manter a saúde do modelo de previsão de Churn em produção, com base nas melhores práticas do mercado.

## 1. O que vamos monitorar?

O monitoramento do modelo não se resume apenas a métricas de software. Acompanhamos ativamente:

* **Métricas de Qualidade Preditiva (Model Drift):**
  * **Acurácia e F1-Score:** Assim que o *ground truth* (retenção ou cancelamento real) se torna disponível no fim do ciclo de faturamento, comparamos com a predição da API. Uma queda de performance indica obsolescência do modelo.

* **Métricas de Data Drift:**
  * **Distribuição das Entradas:** Comparamos estatisticamente (ex: testes Kolmogorov-Smirnov via ferramentas como *Evidently*) se o perfil das chamadas para a API (idade do cliente, uso de fibra ótica) desvia drasticamente do dataset de treinamento.
  * **Distribuição das Predições (Prediction Drift):** Avaliamos se a proporção geral de predições de "Alto Risco" (High Risk) subiu ou desceu de forma brusca e inexplicável.

* **Métricas de Serviço (Observabilidade):**
  * **Latência (Tempo de Resposta):** Acompanhamento do P50, P95 e P99 das requisições via API. A latência é salva pelo middleware no header `X-Process-Time`.
  * **Throughput (RPS):** Quantas requisições o modelo suporta por segundo.
  * **Erros (HTTP 500):** Bugs de software na camada da FastAPI.

* **Métricas de Negócio:**
  * **Retenção:** A recomendação ou ação de negócio baseada na predição de risco está de fato mantendo os clientes na operadora?

---

## 2. Ferramentas e Infraestrutura

Para viabilizar este playbook, o projeto utiliza as seguintes tecnologias:

1. **Logging em JSON Estruturado (Backend):**
   * A aplicação `api.py` salva cada predição e input no arquivo `models/inference_logs.json`. Este log histórico serve como base de dados primária para análise de drift e depuração.
2. **Prometheus & Grafana (Visualização e Alertas):**
   * Usados em conjunto, o Prometheus raspa métricas expostas pela infraestrutura (latência e throughput) e o Grafana as exibe, enviando alertas (Slack/Email) quando *thresholds* são violados.
3. **MLFlow Tracking e Model Registry:**
   * **Tracking:** Usado para logar os experimentos e *runs* diárias/semanais comparando a performance em novos lotes de dados.
   * **Registry:** Gerencia os estágios das versões (ex: Staging vs Production). Essencial para *Rollbacks* seguros.

---

## 3. Estratégias de Retreino e Intervenção

O modelo passará por ciclos de retreino sob as seguintes condições:

* **Gatilho por degradação (Performance/Drift):** 
  * Acionado automaticamente ou manualmente caso o F1-Score médio de produção caia abaixo de 80% ou o *Evidently* dispare um alerta crítico de Data Drift.
* **Agendamento Periódico:**
  * Preventivamente, novos lotes de dados de faturamento (novas safras de clientes) acionarão um job de retreino (ex: mensal) para incorporar os padrões mais recentes.
* **Rollback de Emergência:**
  * Caso uma nova versão implantada apresente anomalias (muitos erros na API ou viés severo) e não haja tempo para retreinar, revertemos no *MLFlow Registry* a versão `Production` para a versão `Archived` anterior (Shadow Deployment).

> **Lembre-se (Lição Prática):** Não existe modelo *fire-and-forget*. Ferramentas são apenas facilitadores para um processo governado de evolução contínua da Inteligência Artificial.
