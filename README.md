<div align="center">
  
  # Telco Churn Prediction Pipeline
  
  **Pipeline de ML end-to-end para previsão de churn — baselines + Rede Neural (PyTorch), rastreados com MLflow e servidos via batch (Airflow) e API REST (FastAPI).**
  
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/MLflow-0194E2.svg?style=for-the-badge&logo=MLflow&logoColor=white" alt="MLflow" />
  </p>

  <sub>Versão do modelo: <code>1.0.0</code> · ver <a href="src/version.py"><code>src/version.py</code></a></sub>
</div>

---

## O Business Case

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. Este projeto entrega um **modelo preditivo end-to-end** que classifica clientes com risco iminente de cancelamento (Churn), permitindo ações de retenção proativas e direcionadas.

> **Foco analítico:** o threshold de decisão foi reduzido para **0.3** para priorizar o *Recall*, mitigando **Falsos Negativos** (deixar um cliente cancelar sem intervir) — o maior custo financeiro da operação. Justificativa em [`notebooks/ML_experiments_decisions.md`](notebooks/ML_experiments_decisions.md).

O modelo **servido em produção é a Regressão Logística** (interpretável, leve e empatada com o MLP nas métricas — ver [`notebooks/NN_MLP_experiments_decisions.md`](notebooks/NN_MLP_experiments_decisions.md)). O MLP é treinado e rastreado no MLflow para comparação.

---

## Stack & Arquitetura

| Componente | Tecnologia | Propósito |
| :--- | :--- | :--- |
| **Pipeline reproduzível** | `scikit-learn` | `ColumnTransformer` (OneHotEncoder + StandardScaler) + transformer custom, num único artefato sem *data leakage*. |
| **Rede Neural** | `PyTorch` | MLP (64→32) com *early stopping*, comparada aos baselines. |
| **Tracking** | `MLflow` | Parâmetros, métricas e artefatos dos 3 modelos (backend sqlite). |
| **Validação de dados** | `pandera` | Contrato do dataset (batch/treino). |
| **API de inferência** | `FastAPI` + `Pydantic` | `/predict` e `/health`, validação de contrato, logging estruturado, middleware de latência. |
| **Deploy batch** | `Airflow` | Job diário que escora a base inteira (arquitetura primária). |
| **Tooling** | `ruff`, `taskipy`, `Makefile`, `pytest` | Lint, formatação, atalhos e testes. |

Decisão de deploy (batch × real-time) documentada em [`docs/deployment_architecture.md`](docs/deployment_architecture.md).

---

## Estrutura do Projeto

```text
telco-churn-challenge/
├── data/raw/              # Dataset IBM + metadata.md (imutável)
├── dags/                  # Airflow DAG do scoring diário
├── docs/                  # Model Card + arquitetura de deploy
├── notebooks/             # EDA, baselines (ML) e Rede Neural (NN) — logam no MLflow
├── src/
│   ├── config.py          # Constantes, features, threshold, MLflow
│   ├── version.py         # MODEL_VERSION (semver)
│   ├── data/              # io.py (load/save) · schema.py (pandera)
│   ├── modeling/          # preprocessing · pipeline · neural_net · metrics · train
│   ├── api/               # app · schemas · service · middleware · entry
│   └── batch/             # score.py (job de scoring em lote)
├── tests/                 # unit · schema · smoke · api
├── Makefile · pyproject.toml
```

## Tech Challenge 1 - Mateus Munhoz. RA: RM375436 | Lucas Munhoz. RA: RM374691 | Gabriel Figueira (RM374505)

---

## Como Executar o Projeto (Passo a Passo)

Preparamos o ambiente para ser executado de forma simples, com duas frentes principais: A **API de Previsão (Backend)** e o **Dashboard Interativo (Frontend)**.

### Pré-requisitos
O ambiente vive em `notebooks/.venv`. Requer Python 3.10+.
```bash
make install          # ou: notebooks/.venv/bin/pip install -e ".[dev]"
```

### Passo 1: Iniciando a API (O Cérebro do Modelo)
Abra um terminal na pasta raiz do projeto (`telco-churn-challenge`) e execute:
```bash
make run
```
*(Se preferir rodar manualmente: `.venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000`)*

- **Como testar a API diretamente:** 
  Acesse: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). 

### Passo 2: Iniciando o Dashboard (A Interface Visual)
Deixe a API rodando no primeiro terminal. Abra **um novo terminal** na pasta raiz e execute:
```bash
.venv/bin/streamlit run src/app.py
```
Acesse [http://localhost:8501](http://localhost:8501) no seu navegador. O Dashboard enviará os dados para a API e exibirá o risco de cancelamento e a latência na tela!

---

##  Troubleshooting (Solução de Problemas)

### Erro: `{"detail": "Model unavailable"}` no Dashboard ou `/predict`
Esse erro ocorre porque a API iniciou em **modo degradado**, ou seja, ela ligou mas não encontrou o arquivo físico do modelo pré-treinado (`models/model.joblib`). Isso é comum ao clonar o repositório pela primeira vez (já que os modelos não sobem para o Git).

**Como resolver:**
Basta treinar os modelos localmente para gerar os artefatos. No terminal, execute:
```bash
make train
# Ou manualmente: .venv/bin/python -m src.modeling.train
```
Após o script concluir (ele treinará os baselines e o MLP e salvará o `.joblib`), **reinicie a API** (ou o container) e ela já voltará a responder com as previsões perfeitamente.

---

## Comandos Úteis (Para Desenvolvedores)

| Comando | O que faz |
|---|---|
| `make train` | Treina Dummy + LogisticRegression + MLP, loga tudo no MLflow e salva o pipeline. |
| `make run` | Sobe a API (uvicorn) em `localhost:8000`. |
| `make score` | Roda o scoring em lote sobre o dataset e grava em `data/predictions/`. |
| `make test` | Roda a suíte de testes (pytest). |
| `make lint` / `make format` | Verifica / corrige lint e formatação (ruff). |

## MLflow
Os treinamentos gravam no backend sqlite (`sqlite:///mlflow.db`). Para inspecionar:
```bash
notebooks/.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

