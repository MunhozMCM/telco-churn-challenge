<div align="center">
  
  # 📉 Telco Churn Prediction Pipeline
  
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

---

## Setup

Requer Python 3.10+. O ambiente vive em `notebooks/.venv`.

```bash
make install          # ou: notebooks/.venv/bin/pip install -e ".[dev]"
```

## Comandos

| Comando | O que faz |
|---|---|
| `make train` | Treina Dummy + LogisticRegression + MLP, loga tudo no MLflow e salva `models/model.joblib` + `meta.json`. |
| `make run` | Sobe a API (uvicorn) em `localhost:8000` (`/docs` para o Swagger). |
| `make score` | Roda o scoring em lote sobre o dataset e grava em `data/predictions/`. |
| `make test` | Roda a suíte de testes (pytest). |
| `make lint` / `make format` | Verifica / corrige lint e formatação (ruff). |

> Sem `make`? Os mesmos atalhos existem via taskipy: `task train`, `task run`, `task test`… (com o venv ativado). E sempre dá para chamar direto: `notebooks/.venv/bin/python -m src.modeling.train`.

## Exemplo — API

```bash
make run
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "Zip Code": 90003, "Latitude": 33.96, "Longitude": -118.27,
  "Tenure Months": 2, "Monthly Charges": 53.85, "CLTV": 3239,
  "Gender": "Male", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
  "Phone Service": "Yes", "Multiple Lines": "No", "Internet Service": "DSL",
  "Online Security": "No", "Online Backup": "Yes", "Device Protection": "No",
  "Tech Support": "No", "Streaming TV": "No", "Streaming Movies": "No",
  "Contract": "Month-to-month", "Paperless Billing": "Yes", "Payment Method": "Mailed check"
}'
# → {"churn_probability": 0.4378, "churn_flag": 1, "risk_level": "High", "threshold": 0.3}
```

## MLflow

`make train` e a execução dos notebooks gravam no mesmo backend sqlite
(`sqlite:///mlflow.db`, experimento `telco-churn`). Para inspecionar:

```bash
notebooks/.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Runs dos notebooks são marcados com a tag `source` para distingui-los dos runs de `src/modeling/train.py`.

---

## Tech Challenge 1 — Mateus Munhoz (RM375436) · Lucas Munhoz (RM374691)
