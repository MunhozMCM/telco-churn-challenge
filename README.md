<div align="center">
  
  # 📉 Telco Churn Prediction Pipeline
  
  **Rede Neural Multi-Layer Perceptron (MLP) com deploy em produção via API REST.**
  
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=FastAPI&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
    <img src="https://img.shields.io/badge/MLflow-0194E2.svg?style=for-the-badge&logo=MLflow&logoColor=white" alt="MLflow" />
  </p>
</div>

---

## O Business Case

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. Este projeto entrega um **modelo preditivo end-to-end** que classifica clientes com risco iminente de cancelamento (Churn), permitindo ações de retenção proativas e direcionadas.

> **Foco Analítico:** A arquitetura e as métricas foram otimizadas para priorizar o *F1-Score*, mitigando **Falsos Negativos** (deixar um cliente cancelar sem intervir), o que representa o maior custo financeiro para a operação.

---

## Stack Tecnológico & Arquitetura

O projeto foi construído seguindo as melhores práticas de Engenharia de Software aplicadas a Machine Learning (MLOps).

| Componente | Tecnologia | Propósito |
| :--- | :--- | :--- |
| **Deep Learning** | `PyTorch` | Construção do MLP customizado com *Early Stopping*. |
| **Baselines & Pipelines** | `Scikit-Learn` | Modelos lineares de comparação e padronização de features (`StandardScaler`). |
| **Tracking & Registro** | `MLflow` | Versionamento de parâmetros, métricas e modelos. |
| **Inference API** | `FastAPI` | Deploy do modelo com endpoints assíncronos e validação de schema. |
| **Data Validation** | `Pydantic` | Tipagem estática rigorosa para os payloads da API. |

---

## Estrutura do Projeto

```text
telco-churn-challenge/
├── data/               # Dataset original da IBM (ignorado pelo git)
├── docs/               # Model Card documentando vieses e limitações
├── mlruns/             # Histórico de rastreamento local (MLflow)
├── models/             # Artefatos serilizados (.pth e .pkl)
├── notebooks/          # Exploração (EDA) e experimentação da Rede Neural
├── src/
│   └── api.py          # Código-fonte da aplicação FastAPI
├── .gitignore          # Regras de exclusão do repositório
├── pyproject.toml      # Configuração de dependências e regras do linter (Ruff)
└── README.md           # Documentação central

## Tech Challenge 1 - Mateus Munhoz. RA: RM375436 | Lucas Munhoz. RA: RM374691 |
