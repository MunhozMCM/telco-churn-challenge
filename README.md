\# Telco Churn Prediction End-to-End Pipeline



Este repositório contém a solução completa para o Tech Challenge de Machine Learning. O projeto abrange desde a Análise Exploratória de Dados (EDA) até o deploy de uma Rede Neural (PyTorch) servida via API REST (FastAPI).



\## Estrutura do Projeto

\* `data/`: Dataset original (Telco Customer Churn).

\* `docs/`: Model Card e documentação adicional.

\* `models/`: Pesos do modelo PyTorch (`.pth`) e artefatos do Scikit-Learn (`.pkl`).

\* `notebooks/`: Notebooks de EDA e treinamento documentado.

\* `src/`: Código fonte da aplicação FastAPI.

\* `mlruns/`: Rastreamento de experimentos do MLflow.



\## Setup e Execução



\### 1. Pré-requisitos

\* Python 3.10+

\* Git



\### 2. Instalação

Clone o repositório e crie um ambiente virtual:

```bash

git clone <SUA\_URL\_DO\_GITHUB\_AQUI>

cd telco-churn-challenge

python -m venv venv

\# Windows: venv\\Scripts\\activate | Mac/Linux: source venv/bin/activate

pip install -e .

