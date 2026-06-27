# Deployment Architecture — Batch vs. Real-Time

This document records the deployment decision required by Etapa 4 ("documentar
arquitetura de deploy escolhida — batch vs. real-time — + justificativa").

## Decision

**Primary deployment: daily BATCH scoring**, orchestrated by Apache Airflow.
A real-time **FastAPI** service is also provided for on-demand / ad-hoc lookups
and to satisfy the Etapa 3 API requirement, but it is not the primary path.

## Why batch

Churn is not a millisecond decision. Retention actions (offers, calls, emails)
are planned and executed on a daily or weekly cadence, so predictions computed
once per night are perfectly timely.

| Factor | Batch | Real-time API |
|---|---|---|
| Latency requirement | none (results consumed next day) | sub-200 ms |
| Throughput | whole customer base in one job | per-request |
| Cost | low (job runs, then nothing) | a server must stay up 24/7 |
| Operational complexity | a scheduled job + a table | load balancer, replicas, autoscaling |

Against the SLO that matters here — "fresh churn scores available each morning"
— batch wins on cost and simplicity with no downside. Real-time serving would add
always-on infrastructure for a latency guarantee the business does not need.

## How it works

```
data/raw (daily batch)  ──►  Airflow DAG (@daily)
                                  │
                                  ▼
                    src/batch/score.run_batch_scoring
                       │  load model.joblib (LR pipeline)
                       │  pandera-validate the batch (contract gate)
                       │  predict_proba → threshold (0.3)
                       ▼
                    data/predictions/predictions_<date>.csv
                       └─► consumed by retention / CRM workflows
```

- **DAG:** [`dags/churn_batch_scoring_dag.py`](../dags/churn_batch_scoring_dag.py)
  — `schedule="@daily"`, `catchup=False`, one `PythonOperator`.
- **Job logic:** [`src/batch/score.py`](../src/batch/score.py) — reuses the exact
  same model artifact, data contract and decision threshold as the API
  (`src/api/service.py`), so the two paths can never drift.

Airflow is intentionally **not** a project dependency (its pins conflict with the
torch/sklearn stack). The DAG file is import-guarded so the repo and test suite
load without Airflow; deploy it to an existing Airflow instance that has this
project installed (`pip install -e .`).

## The real-time API (secondary)

[`src/api/app.py`](../src/api/app.py) exposes `POST /predict` and `GET /health`
for single-customer, on-demand scoring (e.g. an agent checking one account). Same
model, same threshold, same contract — just a synchronous front door. Run it with
`make run` (or `churn-api`).

## When to revisit

Switch the primary path to real-time (online) serving only if a use case appears
that needs a churn score *the moment* a customer interacts — e.g. dynamic
in-session retention offers. At that point add replicas behind a load balancer and
promote the FastAPI service to primary; the batch job can remain for bulk scoring.
