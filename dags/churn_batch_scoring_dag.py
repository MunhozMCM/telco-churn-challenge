"""Airflow DAG — daily Telco churn batch scoring.

The chosen production deployment is **batch** (churn is acted on daily, so high
throughput at low cost beats real-time latency — see
``docs/deployment_architecture.md``). This DAG runs the model over the day's
customer batch and writes predictions for downstream retention workflows.

Airflow is intentionally NOT a project dependency (heavy, conflicting pins). The
imports below are guarded so the repo — and the test suite — import cleanly
without Airflow installed; deploy this file to an existing Airflow instance whose
environment also has this project installed.

Setup: copy/symlink this file into your Airflow ``dags/`` folder, ensure the
project is importable (``pip install -e .``), and configure the connection /
input path via the ``CHURN_BATCH_INPUT`` Airflow Variable or env var.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

# Default daily input location (override via Airflow Variable / env var).
DEFAULT_INPUT = os.environ.get(
    "CHURN_BATCH_INPUT", "data/raw/Telco_customer_churn.xlsx"
)


def _score(**_context) -> str:
    """Task callable — runs the shared batch scoring function."""
    from src.batch.score import run_batch_scoring

    output = run_batch_scoring(DEFAULT_INPUT)
    return str(output)


try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    default_args = {
        "owner": "data-science",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="churn_batch_scoring",
        description="Daily Telco churn scoring (Logistic Regression pipeline)",
        schedule="@daily",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        default_args=default_args,
        tags=["churn", "batch", "inference"],
    ) as dag:
        PythonOperator(
            task_id="score_customers",
            python_callable=_score,
        )
except ImportError:
    # Airflow not installed (e.g. local dev / CI) — the module still imports so
    # tests can verify it parses and the task callable can be exercised directly.
    dag = None
