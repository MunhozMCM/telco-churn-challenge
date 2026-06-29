"""Train + evaluate Dummy, Logistic Regression and MLP, tracked with MLflow.

Runs the full reproducible flow: load → validate (pandera) → stratified split →
fit each model → log params/metrics/artifacts to MLflow. The Logistic Regression
pipeline is persisted as the production artifact (``models/model.joblib`` +
``meta.json``); the MLP is tracked in MLflow but not served (it ties LR on this
data — see notebooks/NN_MLP_experiments_decisions.md).
"""

from __future__ import annotations

from datetime import datetime, timezone

import mlflow
import numpy as np
import sklearn
from loguru import logger
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import (
    EXPERIMENT_NAME,
    FEATURES,
    ID_COLUMN,
    MLFLOW_TRACKING_URI,
    SEED,
    TARGET,
    TEST_SIZE,
    THRESHOLD,
)
from src.data.io import (
    dataframe_fingerprint,
    load_telco_churn,
    save_metadata,
    save_pipeline,
)
from src.data.schema import validate_input
from src.modeling.metrics import compute_metrics
try:
    from src.modeling.neural_net import predict_proba_mlp, train_mlp
    HAS_TORCH = True
except (ImportError, OSError) as e:
    logger.warning("PyTorch not found or failed to load ({}). MLP training will be skipped.", e)
    HAS_TORCH = False
    
    # Workaround: Se o PyTorch estiver quebrado no Windows (WinError 126), 
    # bibliotecas como skops/scipy/sklearn crashem ao tentar importar 'torch' implicitamente.
    # Criamos um mock vazio para blindar o ambiente.
    import sys
    from types import ModuleType
    mock_torch = ModuleType("torch")
    mock_torch.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = mock_torch
from src.modeling.pipeline import build_pipeline
from src.version import MODEL_VERSION


def _split(df):
    X = df[list(FEATURES) + [ID_COLUMN]]
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)


def _log_run(run_name, params, metrics, *, tags, sklearn_model=None):
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if sklearn_model is not None:
            mlflow.sklearn.log_model(
    sklearn_model,
    name="model",
    skops_trusted_types=["src.modeling.preprocessing.TelcoRawCleaner"]
)
    logger.info("logged MLflow run '{}' | metrics={}", run_name, metrics)


def train() -> dict:
    """Train all models, log to MLflow, persist the LR production artifact."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_telco_churn()
    validate_input(df)  # fail fast on contract violations
    fingerprint = dataframe_fingerprint(df)
    base_tags = {
        "data_fingerprint": fingerprint,
        "n_rows": str(len(df)),
        "model_version": MODEL_VERSION,
        "threshold": str(THRESHOLD),
    }
    logger.info("loaded {} rows | fingerprint={}", len(df), fingerprint[:12])

    X_train, X_test, y_train, y_test = _split(df)

    # --- Baseline: DummyClassifier ---
    dummy = build_pipeline(DummyClassifier(strategy="most_frequent"))
    dummy.fit(X_train, y_train)
    dummy_prob = dummy.predict_proba(X_test)[:, 1]
    dummy_pred = (dummy_prob >= THRESHOLD).astype(int)
    dummy_metrics = compute_metrics(y_test, dummy_pred, dummy_prob)
    _log_run(
        "dummy_classifier",
        {"model_type": "DummyClassifier", "strategy": "most_frequent"},
        dummy_metrics,
        tags=base_tags,
        sklearn_model=dummy,
    )

    # --- Logistic Regression (production model) ---
    lr = build_pipeline(LogisticRegression(max_iter=10000, random_state=SEED))
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= THRESHOLD).astype(int)
    lr_metrics = compute_metrics(y_test, lr_pred, lr_prob)
    _log_run(
        "logistic_regression",
        {"model_type": "LogisticRegression", "max_iter": 10000},
        lr_metrics,
        tags=base_tags,
        sklearn_model=lr,
    )

    # --- MLP (tracked, not served) ---
    if HAS_TORCH:
        prep = build_pipeline(LogisticRegression())  # reuse cleaner+preprocessor
        prep = prep[:-1]  # drop the estimator step → just preprocessing
        
        # Internal split for early stopping to avoid test set leakage
        X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
            X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
        )
        
        X_train_p = prep.fit_transform(X_train_inner)
        X_val_p = prep.transform(X_val_inner)
        X_test_p = prep.transform(X_test)
        
        mlp_model, mlp_params = train_mlp(
            X_train_p, y_train_inner.to_numpy(), X_val_p, y_val_inner.to_numpy()
        )
        mlp_prob = predict_proba_mlp(mlp_model, X_test_p)
        mlp_pred = (mlp_prob >= THRESHOLD).astype(int)
        mlp_metrics = compute_metrics(y_test, mlp_pred, mlp_prob)
        _log_run(
            "mlp_pytorch",
            {"model_type": "ChurnMLP", **mlp_params},
            mlp_metrics,
            tags=base_tags,
        )
    else:
        mlp_metrics = {}

    # --- Persist the LR pipeline as the production artifact ---
    save_pipeline(lr)
    meta = {
        "model_version": MODEL_VERSION,
        "model_type": "LogisticRegression",
        "threshold": THRESHOLD,
        "sklearn_version": sklearn.__version__,
        "data_fingerprint": fingerprint,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "features": list(FEATURES),
        "metrics": lr_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    save_metadata(meta)
    logger.info("saved production model.joblib + meta.json | LR metrics={}", lr_metrics)

    return {
        "dummy": dummy_metrics,
        "logistic_regression": lr_metrics,
        "mlp": mlp_metrics,
    }


if __name__ == "__main__":
    np.random.seed(SEED)
    results = train()
    for name, metrics in results.items():
        logger.info("{}: {}", name, metrics)
