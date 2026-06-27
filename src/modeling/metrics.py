"""Shared classification metrics for churn models.

Centralises the helper previously duplicated in ML_experiments.ipynb and
NN_MLP_experiments.ipynb so every model is scored identically.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true,
    y_pred,
    y_prob=None,
) -> dict[str, float]:
    """Return the project's standard metric set as a plain dict.

    Includes the technical metrics required by the challenge (AUC-ROC, PR-AUC,
    F1) plus precision/recall/accuracy. AUC metrics need probabilities; when
    ``y_prob`` is omitted they are skipped.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return metrics
