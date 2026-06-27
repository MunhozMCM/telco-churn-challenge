"""Smoke test — the pipeline loads, fits and predicts with the expected shape."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import FEATURES, ID_COLUMN, TARGET
from src.modeling.pipeline import build_pipeline


def test_pipeline_fit_predict(telco_sample):
    X = telco_sample[list(FEATURES) + [ID_COLUMN]]
    y = telco_sample[TARGET]

    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    pipe.fit(X, y)  # loads + trains without raising

    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]

    assert preds.shape == (len(telco_sample),)
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_ignores_unknown_category(trained_pipeline, telco_sample):
    # handle_unknown="ignore" must keep serving when an unseen category arrives.
    X = telco_sample[list(FEATURES) + [ID_COLUMN]].copy()
    X.loc[0, "Payment Method"] = "Crypto wallet"  # never seen in training
    proba = trained_pipeline.predict_proba(X)[:, 1]
    assert np.isfinite(proba).all()
