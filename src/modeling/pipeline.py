"""Assemble the full inference pipeline: raw cleaning → preprocessing → model.

``build_pipeline`` injects the estimator (Dependency Inversion) so the same
preprocessing serves a DummyClassifier, LogisticRegression or any sklearn
estimator with no other change (Liskov substitution).
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.modeling.preprocessing import TelcoRawCleaner, build_preprocessor


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Return an unfitted pipeline ending in ``estimator``.

    Each call builds a fresh, untrained pipeline — avoids accidentally reusing a
    fitted object across experiments.
    """
    return Pipeline(
        [
            ("cleaner", TelcoRawCleaner()),
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )
