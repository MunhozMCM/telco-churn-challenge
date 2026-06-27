"""Shared pytest fixtures — synthetic Telco data and a trained pipeline.

Synthetic data (never the real dataset) keeps tests fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import (
    CATEGORY_LEVELS,
    ID_COLUMN,
    NUMERIC_RANGES,
    TARGET,
)
from src.modeling.pipeline import build_pipeline


def _make_rows(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, list] = {ID_COLUMN: [f"CUST-{i:04d}" for i in range(n)]}
    for col, (low, high) in NUMERIC_RANGES.items():
        if isinstance(low, int) and isinstance(high, int):
            data[col] = rng.integers(low, high, size=n)
        else:
            data[col] = rng.uniform(low, high, size=n).round(2)
    for col, levels in CATEGORY_LEVELS.items():
        data[col] = rng.choice(list(levels), size=n)
    # Ensure both classes are present for estimators that need them.
    data[TARGET] = ([0, 1] * ((n // 2) + 1))[:n]
    return pd.DataFrame(data)


@pytest.fixture
def telco_sample() -> pd.DataFrame:
    """A small, valid raw-Telco frame (features + id + target)."""
    return _make_rows(40)


@pytest.fixture
def valid_payload() -> dict:
    """A valid /predict request body keyed by the real column names (aliases)."""
    return {
        "Zip Code": 90003,
        "Latitude": 33.96,
        "Longitude": -118.27,
        "Tenure Months": 12,
        "Monthly Charges": 70.35,
        "CLTV": 4000,
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "DSL",
        "Online Security": "Yes",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "Yes",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "One year",
        "Paperless Billing": "Yes",
        "Payment Method": "Mailed check",
    }


@pytest.fixture(scope="session")
def trained_pipeline():
    """A pipeline trained on synthetic data — reused across tests (session scope)."""
    df = _make_rows(60, seed=1)
    from src.config import FEATURES

    X = df[list(FEATURES) + [ID_COLUMN]]
    y = df[TARGET]
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=42))
    pipe.fit(X, y)
    return pipe
