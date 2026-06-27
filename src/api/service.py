"""Pure ML service layer — no HTTP here.

Keeping prediction logic out of the route makes it testable without spinning up
the API and reusable by the batch job (Interface Segregation).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.api.schemas import CustomerData
from src.config import THRESHOLD


def to_features(data: CustomerData) -> pd.DataFrame:
    """Turn a validated request into a 1-row DataFrame with real column names.

    ``by_alias=True`` yields the dataset's column names ("Monthly Charges", …),
    which is the contract the sklearn pipeline selects on.
    """
    return pd.DataFrame([data.model_dump(by_alias=True)])


def predict_churn(
    model: Any, data: CustomerData, threshold: float = THRESHOLD
) -> dict[str, Any]:
    """Score one customer and apply the decision threshold."""
    probability = float(model.predict_proba(to_features(data))[:, 1][0])
    flag = int(probability >= threshold)
    return {
        "churn_probability": round(probability, 4),
        "churn_flag": flag,
        "risk_level": "High" if flag else "Low",
        "threshold": threshold,
    }
