"""Input/output helpers for the IBM Telco Customer Churn dataset."""

from typing import Any

import joblib
import pandas as pd

from src.config import (
    DEFAULT_PIPELINE_NAME,
    MODELS_DIR,
    TELCO_CHURN_DATA_PATH,
    TELCO_CHURN_SHEET_NAME,
)


def load_telco_churn() -> pd.DataFrame:
    """Load the IBM Telco Customer Churn dataset from its Excel workbook."""
    telco_churn_df = pd.read_excel(
        TELCO_CHURN_DATA_PATH,
        sheet_name=TELCO_CHURN_SHEET_NAME,
    )
    return telco_churn_df


def save_pipeline(pipeline: Any, name: str = DEFAULT_PIPELINE_NAME) -> None:
    """Serialize a trained pipeline in the project's models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / name)


def load_pipeline(name: str = DEFAULT_PIPELINE_NAME) -> Any:
    """Load a trusted serialized pipeline from the models directory."""
    return joblib.load(MODELS_DIR / name)
