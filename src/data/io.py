"""Input/output helpers for the IBM Telco Customer Churn dataset.

Everything that touches the "outside world" — reading the dataset, serialising
and reloading the trained pipeline, writing model metadata — is isolated here
(single responsibility). If the data source changes, only this module changes.
"""

import hashlib
import json
from typing import Any

import joblib
import pandas as pd

from src.config import (
    DEFAULT_METADATA_NAME,
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


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Return a stable SHA-256 fingerprint of a DataFrame's contents.

    Lets us record exactly which data slice a model was trained on (auditability).
    """
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return digest.hexdigest()


def save_pipeline(pipeline: Any, name: str = DEFAULT_PIPELINE_NAME) -> None:
    """Serialize a trained pipeline in the project's models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / name)


def load_pipeline(name: str = DEFAULT_PIPELINE_NAME) -> Any:
    """Load a trusted serialized pipeline from the models directory.

    Security note: ``joblib.load`` executes code embedded in the artifact — only
    load files this project produced.
    """
    return joblib.load(MODELS_DIR / name)


def save_metadata(meta: dict[str, Any], name: str = DEFAULT_METADATA_NAME) -> None:
    """Persist model metadata (version, lib versions, data hash, metrics)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / name, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def load_metadata(name: str = DEFAULT_METADATA_NAME) -> dict[str, Any]:
    """Load model metadata written alongside the pipeline artifact."""
    with open(MODELS_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)
