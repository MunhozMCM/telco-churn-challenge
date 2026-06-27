"""Pydantic V2 contract for the inference API.

Guards a single request at the API edge (the per-record counterpart to the
pandera schema used for batches). Field names use aliases so the JSON payload
matches the dataset's real column names (e.g. "Monthly Charges").
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config import NUMERIC_RANGES

_zip = NUMERIC_RANGES["Zip Code"]
_lat = NUMERIC_RANGES["Latitude"]
_lon = NUMERIC_RANGES["Longitude"]
_tenure = NUMERIC_RANGES["Tenure Months"]
_monthly = NUMERIC_RANGES["Monthly Charges"]
_cltv = NUMERIC_RANGES["CLTV"]


class CustomerData(BaseModel):
    """One customer's features, validated against the data contract.

    ``populate_by_name`` lets clients send either the alias ("Monthly Charges")
    or the field name (monthly_charges); ``extra="forbid"`` rejects unknown keys.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # Numeric features
    zip_code: int = Field(..., ge=_zip[0], le=_zip[1], alias="Zip Code")
    latitude: float = Field(..., ge=_lat[0], le=_lat[1], alias="Latitude")
    longitude: float = Field(..., ge=_lon[0], le=_lon[1], alias="Longitude")
    tenure_months: int = Field(..., ge=_tenure[0], le=_tenure[1], alias="Tenure Months")
    monthly_charges: float = Field(
        ..., ge=_monthly[0], le=_monthly[1], alias="Monthly Charges"
    )
    cltv: int = Field(..., ge=_cltv[0], le=_cltv[1], alias="CLTV")

    # Categorical features (Literal enforces the allowed categories)
    gender: Literal["Female", "Male"] = Field(..., alias="Gender")
    senior_citizen: Literal["No", "Yes"] = Field(..., alias="Senior Citizen")
    partner: Literal["No", "Yes"] = Field(..., alias="Partner")
    dependents: Literal["No", "Yes"] = Field(..., alias="Dependents")
    phone_service: Literal["No", "Yes"] = Field(..., alias="Phone Service")
    multiple_lines: Literal["No", "No phone service", "Yes"] = Field(
        ..., alias="Multiple Lines"
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., alias="Internet Service"
    )
    online_security: Literal["No", "No internet service", "Yes"] = Field(
        ..., alias="Online Security"
    )
    online_backup: Literal["No", "No internet service", "Yes"] = Field(
        ..., alias="Online Backup"
    )
    device_protection: Literal["No", "No internet service", "Yes"] = Field(
        ..., alias="Device Protection"
    )
    tech_support: Literal["No", "No internet service", "Yes"] = Field(
        ..., alias="Tech Support"
    )
    streaming_tv: Literal["No", "No internet service", "Yes"] = Field(
        ..., alias="Streaming TV"
    )
    streaming_movies: Literal["No", "No internet service", "Yes"] = Field(
        ..., alias="Streaming Movies"
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., alias="Contract"
    )
    paperless_billing: Literal["No", "Yes"] = Field(..., alias="Paperless Billing")
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(..., alias="Payment Method")


class PredictionOut(BaseModel):
    """Inference response contract."""

    churn_probability: float = Field(..., description="P(churn) in [0, 1]")
    churn_flag: int = Field(..., description="1 if probability >= threshold else 0")
    risk_level: str = Field(..., description="'High' if churn_flag else 'Low'")
    threshold: float = Field(..., description="Decision threshold applied")
