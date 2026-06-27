"""Schema-validation tests — the pandera data contract gate."""

from __future__ import annotations

import pandera.errors as pa_errors
import pytest

from src.data.schema import validate_input


def test_accepts_valid_data(telco_sample):
    # Should not raise.
    validate_input(telco_sample)


def test_rejects_negative_tenure(telco_sample):
    bad = telco_sample.copy()
    bad.loc[0, "Tenure Months"] = -5
    with pytest.raises((pa_errors.SchemaError, pa_errors.SchemaErrors)):
        validate_input(bad)


def test_rejects_unknown_category(telco_sample):
    bad = telco_sample.copy()
    bad.loc[0, "Contract"] = "Lifetime"  # not an allowed level
    with pytest.raises((pa_errors.SchemaError, pa_errors.SchemaErrors)):
        validate_input(bad)


def test_rejects_missing_required_column(telco_sample):
    bad = telco_sample.drop(columns=["Monthly Charges"])
    with pytest.raises((pa_errors.SchemaError, pa_errors.SchemaErrors)):
        validate_input(bad)
