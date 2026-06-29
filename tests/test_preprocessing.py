"""Unit tests for the custom TelcoRawCleaner transformer."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone

from src.modeling.preprocessing import TelcoRawCleaner


def test_coerces_numeric_strings_to_numbers():
    df = pd.DataFrame(
        {"Latitude": ["34.0", "35.0"], "Monthly Charges": ["50.5", "60"]}
    )
    out = TelcoRawCleaner().transform(df)
    assert pd.api.types.is_numeric_dtype(out["Latitude"])
    assert out["Monthly Charges"].tolist() == [50.5, 60.0]


def test_strips_categorical_whitespace():
    df = pd.DataFrame(
        {"Gender": [" Male ", "Female"], "Contract": ["Two year ", " One year"]}
    )
    out = TelcoRawCleaner().transform(df)
    assert out["Gender"].tolist() == ["Male", "Female"]
    assert out["Contract"].tolist() == ["Two year", "One year"]


def test_does_not_mutate_input():
    df = pd.DataFrame({"Latitude": ["34.0"]})
    original = df.copy()
    TelcoRawCleaner().transform(df)
    pd.testing.assert_frame_equal(df, original)


def test_fit_returns_self():
    cleaner = TelcoRawCleaner()
    assert cleaner.fit(pd.DataFrame({"Gender": ["Male"]})) is cleaner


def test_clone_preserves_params():
    # GridSearchCV / Pipeline rely on clone(): get_params/set_params must round-trip.
    cleaner = TelcoRawCleaner()
    assert cleaner.get_params() == clone(cleaner).get_params()
