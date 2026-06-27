"""Preprocessing: a custom raw-cleaning transformer + the column preprocessor.

The notebook did its cleaning imperatively (``pd.to_numeric``, ``str.strip``,
``get_dummies``). Here that logic is encapsulated so it travels with the model
and is applied identically in training, the API and the batch job — the
train/serve consistency the textbook stresses.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class TelcoRawCleaner(BaseEstimator, TransformerMixin):
    """Defensive cleaner for raw Telco rows (stateless → ``fit`` returns self).

    Coerces the numeric feature columns to numbers (a CSV/Excel export can ship
    them as strings) and strips whitespace from the categorical columns, so an
    upstream format change becomes a noisy failure instead of a silent NaN.
    """

    def __init__(
        self,
        numeric_features: tuple[str, ...] | None = None,
        categorical_features: tuple[str, ...] | None = None,
    ):
        # __init__ only stores params as received (BaseEstimator contract).
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        return self  # nothing to learn

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = self.numeric_features or NUMERIC_FEATURES
        categorical = self.categorical_features or CATEGORICAL_FEATURES

        X = X.copy()  # never mutate the caller's frame
        for col in numeric:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        for col in categorical:
            if col in X.columns:
                X[col] = X[col].astype(str).str.strip()
        return X


def build_preprocessor() -> ColumnTransformer:
    """Scale numerics and one-hot encode categoricals, selecting by name.

    ``handle_unknown="ignore"`` keeps the encoder serve-safe: an unseen category
    becomes all-zeros instead of crashing. (We skip ``drop`` because the
    regularised models tolerate the mild collinearity, and dropping is
    incompatible with ``handle_unknown="ignore"``.)
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
