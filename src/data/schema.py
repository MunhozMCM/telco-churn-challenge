"""pandera data contract for the Telco scoring input.

This is the gate for the data pipeline (training + batch): it validates that the
raw feature columns exist, have the right type, and fall in plausible ranges /
allowed categories before anything touches the model. It is the tabular
counterpart to the Pydantic model that guards the API edge.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from src.config import CATEGORY_LEVELS, NUMERIC_RANGES


def _numeric_column(col: str) -> Column:
    low, high = NUMERIC_RANGES[col]
    return Column(
        float,
        checks=Check.in_range(low, high),
        nullable=False,
        coerce=True,
        required=True,
    )


def _categorical_column(col: str) -> Column:
    return Column(
        str,
        checks=Check.isin(list(CATEGORY_LEVELS[col])),
        nullable=False,
        coerce=True,
        required=True,
    )


# Built from config so the contract has a single source of truth.
TELCO_INPUT_SCHEMA: DataFrameSchema = DataFrameSchema(
    {
        **{col: _numeric_column(col) for col in NUMERIC_RANGES},
        **{col: _categorical_column(col) for col in CATEGORY_LEVELS},
    },
    strict=False,  # extra columns (CustomerID, target) are allowed through
    coerce=True,
)


def validate_input(df: pa.typing.pandas.DataFrame, *, lazy: bool = True):
    """Validate a scoring/training frame against the input contract.

    Raises ``pandera.errors.SchemaError(s)`` if the data violates the contract.
    Returns the (possibly coerced) DataFrame on success.
    """
    return TELCO_INPUT_SCHEMA.validate(df, lazy=lazy)
