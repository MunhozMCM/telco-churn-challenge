"""Daily batch scoring job — the project's primary deployment pattern.

Loads the production pipeline, validates an input batch against the data
contract, scores it, and writes ``CustomerID, churn_probability, churn_flag`` to
disk. Designed to be invoked by an Airflow DAG (see ``dags/``) or the CLI/Make
target. Shares the threshold and contract with the API — no duplicated ML logic.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import (
    FEATURES,
    ID_COLUMN,
    PREDICTIONS_DIR,
    THRESHOLD,
)
from src.data.io import load_pipeline
from src.data.schema import validate_input


def _read(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    return pd.read_csv(input_path)


def run_batch_scoring(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    threshold: float = THRESHOLD,
) -> Path:
    """Score a batch of customers and write predictions; return the output path."""
    input_path = Path(input_path)
    logger.info("batch scoring — reading {}", input_path)
    df = _read(input_path)

    validate_input(df)  # contract gate — fail fast on bad/drifted data

    model = load_pipeline()
    probabilities = model.predict_proba(df[list(FEATURES)])[:, 1]
    flags = (probabilities >= threshold).astype(int)

    ids = df[ID_COLUMN] if ID_COLUMN in df.columns else pd.RangeIndex(len(df))
    predictions = pd.DataFrame(
        {
            ID_COLUMN: ids,
            "churn_probability": probabilities.round(4),
            "churn_flag": flags,
        }
    )

    if output_path is None:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PREDICTIONS_DIR / f"predictions_{date.today().isoformat()}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    logger.info(
        "scored {} customers ({} flagged churn) → {}",
        len(predictions),
        int(flags.sum()),
        output_path,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Telco churn batch scoring")
    parser.add_argument("input", help="Path to the input batch (csv/parquet/xlsx)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: data/predictions/)",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=THRESHOLD, help="Decision threshold"
    )
    args = parser.parse_args()
    run_batch_scoring(args.input, args.output, threshold=args.threshold)


if __name__ == "__main__":
    main()
