"""Prepare ACIS raw insurance data for reproducible analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from src.data_loader import load_insurance_data


LOGGER = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure readable console logging for DVC pipeline runs."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_text_columns(df: pd.DataFrame) -> list[str]:
    """Return text-like columns that can safely be stripped."""
    return [
        column
        for column in df.columns
        if is_object_dtype(df[column]) or is_string_dtype(df[column])
    ]


def standardize_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and convert blank strings to missing values."""
    cleaned = df.copy()

    for column in get_text_columns(cleaned):
        cleaned[column] = cleaned[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        cleaned[column] = cleaned[column].replace("", pd.NA)

    return cleaned


def clean_insurance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative, auditable cleaning to the raw insurance data."""
    LOGGER.info("Standardizing text values")
    cleaned = standardize_text_values(df)

    LOGGER.info("Removing exact duplicate rows")
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


def build_cleaning_report(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact summary of the cleaning step."""
    return {
        "raw_rows": int(len(raw_df)),
        "cleaned_rows": int(len(cleaned_df)),
        "rows_removed": int(len(raw_df) - len(cleaned_df)),
        "columns": int(cleaned_df.shape[1]),
        "missing_values": int(cleaned_df.isna().sum().sum()),
    }


def save_cleaned_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save cleaned data to CSV and create the output folder if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_cleaning_pipeline(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Load raw data, clean it, save the result, and return a report."""
    LOGGER.info("Loading raw data from %s", input_path)
    raw_df = load_insurance_data(input_path)

    cleaned_df = clean_insurance_data(raw_df)

    LOGGER.info("Saving cleaned data to %s", output_path)
    save_cleaned_data(cleaned_df, output_path)

    report = build_cleaning_report(raw_df, cleaned_df)
    LOGGER.info("Cleaning report: %s", report)
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the cleaning pipeline."""
    parser = argparse.ArgumentParser(description="Clean ACIS insurance data.")
    parser.add_argument("--input", default="data/raw/insurance_data.csv")
    parser.add_argument("--output", default="data/processed/insurance_data_cleaned.csv")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line cleaning pipeline."""
    args = parse_args()
    configure_logging(args.log_level)
    run_cleaning_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
