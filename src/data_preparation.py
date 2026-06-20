"""Prepare raw ACIS insurance data for analysis and modeling."""

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
    """Configure console logging for command-line pipeline runs."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_text_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that should be treated as text-like fields."""
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


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and reset the index."""
    return df.drop_duplicates().reset_index(drop=True)


def build_cleaning_report(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact report describing the cleaning changes."""
    return {
        "raw_rows": int(len(raw_df)),
        "cleaned_rows": int(len(cleaned_df)),
        "rows_removed": int(len(raw_df) - len(cleaned_df)),
        "raw_columns": int(raw_df.shape[1]),
        "cleaned_columns": int(cleaned_df.shape[1]),
        "missing_values": int(cleaned_df.isna().sum().sum()),
    }


def clean_insurance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw ACIS insurance dataset conservatively.

    The cleaning step is intentionally conservative for auditability. It removes
    exact duplicate rows, trims whitespace from text columns, and standardizes
    empty strings to missing values without imputing or dropping partial records.
    """
    LOGGER.info("Standardizing text fields")
    cleaned = standardize_text_values(df)

    LOGGER.info("Removing exact duplicate rows")
    cleaned = remove_duplicate_rows(cleaned)

    return cleaned


def save_cleaned_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save cleaned data as a CSV file, creating parent folders if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_cleaning_pipeline(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Run the full raw-to-cleaned data pipeline and return a cleaning report."""
    LOGGER.info("Loading raw insurance data from %s", input_path)
    raw_data = load_insurance_data(input_path)
    LOGGER.info("Loaded raw data with %s rows and %s columns", *raw_data.shape)

    cleaned_data = clean_insurance_data(raw_data)
    report = build_cleaning_report(raw_data, cleaned_data)

    LOGGER.info("Saving cleaned insurance data to %s", output_path)
    save_cleaned_data(cleaned_data, output_path)

    LOGGER.info(
        "Cleaning complete: %s rows removed; %s rows saved",
        report["rows_removed"],
        report["cleaned_rows"],
    )

    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the data preparation step."""
    parser = argparse.ArgumentParser(description="Clean ACIS insurance data.")
    parser.add_argument(
        "--input",
        default="data/raw/insurance_data.csv",
        help="Path to the raw insurance dataset.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/insurance_data_cleaned.csv",
        help="Path where the cleaned dataset should be written.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for pipeline messages.",
    )
    return parser.parse_args()


def main() -> None:
    """Load raw data, apply conservative cleaning, and save the result."""
    args = parse_args()
    configure_logging(args.log_level)
    report = run_cleaning_pipeline(args.input, args.output)
    LOGGER.info("Cleaning report: %s", report)


if __name__ == "__main__":
    main()
