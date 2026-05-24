"""Prepare raw ACIS insurance data for analysis and modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

from src.data_loader import load_insurance_data


def clean_insurance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw insurance dataset.

    The cleaning step is intentionally conservative for auditability. It removes
    exact duplicate rows, trims whitespace from text columns, and standardizes
    empty strings to missing values without imputing or dropping partial records.
    """
    cleaned = df.copy()

    text_columns = [
        column
        for column in cleaned.columns
        if is_object_dtype(cleaned[column]) or is_string_dtype(cleaned[column])
    ]

    for column in text_columns:
        cleaned[column] = cleaned[column].str.strip()
        cleaned[column] = cleaned[column].replace("", pd.NA)

    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def save_cleaned_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save cleaned data as a CSV file, creating parent folders if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


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
    return parser.parse_args()


def main() -> None:
    """Load raw data, apply conservative cleaning, and save the result."""
    args = parse_args()
    raw_data = load_insurance_data(args.input)
    cleaned_data = clean_insurance_data(raw_data)
    save_cleaned_data(cleaned_data, args.output)


if __name__ == "__main__":
    main()
