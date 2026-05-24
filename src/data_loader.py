"""Data loading and inspection utilities for the ACIS insurance project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".xlsx", ".xls", ".parquet", ".json"}


def load_insurance_data(file_path: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    """Load an insurance dataset safely with pandas.

    The loader chooses the correct pandas reader from the file extension and
    raises clear errors for missing files, unsupported formats, or empty data.

    Parameters
    ----------
    file_path:
        Path to the insurance dataset.
    **read_kwargs:
        Additional keyword arguments passed to the selected pandas reader.

    Returns
    -------
    pandas.DataFrame
        The loaded insurance dataset.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{extension}'. Use: {supported}")

    try:
        if extension in {".csv", ".txt"}:
            df = pd.read_csv(path, **read_kwargs)
        elif extension in {".xlsx", ".xls"}:
            df = pd.read_excel(path, **read_kwargs)
        elif extension == ".parquet":
            df = pd.read_parquet(path, **read_kwargs)
        else:
            df = pd.read_json(path, **read_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset from {path}") from exc

    if df.empty:
        raise ValueError(f"Loaded dataset is empty: {path}")

    return df


def inspect_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Return data types, missing counts, and uniqueness by column."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_null_count": df.notna().sum().values,
            "missing_count": df.isna().sum().values,
            "missing_percentage": (df.isna().mean() * 100).round(2).values,
            "unique_count": df.nunique(dropna=True).values,
        }
    )


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a high-level summary of size, missingness, and duplicates."""
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percentage": round((missing_cells / total_cells) * 100, 2)
        if total_cells
        else 0.0,
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round((duplicate_rows / len(df)) * 100, 2)
        if len(df)
        else 0.0,
        "column_types": df.dtypes.astype(str).to_dict(),
    }


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted missing-value summary for every column."""
    summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percentage": (df.isna().mean() * 100).round(2),
        }
    )
    return summary.sort_values("missing_count", ascending=False)


def duplicate_row_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """Return duplicate row count and duplicate percentage."""
    duplicate_rows = int(df.duplicated().sum())

    return {
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": round((duplicate_rows / len(df)) * 100, 2)
        if len(df)
        else 0.0,
    }


def load_and_summarize(
    file_path: str | Path, **read_kwargs: Any
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a dataset and return the DataFrame with a dataset summary."""
    df = load_insurance_data(file_path, **read_kwargs)
    return df, summarize_dataset(df)
