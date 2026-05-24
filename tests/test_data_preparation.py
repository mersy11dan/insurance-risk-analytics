import pandas as pd

from src.data_preparation import (
    build_cleaning_report,
    clean_insurance_data,
    standardize_text_values,
)


def test_standardize_text_values_trims_and_converts_blanks() -> None:
    df = pd.DataFrame(
        {
            "Province": [" Gauteng ", ""],
            "TotalPremium": [100.0, 200.0],
        }
    )

    cleaned = standardize_text_values(df)

    assert cleaned.loc[0, "Province"] == "Gauteng"
    assert pd.isna(cleaned.loc[1, "Province"])


def test_clean_insurance_data_removes_exact_duplicates() -> None:
    df = pd.DataFrame(
        {
            "Province": ["Gauteng", "Gauteng", "Western Cape"],
            "TotalPremium": [100.0, 100.0, 200.0],
            "TotalClaims": [40.0, 40.0, 80.0],
        }
    )

    cleaned = clean_insurance_data(df)

    assert len(cleaned) == 2
    assert cleaned.index.tolist() == [0, 1]


def test_build_cleaning_report_summarizes_changes() -> None:
    raw_df = pd.DataFrame({"Province": ["A", "A", None]})
    cleaned_df = pd.DataFrame({"Province": ["A", None]})

    report = build_cleaning_report(raw_df, cleaned_df)

    assert report["raw_rows"] == 3
    assert report["cleaned_rows"] == 2
    assert report["rows_removed"] == 1
    assert report["missing_values"] == 1
