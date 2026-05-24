import pandas as pd

from src.data_loader import duplicate_row_summary, summarize_dataset
from src.eda_utils import calculate_loss_ratio, group_by_province


def test_dataset_summary_counts_rows_columns_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "Province": ["A", "A", "B"],
            "TotalPremium": [100.0, 100.0, 200.0],
            "TotalClaims": [40.0, 40.0, 80.0],
        }
    )

    summary = summarize_dataset(df)
    duplicates = duplicate_row_summary(df)

    assert summary["rows"] == 3
    assert summary["columns"] == 3
    assert duplicates["duplicate_rows"] == 1


def test_risk_helpers_calculate_loss_ratio_and_group_summary() -> None:
    df = pd.DataFrame(
        {
            "Province": ["A", "A", "B"],
            "TotalPremium": [100.0, 300.0, 200.0],
            "TotalClaims": [50.0, 100.0, 20.0],
        }
    )

    assert calculate_loss_ratio(df) == 170.0 / 600.0

    grouped = group_by_province(df)

    assert set(grouped["Province"]) == {"A", "B"}
    assert "loss_ratio" in grouped.columns
    assert "margin" in grouped.columns
