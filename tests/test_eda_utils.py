import pandas as pd
import pytest

from src.eda_utils import (
    calculate_loss_ratio,
    calculate_margin,
    group_by_gender,
    group_by_province,
    group_by_vehicle_type,
    missing_value_summary,
)


@pytest.fixture
def sample_insurance_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Province": ["Gauteng", "Gauteng", "Western Cape", "Western Cape"],
            "VehicleType": ["SUV", "Sedan", "SUV", "Truck"],
            "Gender": ["Female", "Male", "Female", "Male"],
            "TotalPremium": [100.0, 200.0, 300.0, 400.0],
            "TotalClaims": [40.0, 100.0, 60.0, 200.0],
            "CustomValueEstimate": [120_000.0, None, 180_000.0, None],
        }
    )


def test_calculate_loss_ratio(sample_insurance_data: pd.DataFrame) -> None:
    loss_ratio = calculate_loss_ratio(sample_insurance_data)

    assert loss_ratio == pytest.approx(400.0 / 1000.0)


def test_calculate_margin(sample_insurance_data: pd.DataFrame) -> None:
    margin = calculate_margin(sample_insurance_data)

    assert margin == 600.0


def test_missing_value_summary_counts_missing_values(
    sample_insurance_data: pd.DataFrame,
) -> None:
    summary = missing_value_summary(sample_insurance_data)

    assert summary.loc["CustomValueEstimate", "missing_count"] == 2
    assert summary.loc["CustomValueEstimate", "missing_percentage"] == 50.0
    assert summary.loc["Province", "missing_count"] == 0


def test_group_by_province_summarizes_risk_metrics(
    sample_insurance_data: pd.DataFrame,
) -> None:
    grouped = group_by_province(sample_insurance_data)
    gauteng = grouped[grouped["Province"] == "Gauteng"].iloc[0]

    assert gauteng["policy_count"] == 2
    assert gauteng["total_premium"] == 300.0
    assert gauteng["total_claims"] == 140.0
    assert gauteng["loss_ratio"] == pytest.approx(140.0 / 300.0)
    assert gauteng["margin"] == 160.0


def test_group_by_vehicle_type_summarizes_risk_metrics(
    sample_insurance_data: pd.DataFrame,
) -> None:
    grouped = group_by_vehicle_type(sample_insurance_data)
    suv = grouped[grouped["VehicleType"] == "SUV"].iloc[0]

    assert suv["policy_count"] == 2
    assert suv["total_premium"] == 400.0
    assert suv["total_claims"] == 100.0
    assert suv["loss_ratio"] == pytest.approx(0.25)


def test_group_by_gender_summarizes_risk_metrics(
    sample_insurance_data: pd.DataFrame,
) -> None:
    grouped = group_by_gender(sample_insurance_data)
    female = grouped[grouped["Gender"] == "Female"].iloc[0]

    assert female["policy_count"] == 2
    assert female["total_premium"] == 400.0
    assert female["total_claims"] == 100.0
    assert female["margin"] == 300.0
