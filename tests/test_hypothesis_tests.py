import pandas as pd

from src.hypothesis_tests import (
    add_margin_and_claim_flag,
    build_hypothesis_summary,
    chi_square_test,
    claim_rate_z_test,
    independent_t_test,
)


def test_add_margin_and_claim_flag() -> None:
    df = pd.DataFrame({"TotalPremium": [100, 200], "TotalClaims": [0, 50]})

    result = add_margin_and_claim_flag(df)

    assert result["Margin"].tolist() == [100, 150]
    assert result["HasClaim"].tolist() == [0, 1]


def test_chi_square_test_returns_summary_fields() -> None:
    df = pd.DataFrame(
        {
            "Province": ["A", "A", "B", "B", "B", "A"],
            "HasClaim": [1, 0, 1, 1, 0, 0],
        }
    )

    result = chi_square_test(
        df,
        group_col="Province",
        outcome_col="HasClaim",
        hypothesis="Claim frequency differs by province.",
        kpi="Claim Frequency",
    )

    assert result["test_used"] == "Chi-square test of independence"
    assert "p_value" in result
    assert "business_interpretation" in result


def test_independent_t_test_returns_group_means() -> None:
    df = pd.DataFrame(
        {
            "ZipCode": ["1000", "1000", "1000", "2000", "2000", "2000"],
            "Margin": [100, 120, 140, 80, 90, 110],
        }
    )

    result = independent_t_test(
        df,
        group_col="ZipCode",
        value_col="Margin",
        control_group="1000",
        test_group="2000",
        hypothesis="Margin differs by zip code.",
        kpi="Margin",
    )

    assert result["control_mean"] == 120
    assert result["test_mean"] == 280 / 3


def test_claim_rate_z_test_returns_rates() -> None:
    df = pd.DataFrame(
        {
            "Gender": ["Female", "Female", "Female", "Male", "Male", "Male"],
            "TotalClaims": [0, 100, 50, 0, 0, 20],
        }
    )

    result = claim_rate_z_test(
        df,
        group_col="Gender",
        claims_col="TotalClaims",
        control_group="Female",
        test_group="Male",
        hypothesis="Claim frequency differs by gender.",
    )

    assert result["control_rate"] == 2 / 3
    assert result["test_rate"] == 1 / 3


def test_build_hypothesis_summary_selects_report_columns() -> None:
    summary = build_hypothesis_summary(
        [
            {
                "hypothesis": "Risk differs.",
                "kpi": "Claim Frequency",
                "test_used": "Chi-square",
                "p_value": 0.1,
                "decision": "Do not reject null",
                "business_interpretation": "No evidence.",
                "unused": "drop me",
            }
        ]
    )

    assert "unused" not in summary.columns
    assert summary.loc[0, "hypothesis"] == "Risk differs."
