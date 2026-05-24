import pandas as pd

from src.hypothesis_tests import (
    build_hypothesis_summary,
    chi_square_test,
    claim_rate_z_test,
    independent_t_test,
    interpret_p_value,
)


def test_interpret_p_value_returns_business_friendly_text() -> None:
    interpretation = interpret_p_value(0.01, alpha=0.05, business_metric="margin")

    assert "statistically significant evidence" in interpretation
    assert "margin" in interpretation


def test_chi_square_test_returns_structured_result() -> None:
    df = pd.DataFrame(
        {
            "Province": ["A", "A", "B", "B", "B", "A"],
            "HasClaim": [1, 0, 1, 1, 0, 0],
        }
    )

    result = chi_square_test(df, group_col="Province", outcome_col="HasClaim")

    assert result["test_name"] == "chi_square_independence"
    assert result["group_column"] == "Province"
    assert result["outcome_column"] == "HasClaim"
    assert "p_value" in result
    assert "interpretation" in result


def test_independent_t_test_returns_group_means() -> None:
    df = pd.DataFrame(
        {
            "Gender": ["Female", "Female", "Female", "Male", "Male", "Male"],
            "Margin": [100.0, 120.0, 140.0, 80.0, 90.0, 110.0],
        }
    )

    result = independent_t_test(
        df,
        group_col="Gender",
        value_col="Margin",
        group_a="Female",
        group_b="Male",
    )

    assert result["test_name"] == "independent_t_test"
    assert result["group_a_mean"] == 120.0
    assert result["group_b_mean"] == 280.0 / 3.0


def test_claim_rate_z_test_returns_proportions() -> None:
    df = pd.DataFrame(
        {
            "Province": ["A", "A", "A", "B", "B", "B"],
            "TotalClaims": [10.0, 0.0, 5.0, 0.0, 0.0, 3.0],
        }
    )

    result = claim_rate_z_test(df, group_col="Province", group_a="A", group_b="B")

    assert result["test_name"] == "two_proportion_z_test"
    assert result["group_a_proportion"] == 2 / 3
    assert result["group_b_proportion"] == 1 / 3


def test_build_hypothesis_summary_returns_table() -> None:
    result = {
        "test_name": "example",
        "business_question": "Does risk differ?",
        "statistic": 1.2,
        "p_value": 0.03,
        "alpha": 0.05,
        "reject_null": True,
        "sample_size": 10,
        "interpretation": "Evidence of a difference.",
        "contingency_table": "not needed in summary",
    }

    summary = build_hypothesis_summary([result])

    assert list(summary.columns) == [
        "test_name",
        "business_question",
        "statistic",
        "p_value",
        "alpha",
        "reject_null",
        "sample_size",
        "interpretation",
    ]
    assert summary.loc[0, "test_name"] == "example"
