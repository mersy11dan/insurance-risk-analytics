"""Reusable hypothesis testing helpers for ACIS insurance risk analytics."""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy import stats


DEFAULT_ALPHA = 0.05
DEFAULT_CLAIMS_COL = "TotalClaims"
DEFAULT_MARGIN_COL = "Margin"
DEFAULT_PROVINCE_COL = "Province"
DEFAULT_ZIP_COL = "PostalCode"
DEFAULT_GENDER_COL = "Gender"


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")


def interpret_p_value(
    p_value: float,
    alpha: float = DEFAULT_ALPHA,
    business_metric: str = "the selected metric",
) -> str:
    """Return a business-friendly interpretation of a p-value."""
    if pd.isna(p_value):
        return "The test could not produce a reliable p-value for this comparison."

    if p_value < alpha:
        return (
            f"There is statistically significant evidence that {business_metric} "
            f"differs between the compared groups at alpha={alpha}."
        )

    return (
        f"There is not enough statistical evidence to conclude that {business_metric} "
        f"differs between the compared groups at alpha={alpha}."
    )


def _format_result(
    *,
    test_name: str,
    business_question: str,
    null_hypothesis: str,
    alternative_hypothesis: str,
    statistic: float,
    p_value: float,
    alpha: float,
    business_metric: str,
    sample_size: int,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a structured result that can be added to a summary table."""
    result = {
        "test_name": test_name,
        "business_question": business_question,
        "null_hypothesis": null_hypothesis,
        "alternative_hypothesis": alternative_hypothesis,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "reject_null": bool(p_value < alpha) if not pd.isna(p_value) else False,
        "sample_size": sample_size,
        "interpretation": interpret_p_value(p_value, alpha, business_metric),
    }

    if extra_fields:
        result.update(extra_fields)

    return result


def chi_square_test(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    alpha: float = DEFAULT_ALPHA,
    business_question: str | None = None,
) -> dict[str, Any]:
    """Test whether two categorical variables are independent.

    Use this for questions such as whether claim occurrence differs by province,
    zip code, gender, or another categorical segment.
    """
    _validate_columns(df, [group_col, outcome_col])

    test_data = df[[group_col, outcome_col]].dropna()
    contingency_table = pd.crosstab(test_data[group_col], test_data[outcome_col])

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError("Chi-square test requires at least two groups and two outcomes.")

    statistic, p_value, degrees_of_freedom, expected_counts = stats.chi2_contingency(
        contingency_table
    )

    metric = f"{outcome_col} distribution"
    question = (
        business_question
        or f"Does {metric} differ across {group_col} segments?"
    )

    return _format_result(
        test_name="chi_square_independence",
        business_question=question,
        null_hypothesis=f"{outcome_col} is independent of {group_col}.",
        alternative_hypothesis=f"{outcome_col} is associated with {group_col}.",
        statistic=float(statistic),
        p_value=float(p_value),
        alpha=alpha,
        business_metric=metric,
        sample_size=int(contingency_table.to_numpy().sum()),
        extra_fields={
            "group_column": group_col,
            "outcome_column": outcome_col,
            "degrees_of_freedom": int(degrees_of_freedom),
            "contingency_table": contingency_table,
            "expected_counts": pd.DataFrame(
                expected_counts,
                index=contingency_table.index,
                columns=contingency_table.columns,
            ),
        },
    )


def independent_t_test(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    group_a: str,
    group_b: str,
    alpha: float = DEFAULT_ALPHA,
    equal_var: bool = False,
    business_question: str | None = None,
) -> dict[str, Any]:
    """Compare the mean of a numeric metric between two groups.

    Welch's t-test is used by default because insurance segments often have
    different sample sizes and variances.
    """
    _validate_columns(df, [group_col, value_col])

    group_a_values = df.loc[df[group_col] == group_a, value_col].dropna()
    group_b_values = df.loc[df[group_col] == group_b, value_col].dropna()

    if len(group_a_values) < 2 or len(group_b_values) < 2:
        raise ValueError("T-test requires at least two non-missing values per group.")

    statistic, p_value = stats.ttest_ind(
        group_a_values,
        group_b_values,
        equal_var=equal_var,
        nan_policy="omit",
    )

    metric = f"mean {value_col}"
    question = (
        business_question
        or f"Does {metric} differ between {group_a} and {group_b}?"
    )

    return _format_result(
        test_name="independent_t_test",
        business_question=question,
        null_hypothesis=f"{metric} is equal for {group_a} and {group_b}.",
        alternative_hypothesis=f"{metric} differs between {group_a} and {group_b}.",
        statistic=float(statistic),
        p_value=float(p_value),
        alpha=alpha,
        business_metric=metric,
        sample_size=int(len(group_a_values) + len(group_b_values)),
        extra_fields={
            "group_column": group_col,
            "value_column": value_col,
            "group_a": group_a,
            "group_b": group_b,
            "group_a_mean": float(group_a_values.mean()),
            "group_b_mean": float(group_b_values.mean()),
            "group_a_size": int(len(group_a_values)),
            "group_b_size": int(len(group_b_values)),
            "equal_variance_assumed": equal_var,
        },
    )


def two_proportion_z_test(
    successes_a: int,
    observations_a: int,
    successes_b: int,
    observations_b: int,
    alpha: float = DEFAULT_ALPHA,
    business_question: str = "Do the two proportions differ?",
    business_metric: str = "claim rate",
) -> dict[str, Any]:
    """Compare two proportions with a pooled two-sided z-test."""
    if observations_a <= 0 or observations_b <= 0:
        raise ValueError("Observation counts must be greater than zero.")

    if successes_a > observations_a or successes_b > observations_b:
        raise ValueError("Success counts cannot exceed observation counts.")

    proportion_a = successes_a / observations_a
    proportion_b = successes_b / observations_b
    pooled_proportion = (successes_a + successes_b) / (observations_a + observations_b)
    standard_error = (
        pooled_proportion
        * (1 - pooled_proportion)
        * ((1 / observations_a) + (1 / observations_b))
    ) ** 0.5

    if standard_error == 0:
        statistic = float("nan")
        p_value = float("nan")
    else:
        statistic = (proportion_a - proportion_b) / standard_error
        p_value = 2 * stats.norm.sf(abs(statistic))

    return _format_result(
        test_name="two_proportion_z_test",
        business_question=business_question,
        null_hypothesis="The two proportions are equal.",
        alternative_hypothesis="The two proportions are different.",
        statistic=float(statistic),
        p_value=float(p_value),
        alpha=alpha,
        business_metric=business_metric,
        sample_size=int(observations_a + observations_b),
        extra_fields={
            "group_a_successes": int(successes_a),
            "group_a_observations": int(observations_a),
            "group_a_proportion": float(proportion_a),
            "group_b_successes": int(successes_b),
            "group_b_observations": int(observations_b),
            "group_b_proportion": float(proportion_b),
        },
    )


def claim_rate_z_test(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    claims_col: str = DEFAULT_CLAIMS_COL,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Compare claim occurrence rates between two groups using a z-test."""
    _validate_columns(df, [group_col, claims_col])

    group_a_data = df[df[group_col] == group_a]
    group_b_data = df[df[group_col] == group_b]

    successes_a = int((group_a_data[claims_col] > 0).sum())
    successes_b = int((group_b_data[claims_col] > 0).sum())

    return two_proportion_z_test(
        successes_a=successes_a,
        observations_a=len(group_a_data),
        successes_b=successes_b,
        observations_b=len(group_b_data),
        alpha=alpha,
        business_question=(
            f"Does claim rate differ between {group_a} and {group_b} "
            f"within {group_col}?"
        ),
        business_metric="claim rate",
    )


def test_margin_by_gender(
    df: pd.DataFrame,
    gender_a: str,
    gender_b: str,
    gender_col: str = DEFAULT_GENDER_COL,
    margin_col: str = DEFAULT_MARGIN_COL,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Compare average margin between two gender groups."""
    return independent_t_test(
        df=df,
        group_col=gender_col,
        value_col=margin_col,
        group_a=gender_a,
        group_b=gender_b,
        alpha=alpha,
        business_question=f"Does average margin differ between {gender_a} and {gender_b}?",
    )


def test_margin_by_province(
    df: pd.DataFrame,
    province_a: str,
    province_b: str,
    province_col: str = DEFAULT_PROVINCE_COL,
    margin_col: str = DEFAULT_MARGIN_COL,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Compare average margin between two provinces."""
    return independent_t_test(
        df=df,
        group_col=province_col,
        value_col=margin_col,
        group_a=province_a,
        group_b=province_b,
        alpha=alpha,
        business_question=(
            f"Does average margin differ between {province_a} and {province_b}?"
        ),
    )


def test_margin_by_zip_code(
    df: pd.DataFrame,
    zip_a: str,
    zip_b: str,
    zip_col: str = DEFAULT_ZIP_COL,
    margin_col: str = DEFAULT_MARGIN_COL,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Compare average margin between two zip or postal code groups."""
    return independent_t_test(
        df=df,
        group_col=zip_col,
        value_col=margin_col,
        group_a=zip_a,
        group_b=zip_b,
        alpha=alpha,
        business_question=f"Does average margin differ between {zip_a} and {zip_b}?",
    )


def build_hypothesis_summary(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert structured hypothesis test results into a summary table."""
    summary_columns = [
        "test_name",
        "business_question",
        "statistic",
        "p_value",
        "alpha",
        "reject_null",
        "sample_size",
        "interpretation",
    ]

    summary = pd.DataFrame(results)
    available_columns = [column for column in summary_columns if column in summary.columns]
    return summary[available_columns]
