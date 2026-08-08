"""Reusable hypothesis testing helpers for ACIS insurance analytics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_ALPHA = 0.05


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if any required column is missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")


def interpret_p_value(
    p_value: float,
    alpha: float = DEFAULT_ALPHA,
    rejected_message: str | None = None,
    not_rejected_message: str | None = None,
) -> str:
    """Return a business-friendly interpretation for a hypothesis test."""
    if np.isnan(p_value):
        return "The test could not produce a valid p-value."

    if p_value < alpha:
        return rejected_message or (
            "Reject the null hypothesis. The observed difference is statistically "
            "significant and should be investigated as a potential business signal."
        )

    return not_rejected_message or (
        "Do not reject the null hypothesis. The data does not provide enough "
        "evidence of a statistically significant difference."
    )


def format_test_result(
    *,
    hypothesis: str,
    test_used: str,
    p_value: float,
    statistic: float,
    decision: str,
    business_interpretation: str,
    kpi: str,
    control_group: str | None = None,
    test_group: str | None = None,
    sample_size: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a structured result row suitable for a report table."""
    result = {
        "hypothesis": hypothesis,
        "kpi": kpi,
        "test_used": test_used,
        "statistic": statistic,
        "p_value": p_value,
        "decision": decision,
        "business_interpretation": business_interpretation,
        "control_group": control_group,
        "test_group": test_group,
        "sample_size": sample_size,
    }
    if extra:
        result.update(extra)
    return result


def chi_square_test(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    hypothesis: str,
    kpi: str,
    alpha: float = DEFAULT_ALPHA,
    rejected_message: str | None = None,
    not_rejected_message: str | None = None,
) -> dict[str, Any]:
    """Run a chi-square independence test for categorical segment risk."""
    validate_columns(df, [group_col, outcome_col])
    test_data = df[[group_col, outcome_col]].dropna()
    contingency_table = pd.crosstab(test_data[group_col], test_data[outcome_col])

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError("Chi-square test requires at least two groups and outcomes.")

    statistic, p_value, degrees_of_freedom, expected_counts = stats.chi2_contingency(
        contingency_table
    )
    decision = "Reject null" if p_value < alpha else "Do not reject null"

    return format_test_result(
        hypothesis=hypothesis,
        test_used="Chi-square test of independence",
        p_value=float(p_value),
        statistic=float(statistic),
        decision=decision,
        business_interpretation=interpret_p_value(
            float(p_value), alpha, rejected_message, not_rejected_message
        ),
        kpi=kpi,
        sample_size=int(len(test_data)),
        extra={
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
    control_group: Any,
    test_group: Any,
    hypothesis: str,
    kpi: str,
    alpha: float = DEFAULT_ALPHA,
    equal_var: bool = False,
    rejected_message: str | None = None,
    not_rejected_message: str | None = None,
) -> dict[str, Any]:
    """Run Welch's t-test for a numeric KPI between two groups."""
    validate_columns(df, [group_col, value_col])
    control_values = df.loc[df[group_col] == control_group, value_col].dropna()
    test_values = df.loc[df[group_col] == test_group, value_col].dropna()

    if len(control_values) < 2 or len(test_values) < 2:
        raise ValueError("T-test requires at least two observations in each group.")

    statistic, p_value = stats.ttest_ind(
        control_values,
        test_values,
        equal_var=equal_var,
    )
    decision = "Reject null" if p_value < alpha else "Do not reject null"

    return format_test_result(
        hypothesis=hypothesis,
        test_used="Welch's t-test" if not equal_var else "Independent t-test",
        p_value=float(p_value),
        statistic=float(statistic),
        decision=decision,
        business_interpretation=interpret_p_value(
            float(p_value), alpha, rejected_message, not_rejected_message
        ),
        kpi=kpi,
        control_group=str(control_group),
        test_group=str(test_group),
        sample_size=int(len(control_values) + len(test_values)),
        extra={
            "control_mean": float(control_values.mean()),
            "test_mean": float(test_values.mean()),
            "control_size": int(len(control_values)),
            "test_size": int(len(test_values)),
        },
    )


def two_proportion_z_test(
    *,
    control_successes: int,
    control_total: int,
    test_successes: int,
    test_total: int,
    hypothesis: str,
    kpi: str,
    control_group: str,
    test_group: str,
    alpha: float = DEFAULT_ALPHA,
    rejected_message: str | None = None,
    not_rejected_message: str | None = None,
) -> dict[str, Any]:
    """Run a two-sided z-test for two proportions."""
    if control_total <= 0 or test_total <= 0:
        raise ValueError("Group totals must be positive.")
    if control_successes > control_total or test_successes > test_total:
        raise ValueError("Success counts cannot exceed totals.")

    control_rate = control_successes / control_total
    test_rate = test_successes / test_total
    pooled_rate = (control_successes + test_successes) / (control_total + test_total)
    standard_error = (
        pooled_rate
        * (1 - pooled_rate)
        * ((1 / control_total) + (1 / test_total))
    ) ** 0.5

    if standard_error == 0:
        statistic = np.nan
        p_value = np.nan
    else:
        statistic = (control_rate - test_rate) / standard_error
        p_value = 2 * stats.norm.sf(abs(statistic))

    decision = "Reject null" if p_value < alpha else "Do not reject null"

    return format_test_result(
        hypothesis=hypothesis,
        test_used="Two-proportion z-test",
        p_value=float(p_value),
        statistic=float(statistic),
        decision=decision,
        business_interpretation=interpret_p_value(
            float(p_value), alpha, rejected_message, not_rejected_message
        ),
        kpi=kpi,
        control_group=control_group,
        test_group=test_group,
        sample_size=int(control_total + test_total),
        extra={
            "control_rate": float(control_rate),
            "test_rate": float(test_rate),
            "control_successes": int(control_successes),
            "test_successes": int(test_successes),
        },
    )


def claim_rate_z_test(
    df: pd.DataFrame,
    group_col: str,
    claims_col: str,
    control_group: Any,
    test_group: Any,
    hypothesis: str,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Compare claim frequency between two groups using a z-test."""
    validate_columns(df, [group_col, claims_col])
    control = df[df[group_col] == control_group]
    test = df[df[group_col] == test_group]

    return two_proportion_z_test(
        control_successes=int((control[claims_col] > 0).sum()),
        control_total=int(len(control)),
        test_successes=int((test[claims_col] > 0).sum()),
        test_total=int(len(test)),
        hypothesis=hypothesis,
        kpi="Claim Frequency",
        control_group=str(control_group),
        test_group=str(test_group),
        alpha=alpha,
    )


def add_margin_and_claim_flag(
    df: pd.DataFrame,
    premium_col: str = "TotalPremium",
    claims_col: str = "TotalClaims",
) -> pd.DataFrame:
    """Return a copy with Margin and HasClaim columns for testing."""
    validate_columns(df, [premium_col, claims_col])
    result = df.copy()
    result["Margin"] = result[premium_col] - result[claims_col]
    result["HasClaim"] = (result[claims_col] > 0).astype(int)
    return result


def top_two_groups(df: pd.DataFrame, group_col: str) -> tuple[Any, Any]:
    """Return the two largest groups by row count for stable pairwise tests."""
    validate_columns(df, [group_col])
    counts = df[group_col].dropna().value_counts()
    if len(counts) < 2:
        raise ValueError(f"{group_col} must contain at least two groups.")
    group_a, group_b = counts.head(2).index.tolist()
    return group_a, group_b


def build_hypothesis_summary(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert detailed test results into a concise summary table."""
    columns = [
        "hypothesis",
        "kpi",
        "test_used",
        "p_value",
        "decision",
        "business_interpretation",
        "control_group",
        "test_group",
        "sample_size",
    ]
    summary = pd.DataFrame(results)
    return summary[[column for column in columns if column in summary.columns]]
