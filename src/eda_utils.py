"""Reusable EDA helpers for ACIS insurance risk analytics."""

from __future__ import annotations

import pandas as pd

DEFAULT_PREMIUM_COL = "TotalPremium"
DEFAULT_CLAIMS_COL = "TotalClaims"
DEFAULT_PROVINCE_COL = "Province"
DEFAULT_VEHICLE_TYPE_COL = "VehicleType"
DEFAULT_GENDER_COL = "Gender"


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if one or more expected columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")


def descriptive_statistics(
    df: pd.DataFrame, include: str | list[str] = "all"
) -> pd.DataFrame:
    """Return descriptive statistics for numeric and categorical columns."""
    return df.describe(include=include).transpose()


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values by count and percentage."""
    summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percentage": (df.isna().mean() * 100).round(2),
        }
    )
    return summary.sort_values("missing_count", ascending=False)


def calculate_loss_ratio(
    df: pd.DataFrame,
    premium_col: str = DEFAULT_PREMIUM_COL,
    claims_col: str = DEFAULT_CLAIMS_COL,
) -> float:
    """Calculate total claims divided by total premium.

    A lower loss ratio generally indicates a more profitable insurance segment.
    """
    _validate_columns(df, [premium_col, claims_col])

    total_premium = df[premium_col].sum()
    if total_premium == 0:
        return 0.0

    return float(df[claims_col].sum() / total_premium)


def calculate_margin(
    df: pd.DataFrame,
    premium_col: str = DEFAULT_PREMIUM_COL,
    claims_col: str = DEFAULT_CLAIMS_COL,
) -> float:
    """Calculate total underwriting margin as premium minus claims."""
    _validate_columns(df, [premium_col, claims_col])
    return float(df[premium_col].sum() - df[claims_col].sum())


def add_risk_metrics(
    df: pd.DataFrame,
    premium_col: str = DEFAULT_PREMIUM_COL,
    claims_col: str = DEFAULT_CLAIMS_COL,
) -> pd.DataFrame:
    """Return a copy of the data with row-level margin and loss-ratio fields."""
    _validate_columns(df, [premium_col, claims_col])

    result = df.copy()
    result["Margin"] = result[premium_col] - result[claims_col]

    # Avoid division errors for policies with zero or missing premium.
    result["LossRatio"] = result[claims_col].div(result[premium_col]).fillna(0)
    result.loc[result[premium_col] == 0, "LossRatio"] = 0

    return result


def group_risk_summary(
    df: pd.DataFrame,
    group_col: str,
    premium_col: str = DEFAULT_PREMIUM_COL,
    claims_col: str = DEFAULT_CLAIMS_COL,
) -> pd.DataFrame:
    """Summarize exposure, claims, loss ratio, and margin by one group column."""
    _validate_columns(df, [group_col, premium_col, claims_col])

    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            policy_count=(group_col, "size"),
            total_premium=(premium_col, "sum"),
            total_claims=(claims_col, "sum"),
            average_premium=(premium_col, "mean"),
            average_claim=(claims_col, "mean"),
        )
        .reset_index()
    )

    grouped["loss_ratio"] = grouped["total_claims"].div(grouped["total_premium"])
    grouped["loss_ratio"] = grouped["loss_ratio"].fillna(0)
    grouped.loc[grouped["total_premium"] == 0, "loss_ratio"] = 0
    grouped["margin"] = grouped["total_premium"] - grouped["total_claims"]
    grouped["margin_percentage"] = grouped["margin"].div(grouped["total_premium"])
    grouped["margin_percentage"] = grouped["margin_percentage"].fillna(0)

    return grouped.sort_values("loss_ratio", ascending=False)


def group_by_province(
    df: pd.DataFrame,
    province_col: str = DEFAULT_PROVINCE_COL,
    premium_col: str = DEFAULT_PREMIUM_COL,
    claims_col: str = DEFAULT_CLAIMS_COL,
) -> pd.DataFrame:
    """Summarize risk and profitability by province."""
    return group_risk_summary(df, province_col, premium_col, claims_col)


def group_by_vehicle_type(
    df: pd.DataFrame,
    vehicle_type_col: str = DEFAULT_VEHICLE_TYPE_COL,
    premium_col: str = DEFAULT_PREMIUM_COL,
    claims_col: str = DEFAULT_CLAIMS_COL,
) -> pd.DataFrame:
    """Summarize risk and profitability by vehicle type."""
    return group_risk_summary(df, vehicle_type_col, premium_col, claims_col)


def group_by_gender(
    df: pd.DataFrame,
    gender_col: str = DEFAULT_GENDER_COL,
    premium_col: str = DEFAULT_PREMIUM_COL,
    claims_col: str = DEFAULT_CLAIMS_COL,
) -> pd.DataFrame:
    """Summarize risk and profitability by gender."""
    return group_risk_summary(df, gender_col, premium_col, claims_col)
