"""Generate report-ready hypothesis testing summaries for ACIS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import load_insurance_data
from src.eda_utils import add_risk_metrics
from src.hypothesis_tests import (
    chi_square_test,
    claim_rate_z_test,
    independent_t_test,
)


DEFAULT_INPUT = Path("data/processed/insurance_data_cleaned.csv")
DEFAULT_CSV_OUTPUT = Path("reports/hypothesis_test_results.csv")
DEFAULT_MARKDOWN_OUTPUT = Path("reports/hypothesis_test_results.md")
DEFAULT_ALPHA = 0.05


def first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first available column from a list of candidate names."""
    return next((candidate for candidate in candidates if candidate in columns), None)


def top_two_groups(df: pd.DataFrame, group_col: str) -> tuple[Any, Any]:
    """Return the two largest non-missing groups for a stable comparison."""
    group_counts = df[group_col].dropna().value_counts()
    if len(group_counts) < 2:
        raise ValueError(f"{group_col} must contain at least two groups.")

    group_a, group_b = group_counts.head(2).index.tolist()
    return group_a, group_b


def decision_label(reject_null: bool) -> str:
    """Convert a boolean hypothesis decision into report language."""
    return "Reject null" if reject_null else "Do not reject null"


def build_report_row(
    result: dict[str, Any],
    hypothesis: str,
    rejected_interpretation: str,
    not_rejected_interpretation: str,
) -> dict[str, Any]:
    """Convert a statistical result into a final-report-friendly row."""
    reject_null = bool(result["reject_null"])

    return {
        "hypothesis": hypothesis,
        "test_used": result["test_name"],
        "p_value": round(float(result["p_value"]), 6),
        "decision": decision_label(reject_null),
        "business_interpretation": rejected_interpretation
        if reject_null
        else not_rejected_interpretation,
    }


def run_hypothesis_suite(df: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Run ACIS hypothesis tests and return a report-ready summary table."""
    analysis_df = add_risk_metrics(df)
    analysis_df["HasClaim"] = (analysis_df["TotalClaims"] > 0).astype(int)

    columns = analysis_df.columns.tolist()
    province_col = first_existing_column(columns, ["Province", "province"])
    gender_col = first_existing_column(columns, ["Gender", "gender"])
    zip_col = first_existing_column(
        columns, ["ZipCode", "PostalCode", "Postal_Code", "ZIPCode", "Zip", "zip"]
    )

    rows: list[dict[str, Any]] = []

    if province_col:
        result = chi_square_test(
            analysis_df,
            group_col=province_col,
            outcome_col="HasClaim",
            alpha=alpha,
            business_question="Does claim risk differ across provinces?",
        )
        rows.append(
            build_report_row(
                result,
                hypothesis="Claim risk differs across provinces.",
                rejected_interpretation=(
                    "Claim occurrence differs significantly by province. ACIS should "
                    "review province-level pricing, underwriting rules, and marketing "
                    "focus before expanding in higher-risk regions."
                ),
                not_rejected_interpretation=(
                    "The data does not provide sufficient evidence that claim "
                    "occurrence differs by province at the selected significance level."
                ),
            )
        )

    if gender_col:
        result = chi_square_test(
            analysis_df,
            group_col=gender_col,
            outcome_col="HasClaim",
            alpha=alpha,
            business_question="Does claim risk differ by gender?",
        )
        rows.append(
            build_report_row(
                result,
                hypothesis="Claim risk differs by gender.",
                rejected_interpretation=(
                    "Claim occurrence differs significantly by gender. ACIS should "
                    "investigate whether this reflects exposure mix, vehicle choice, "
                    "geography, or policy features before making business decisions."
                ),
                not_rejected_interpretation=(
                    "The data does not provide sufficient evidence that claim "
                    "occurrence differs by gender at the selected significance level."
                ),
            )
        )

    if province_col:
        province_a, province_b = top_two_groups(analysis_df, province_col)
        result = claim_rate_z_test(
            analysis_df,
            group_col=province_col,
            group_a=province_a,
            group_b=province_b,
            alpha=alpha,
        )
        rows.append(
            build_report_row(
                result,
                hypothesis=f"Claim rate differs between {province_a} and {province_b}.",
                rejected_interpretation=(
                    f"Claim rates differ significantly between {province_a} and "
                    f"{province_b}. ACIS should compare local pricing adequacy and "
                    "claim drivers in these high-volume provinces."
                ),
                not_rejected_interpretation=(
                    f"The data does not provide sufficient evidence that claim rates "
                    f"differ between {province_a} and {province_b}."
                ),
            )
        )

    if zip_col:
        zip_a, zip_b = top_two_groups(analysis_df, zip_col)
        result = independent_t_test(
            analysis_df,
            group_col=zip_col,
            value_col="Margin",
            group_a=zip_a,
            group_b=zip_b,
            alpha=alpha,
            business_question=(
                f"Does average margin differ between zip codes {zip_a} and {zip_b}?"
            ),
        )
        rows.append(
            build_report_row(
                result,
                hypothesis=f"Average margin differs between zip codes {zip_a} and {zip_b}.",
                rejected_interpretation=(
                    f"Average margin differs significantly between zip codes {zip_a} "
                    f"and {zip_b}. ACIS should review localized pricing, claim costs, "
                    "and customer mix in these areas."
                ),
                not_rejected_interpretation=(
                    f"The data does not provide sufficient evidence that average "
                    f"margin differs between zip codes {zip_a} and {zip_b}."
                ),
            )
        )

    if gender_col:
        gender_a, gender_b = top_two_groups(analysis_df, gender_col)
        result = independent_t_test(
            analysis_df,
            group_col=gender_col,
            value_col="Margin",
            group_a=gender_a,
            group_b=gender_b,
            alpha=alpha,
            business_question=f"Does average margin differ between {gender_a} and {gender_b}?",
        )
        rows.append(
            build_report_row(
                result,
                hypothesis=f"Average margin differs between {gender_a} and {gender_b}.",
                rejected_interpretation=(
                    f"Average margin differs significantly between {gender_a} and "
                    f"{gender_b}. ACIS should examine whether this is explained by "
                    "vehicle type, geography, policy terms, or exposure differences."
                ),
                not_rejected_interpretation=(
                    f"The data does not provide sufficient evidence that average "
                    f"margin differs between {gender_a} and {gender_b}."
                ),
            )
        )

    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a simple GitHub-flavored Markdown table."""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def save_report_outputs(
    summary: pd.DataFrame,
    csv_output: Path = DEFAULT_CSV_OUTPUT,
    markdown_output: Path = DEFAULT_MARKDOWN_OUTPUT,
) -> None:
    """Save hypothesis results as CSV and Markdown for final-report reuse."""
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(csv_output, index=False)

    rejected = summary[summary["decision"] == "Reject null"]
    markdown_sections = [
        "# Hypothesis Test Results",
        "",
        markdown_table(summary),
        "",
        "## Rejected Hypotheses",
        "",
    ]

    if rejected.empty:
        markdown_sections.append(
            "No hypotheses were rejected at the selected significance level."
        )
    else:
        for _, row in rejected.iterrows():
            markdown_sections.append(
                f"- **{row['hypothesis']}** {row['business_interpretation']}"
            )

    markdown_output.write_text("\n".join(markdown_sections) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for hypothesis summary generation."""
    parser = argparse.ArgumentParser(description="Generate ACIS hypothesis summaries.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return parser.parse_args()


def main() -> None:
    """Generate and save report-ready hypothesis test results."""
    args = parse_args()
    df = load_insurance_data(args.input)
    summary = run_hypothesis_suite(df, alpha=args.alpha)
    save_report_outputs(summary, args.csv_output, args.markdown_output)
    print(f"Saved hypothesis results to {args.csv_output}")
    print(f"Saved report summary to {args.markdown_output}")


if __name__ == "__main__":
    main()
