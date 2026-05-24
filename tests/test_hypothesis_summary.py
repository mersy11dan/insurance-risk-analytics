import pandas as pd

from src.hypothesis_summary import (
    build_report_row,
    markdown_table,
    run_hypothesis_suite,
)


def test_build_report_row_uses_rejected_interpretation() -> None:
    result = {
        "test_name": "independent_t_test",
        "p_value": 0.01,
        "reject_null": True,
    }

    row = build_report_row(
        result,
        hypothesis="Average margin differs by segment.",
        rejected_interpretation="Use this in the report.",
        not_rejected_interpretation="Do not use this.",
    )

    assert row["decision"] == "Reject null"
    assert row["business_interpretation"] == "Use this in the report."


def test_markdown_table_renders_reusable_summary() -> None:
    summary = pd.DataFrame(
        [
            {
                "hypothesis": "Claim risk differs.",
                "test_used": "chi_square_independence",
                "p_value": 0.02,
                "decision": "Reject null",
                "business_interpretation": "Review pricing.",
            }
        ]
    )

    output = markdown_table(summary)

    assert "| hypothesis | test_used | p_value | decision | business_interpretation |" in output
    assert "Claim risk differs." in output


def test_run_hypothesis_suite_returns_report_ready_columns() -> None:
    df = pd.DataFrame(
        {
            "Province": ["A", "A", "A", "B", "B", "B"],
            "Gender": ["Female", "Female", "Male", "Male", "Male", "Female"],
            "ZipCode": ["1000", "1000", "1001", "1001", "1001", "1000"],
            "TotalPremium": [100.0, 120.0, 90.0, 150.0, 160.0, 110.0],
            "TotalClaims": [0.0, 50.0, 0.0, 100.0, 80.0, 0.0],
        }
    )

    summary = run_hypothesis_suite(df)

    assert list(summary.columns) == [
        "hypothesis",
        "test_used",
        "p_value",
        "decision",
        "business_interpretation",
    ]
    assert not summary.empty
