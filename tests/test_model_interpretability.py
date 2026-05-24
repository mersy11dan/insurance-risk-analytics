import pandas as pd

from src.model_interpretability import (
    business_feature_explanation,
    clean_feature_name,
    sample_features,
)


def test_sample_features_returns_reproducible_subset() -> None:
    df = pd.DataFrame({"feature": range(20)})

    sample_a = sample_features(df, max_rows=5, random_state=42)
    sample_b = sample_features(df, max_rows=5, random_state=42)

    assert len(sample_a) == 5
    assert sample_a.equals(sample_b)


def test_clean_feature_name_removes_pipeline_prefixes() -> None:
    assert clean_feature_name("numeric__AnnualPremium") == "AnnualPremium"
    assert clean_feature_name("categorical__Province_Addis Ababa") == (
        "Province Addis Ababa"
    )


def test_business_feature_explanation_is_report_friendly() -> None:
    explanation = business_feature_explanation("numeric__AnnualPremium")

    assert "premium" in explanation.lower()
    assert "claim severity" in explanation.lower()
