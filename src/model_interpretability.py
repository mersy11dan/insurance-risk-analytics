"""Model interpretability helpers for ACIS claim severity models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline


def sample_features(
    X: pd.DataFrame,
    max_rows: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a reproducible sample for interpretability calculations."""
    if len(X) <= max_rows:
        return X.copy()

    return X.sample(n=max_rows, random_state=random_state)


def get_transformed_feature_names(model_pipeline: Pipeline) -> list[str]:
    """Return feature names after preprocessing and one-hot encoding."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    return [str(feature_name) for feature_name in feature_names]


def transform_features(model_pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted preprocessor and return a named feature matrix."""
    preprocessor = model_pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    feature_names = get_transformed_feature_names(model_pipeline)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def clean_feature_name(feature_name: str) -> str:
    """Convert encoded pipeline feature names into report-friendly labels."""
    cleaned = feature_name
    for prefix in ("numeric__", "categorical__"):
        cleaned = cleaned.replace(prefix, "")

    return cleaned.replace("_", " ")


def business_feature_explanation(feature_name: str) -> str:
    """Return a business-facing explanation for a model feature."""
    label = clean_feature_name(feature_name)
    lower_label = label.lower()

    if "premium" in lower_label:
        return (
            f"`{label}` reflects pricing or premium level. If influential, the model "
            "is using premium adequacy as a signal for expected claim severity."
        )
    if "claim" in lower_label or "pastclaims" in lower_label:
        return (
            f"`{label}` reflects prior or current claim behavior. Higher influence "
            "suggests historical claims are important for severity prediction."
        )
    if "vehicle" in lower_label or "automake" in lower_label or "model" in lower_label:
        return (
            f"`{label}` captures vehicle characteristics. Influence here may reflect "
            "repair costs, vehicle value, or risk differences by make and model."
        )
    if "province" in lower_label or "zip" in lower_label:
        return (
            f"`{label}` captures location effects. Influence here may reflect local "
            "claim patterns, repair costs, or driving risk."
        )
    if "gender" in lower_label or "age" in lower_label or "income" in lower_label:
        return (
            f"`{label}` captures customer profile information. Interpret this carefully "
            "alongside fairness, regulation, and other exposure differences."
        )
    if "deductible" in lower_label or "cover" in lower_label or "ncd" in lower_label:
        return (
            f"`{label}` captures policy design. Influence here may reflect coverage "
            "level, customer risk selection, or incentives to claim."
        )

    return (
        f"`{label}` is influential in the model and should be reviewed with the "
        "business team to understand its pricing and underwriting meaning."
    )


def compute_shap_values(
    model_pipeline: Pipeline,
    X: pd.DataFrame,
    max_rows: int = 500,
    random_state: int = 42,
) -> tuple[Any, pd.DataFrame]:
    """Compute SHAP values for a fitted model pipeline.

    Tree models use SHAP's tree explainer. Other estimators fall back to the
    generic explainer on the transformed feature matrix.
    """
    import shap

    X_sample = sample_features(X, max_rows=max_rows, random_state=random_state)
    transformed_X = transform_features(model_pipeline, X_sample)
    estimator = model_pipeline.named_steps["model"]

    estimator_name = estimator.__class__.__name__.lower()
    if "forest" in estimator_name or "xgb" in estimator_name:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed_X)
    else:
        explainer = shap.Explainer(estimator.predict, transformed_X)
        shap_values = explainer(transformed_X).values

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    return shap_values, transformed_X


def shap_feature_importance(
    shap_values: Any,
    transformed_X: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Summarize top features by mean absolute SHAP value."""
    importance = pd.DataFrame(
        {
            "feature": transformed_X.columns,
            "mean_abs_shap": abs(shap_values).mean(axis=0),
        }
    )
    importance["feature_label"] = importance["feature"].map(clean_feature_name)
    importance["business_explanation"] = importance["feature"].map(
        business_feature_explanation
    )

    return importance.sort_values("mean_abs_shap", ascending=False).head(top_n)


def save_shap_summary_plot(
    shap_values: Any,
    transformed_X: pd.DataFrame,
    output_path: str | Path,
    max_display: int = 10,
) -> None:
    """Save a SHAP summary plot for the final report."""
    import shap

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values,
        transformed_X,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
