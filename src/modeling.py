"""Modeling utilities for the ACIS insurance risk analytics project."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TaskType = Literal["regression", "classification"]


def validate_modeling_columns(
    df: pd.DataFrame, target_col: str, feature_cols: list[str] | None = None
) -> list[str]:
    """Validate modeling columns and return the final feature list."""
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    if feature_cols is None:
        feature_cols = [column for column in df.columns if column != target_col]

    missing_features = [column for column in feature_cols if column not in df.columns]
    if missing_features:
        raise KeyError(f"Feature column(s) not found: {', '.join(missing_features)}")

    if not feature_cols:
        raise ValueError("At least one feature column is required for modeling.")

    return feature_cols


def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix `X` and target vector `y`."""
    feature_cols = validate_modeling_columns(df, target_col, feature_cols)
    modeling_data = df[feature_cols + [target_col]].copy()
    modeling_data = modeling_data.dropna(subset=[target_col])

    return modeling_data[feature_cols], modeling_data[target_col]


def infer_feature_types(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature columns from a feature matrix."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
        column for column in X.columns if column not in numeric_features
    ]

    return numeric_features, categorical_features


def build_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric and categorical features.

    Numeric values are median-imputed. Categorical values are imputed with the
    most frequent value and one-hot encoded with unknown categories ignored.
    """
    numeric_features, categorical_features = infer_feature_types(X)

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No numeric or categorical features were found.")

    return ColumnTransformer(transformers=transformers)


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train and test sets."""
    stratify_values = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )


def build_linear_model(task_type: TaskType = "regression", **model_kwargs: Any) -> Any:
    """Build a linear regression or logistic regression estimator."""
    if task_type == "regression":
        return LinearRegression(**model_kwargs)

    return LogisticRegression(max_iter=model_kwargs.pop("max_iter", 1000), **model_kwargs)


def build_random_forest_model(
    task_type: TaskType = "regression",
    random_state: int = 42,
    **model_kwargs: Any,
) -> Any:
    """Build a Random Forest estimator for regression or classification."""
    if task_type == "regression":
        return RandomForestRegressor(random_state=random_state, **model_kwargs)

    return RandomForestClassifier(random_state=random_state, **model_kwargs)


def build_xgboost_model(
    task_type: TaskType = "regression",
    random_state: int = 42,
    **model_kwargs: Any,
) -> Any:
    """Build an XGBoost estimator for regression or classification."""
    if task_type == "regression":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective=model_kwargs.pop("objective", "reg:squarederror"),
            random_state=random_state,
            **model_kwargs,
        )

    from xgboost import XGBClassifier

    return XGBClassifier(
        objective=model_kwargs.pop("objective", "binary:logistic"),
        eval_metric=model_kwargs.pop("eval_metric", "logloss"),
        random_state=random_state,
        **model_kwargs,
    )


def build_model_pipeline(
    model: Any,
    X: pd.DataFrame,
    scale_numeric: bool = False,
) -> Pipeline:
    """Combine preprocessing and an estimator into a single sklearn pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X, scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def train_model(
    df: pd.DataFrame,
    target_col: str,
    model: Any,
    feature_cols: list[str] | None = None,
    task_type: TaskType = "regression",
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numeric: bool = False,
    stratify: bool | None = None,
) -> dict[str, Any]:
    """Train a model pipeline and return fitted objects plus evaluation metrics."""
    X, y = split_features_target(df, target_col, feature_cols)
    should_stratify = task_type == "classification" if stratify is None else stratify
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=should_stratify,
    )

    pipeline = build_model_pipeline(model, X_train, scale_numeric=scale_numeric)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = evaluate_predictions(
        y_true=y_test,
        y_pred=predictions,
        model=pipeline,
        X_test=X_test,
        task_type=task_type,
    )

    return {
        "model": pipeline,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
    }


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics for claim severity or premium models."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_classification(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Return standard binary classification metrics for claim-risk models."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None and len(pd.Series(y_true).unique()) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    return metrics


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    task_type: TaskType,
    model: Pipeline | None = None,
    X_test: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Evaluate predictions for either regression or classification."""
    if task_type == "regression":
        return evaluate_regression(y_true, y_pred)

    y_proba = None
    if model is not None and X_test is not None and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        if probabilities.shape[1] == 2:
            y_proba = probabilities[:, 1]

    return evaluate_classification(y_true, y_pred, y_proba)


def metrics_to_frame(
    model_name: str,
    metrics: dict[str, float],
    task_type: TaskType,
) -> pd.DataFrame:
    """Convert a metrics dictionary into a tidy one-row DataFrame."""
    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                "task_type": task_type,
                **metrics,
            }
        ]
    )


def predict_claim_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return predicted claim probabilities from a fitted classifier pipeline."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError("The supplied model does not support predict_proba.")

    probabilities = model.predict_proba(X)
    if probabilities.shape[1] == 1:
        return probabilities[:, 0]

    return probabilities[:, 1]


def calculate_pure_premium(
    claim_probability: np.ndarray | pd.Series,
    predicted_severity: np.ndarray | pd.Series,
) -> np.ndarray:
    """Calculate pure premium as claim probability times predicted severity."""
    return np.asarray(claim_probability) * np.asarray(predicted_severity)


def calculate_technical_premium(
    pure_premium: np.ndarray | pd.Series,
    expense_loading: float = 0.20,
    risk_loading: float = 0.10,
    profit_margin: float = 0.10,
) -> np.ndarray:
    """Add expense, risk, and profit loads to pure premium.

    Loadings are expressed as proportions. For example, 0.20 means 20%.
    """
    total_loading = 1 + expense_loading + risk_loading + profit_margin
    return np.asarray(pure_premium) * total_loading


def build_pricing_frame(
    policy_ids: pd.Series | None,
    claim_probability: np.ndarray | pd.Series,
    predicted_severity: np.ndarray | pd.Series,
    expense_loading: float = 0.20,
    risk_loading: float = 0.10,
    profit_margin: float = 0.10,
) -> pd.DataFrame:
    """Build a report-ready risk-based premium table."""
    pure_premium = calculate_pure_premium(claim_probability, predicted_severity)
    technical_premium = calculate_technical_premium(
        pure_premium,
        expense_loading=expense_loading,
        risk_loading=risk_loading,
        profit_margin=profit_margin,
    )

    result = pd.DataFrame(
        {
            "claim_probability": np.asarray(claim_probability),
            "predicted_severity": np.asarray(predicted_severity),
            "pure_premium": pure_premium,
            "technical_premium": technical_premium,
        }
    )

    if policy_ids is not None:
        result.insert(0, "policy_id", policy_ids.to_numpy())

    return result
