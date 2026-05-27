import pandas as pd

from src.modeling import (
    build_linear_model,
    build_pricing_frame,
    build_random_forest_model,
    calculate_pure_premium,
    calculate_technical_premium,
    evaluate_classification,
    metrics_to_frame,
    split_features_target,
    train_model,
)


def test_split_features_target_drops_missing_target_rows() -> None:
    df = pd.DataFrame(
        {
            "Province": ["A", "B", "C"],
            "TotalPremium": [100.0, 200.0, 300.0],
            "TotalClaims": [10.0, None, 30.0],
        }
    )

    X, y = split_features_target(df, target_col="TotalClaims")

    assert len(X) == 2
    assert len(y) == 2
    assert "TotalClaims" not in X.columns


def test_train_linear_regression_handles_missing_and_categorical_features() -> None:
    df = pd.DataFrame(
        {
            "Province": ["A", "A", "B", "B", "C", None, "C", "A"],
            "VehicleType": ["SUV", "Sedan", "SUV", "Truck", "Sedan", "SUV", None, "Truck"],
            "TotalPremium": [100, 120, 140, 160, 180, 200, 220, 240],
            "TotalClaims": [10, 12, 15, 18, 20, 21, 23, 25],
        }
    )

    result = train_model(
        df,
        target_col="TotalClaims",
        model=build_linear_model("regression"),
        feature_cols=["Province", "VehicleType", "TotalPremium"],
        task_type="regression",
        test_size=0.25,
    )

    assert "model" in result
    assert "mae" in result["metrics"]
    assert "rmse" in result["metrics"]
    assert "r2" in result["metrics"]


def test_train_random_forest_classifier_returns_classification_metrics() -> None:
    df = pd.DataFrame(
        {
            "Province": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "TotalPremium": [100, 110, 120, 130, 300, 320, 340, 360],
            "HasClaim": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    result = train_model(
        df,
        target_col="HasClaim",
        model=build_random_forest_model(
            "classification",
            n_estimators=10,
            max_depth=2,
        ),
        feature_cols=["Province", "TotalPremium"],
        task_type="classification",
        test_size=0.25,
    )

    assert "accuracy" in result["metrics"]
    assert "precision" in result["metrics"]
    assert "recall" in result["metrics"]
    assert "f1" in result["metrics"]


def test_evaluate_classification_handles_zero_division() -> None:
    metrics = evaluate_classification(
        y_true=pd.Series([0, 1, 1]),
        y_pred=pd.Series([0, 0, 0]),
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_metrics_to_frame_returns_tidy_row() -> None:
    frame = metrics_to_frame(
        model_name="Linear Regression",
        metrics={"mae": 1.5, "rmse": 2.0, "r2": 0.8},
        task_type="regression",
    )

    assert frame.loc[0, "model_name"] == "Linear Regression"
    assert frame.loc[0, "task_type"] == "regression"
    assert frame.loc[0, "mae"] == 1.5


def test_premium_helpers_calculate_expected_loss_and_loaded_premium() -> None:
    pure = calculate_pure_premium([0.1, 0.2], [1000, 2000])
    technical = calculate_technical_premium(
        pure,
        expense_loading=0.2,
        risk_loading=0.1,
        profit_margin=0.1,
    )

    assert pure.tolist() == [100.0, 400.0]
    assert technical.tolist() == [140.0, 560.0]


def test_build_pricing_frame_returns_report_ready_columns() -> None:
    frame = build_pricing_frame(
        policy_ids=pd.Series(["A", "B"]),
        claim_probability=[0.1, 0.2],
        predicted_severity=[1000, 2000],
    )

    assert list(frame.columns) == [
        "policy_id",
        "claim_probability",
        "predicted_severity",
        "pure_premium",
        "technical_premium",
    ]
