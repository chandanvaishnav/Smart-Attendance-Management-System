from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from attendance_utils import build_prediction_features, load_attendance_data

MAX_TRAIN_ROWS = 20000
MAX_VALIDATION_ROWS = 5000
MAX_TEST_ROWS = 5000


def get_model_registry() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def train_and_evaluate_models() -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.Series]:
    df = load_attendance_data().sort_values("Date").reset_index(drop=True)
    X, y = build_prediction_features(df)
    if y is None:
        raise ValueError("Training data must include the Present target column.")

    unique_dates = df["Date"].drop_duplicates().sort_values().to_numpy()
    validation_start = unique_dates[int(len(unique_dates) * 0.7)]
    test_start = unique_dates[int(len(unique_dates) * 0.85)]
    training_mask = df["Date"] < validation_start
    validation_mask = (df["Date"] >= validation_start) & (df["Date"] < test_start)
    test_mask = df["Date"] >= test_start

    X_train = X.loc[training_mask]
    y_train = y.loc[training_mask]
    X_validation = X.loc[validation_mask]
    y_validation = y.loc[validation_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    if len(X_train) > MAX_TRAIN_ROWS:
        X_train = X_train.sample(n=MAX_TRAIN_ROWS, random_state=42)
        y_train = y_train.loc[X_train.index]
    if len(X_validation) > MAX_VALIDATION_ROWS:
        X_validation = X_validation.sample(n=MAX_VALIDATION_ROWS, random_state=43)
        y_validation = y_validation.loc[X_validation.index]
    if len(X_test) > MAX_TEST_ROWS:
        X_test = X_test.sample(n=MAX_TEST_ROWS, random_state=44)
        y_test = y_test.loc[X_test.index]

    results = []
    trained_models: dict[str, object] = {}

    for name, model in get_model_registry().items():
        trained_model = model.fit(X_train, y_train)
        trained_models[name] = trained_model

        validation_predictions = trained_model.predict(X_validation)
        validation_mae = mean_absolute_error(y_validation, validation_predictions)
        test_predictions = trained_model.predict(X_test)
        mae = mean_absolute_error(y_test, test_predictions)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        r2 = r2_score(y_test, test_predictions)

        results.append({
            "Model": name,
            "Validation MAE": float(validation_mae),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2 Score": float(r2),
        })

    results_df = pd.DataFrame(results).sort_values("Validation MAE").reset_index(drop=True)
    return results_df, trained_models, X_test, y_test


def main() -> None:
    results_df, trained_models, y_test, X_test = train_and_evaluate_models()

    print("\n========================================")
    print(" MACHINE LEARNING MODEL COMPARISON")
    print("========================================")
    print("Validation: chronological train/validation/test; predictors use only school and calendar fields.")
    print("Same-day attendance counts are excluded because Present = Enrolled - Absent - Released.")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    print("\nBest Model:", best_model_name)
    print("Model count:", len(trained_models))
    print("\nModel comparison completed successfully.")


if __name__ == "__main__":
    main()