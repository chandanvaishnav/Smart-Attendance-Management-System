from __future__ import annotations

from ML_models import train_and_evaluate_models


def main() -> None:
    results, trained_models, X_test, y_test = train_and_evaluate_models()
    best_result = results.iloc[0]
    model_name = best_result["Model"]
    model = trained_models[model_name]
    sample_predicted = float(model.predict(X_test.iloc[:1])[0])
    sample_actual = float(y_test.iloc[0])
    sample_error = abs(sample_actual - sample_predicted)

    print("\n===================================")
    print(" ATTENDANCE PREDICTION SYSTEM")
    print("===================================")
    print("Final model:", model_name)
    print("Chronological holdout MAE:", round(float(best_result["MAE"]), 2))
    print("\n----- SAMPLE PREDICTION -----")
    print("Actual Present Students:", sample_actual)
    print("Predicted Present Students:", round(sample_predicted, 2))
    print("Absolute Error:", round(sample_error, 2))
    print("\nPrediction completed successfully.")


if __name__ == "__main__":
    main()