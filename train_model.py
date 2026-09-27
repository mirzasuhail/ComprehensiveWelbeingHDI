import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    print("=" * 60)
    print("           HDI INSIGHT - MODEL TRAINING SCRIPT")
    print("=" * 60)

    # 1. Load the dataset
    data_path = os.path.join("data", "hdi_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run dataset preparation.")

    print(f"\n[1] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"    Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")

    # 2. Display basic dataset inspection
    print("\n[2] Dataset Preview:")
    print(df.head(5))

    print("\n[3] Checking missing values:")
    missing_counts = df.isnull().sum()
    print(missing_counts)

    # Handle missing values if any exist
    if df.isnull().sum().sum() > 0:
        df = df.dropna().reset_index(drop=True)
        print("    Dropped missing values. Remaining rows:", len(df))
    else:
        print("    No missing values found.")

    # 3. Select Features (X) and Target (y)
    feature_cols = [
        "Life_Expectancy",
        "Expected_Years_of_Schooling",
        "Mean_Years_of_Schooling",
        "GNI_per_Capita",
    ]
    target_col = "HDI"

    X = df[feature_cols]
    y = df[target_col]

    print(f"\n[4] Features selected: {feature_cols}")
    print(f"    Target selected: '{target_col}'")

    # 4. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"    Train size: {len(X_train)} samples | Test size: {len(X_test)} samples")

    # 5. Model Training and Comparison
    print("\n[5] Training Machine Learning Models...")

    # Model A: Linear Regression (Primary baseline model)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)

    print("\n    --- Model A: Linear Regression ---")
    print(f"    Mean Absolute Error (MAE) : {lr_mae:.4f}")
    print(f"    Root Mean Squared Error   : {lr_rmse:.4f}")
    print(f"    R² Score (Accuracy fit)  : {lr_r2:.4f}")

    # Model B: Random Forest Regressor (Benchmark comparison model)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)

    print("\n    --- Model B: Random Forest Regressor ---")
    print(f"    Mean Absolute Error (MAE) : {rf_mae:.4f}")
    print(f"    Root Mean Squared Error   : {rf_rmse:.4f}")
    print(f"    R² Score (Accuracy fit)  : {rf_r2:.4f}")

    # 6. Model Selection
    # Select Linear Regression as the primary production model due to high interpretability
    selected_model_name = "Linear Regression"
    selected_model = lr_model
    selected_metrics = {
        "model_name": selected_model_name,
        "mae": round(float(lr_mae), 4),
        "rmse": round(float(lr_rmse), 4),
        "r2": round(float(lr_r2), 4),
        "dataset_samples": int(len(df)),
        "feature_names": feature_cols,
        "coefficients": {
            feature: round(float(coef), 6)
            for feature, coef in zip(feature_cols, lr_model.coef_)
        },
        "intercept": round(float(lr_model.intercept_), 6),
        "benchmark_random_forest": {
            "mae": round(float(rf_mae), 4),
            "rmse": round(float(rf_rmse), 4),
            "r2": round(float(rf_r2), 4),
        },
    }

    print(f"\n[6] Selected Model for Deployment: {selected_model_name}")

    # 7. Save Model and Metrics
    os.makedirs("model", exist_ok=True)
    model_output_path = os.path.join("model", "hdi_model.pkl")
    metrics_output_path = os.path.join("model", "metrics.json")

    joblib.dump(selected_model, model_output_path)
    print(f"    Saved model file to: '{model_output_path}'")

    with open(metrics_output_path, "w") as f:
        json.dump(selected_metrics, f, indent=4)
    print(f"    Saved model metrics to: '{metrics_output_path}'")

    print("\n" + "=" * 60)
    print("         TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
