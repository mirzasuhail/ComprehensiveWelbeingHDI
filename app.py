import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# File paths
MODEL_PATH = os.path.join("model", "hdi_model.pkl")
METRICS_PATH = os.path.join("model", "metrics.json")

# Global variables for model and metrics
model = None
metrics = {
    "model_name": "Linear Regression",
    "r2": 0.9807,
    "rmse": 0.0202,
    "mae": 0.0149,
    "dataset_samples": 204
}

def load_ml_resources():
    global model, metrics
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"[SUCCESS] Loaded trained model from {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            model = None
    else:
        print(f"[WARNING] Model file not found at {MODEL_PATH}")

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
            print(f"[SUCCESS] Loaded model metrics from {METRICS_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to load metrics: {e}")

# Load model and metrics on app startup
load_ml_resources()

def get_hdi_tier(hdi_value):
    """
    Standard UNDP Human Development Index Tier Classification.
    Note: This is presentation layer categorization, distinct from the raw ML regression model output.
    """
    if hdi_value >= 0.800:
        return "Very High Human Development", "very-high"
    elif hdi_value >= 0.700:
        return "High Human Development", "high"
    elif hdi_value >= 0.550:
        return "Medium Human Development", "medium"
    else:
        return "Low Human Development", "low"

@app.route("/", methods=["GET"])
def index():
    """Render main application page with initial metrics."""
    return render_template("index.html", metrics=metrics)

@app.route("/predict", methods=["POST"])
def predict():
    """Accept 4 indicator values, validate inputs, run ML prediction, return results."""
    global model
    if model is None:
        # Retry loading if missing initially
        load_ml_resources()
        if model is None:
            return jsonify({
                "success": False,
                "error": "Trained ML model is not available. Please run train_model.py first."
            }), 500

    try:
        # Extract input data from JSON or Form submission
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        raw_life = data.get("life_expectancy")
        raw_exp_sch = data.get("expected_schooling")
        raw_mean_sch = data.get("mean_schooling")
        raw_gni = data.get("gni_per_capita")

        # 1. Presence check
        if any(v is None or str(v).strip() == "" for v in [raw_life, raw_exp_sch, raw_mean_sch, raw_gni]):
            return jsonify({
                "success": False,
                "error": "All 4 socioeconomic indicators are required."
            }), 400

        # 2. Type conversion check
        try:
            life_expectancy = float(raw_life)
            expected_schooling = float(raw_exp_sch)
            mean_schooling = float(raw_mean_sch)
            gni_per_capita = float(raw_gni)
        except ValueError:
            return jsonify({
                "success": False,
                "error": "Invalid input format. All indicator values must be numeric numbers."
            }), 400

        # 3. Numeric range validation
        if not (20.0 <= life_expectancy <= 110.0):
            return jsonify({
                "success": False,
                "error": "Life expectancy must be between 20.0 and 110.0 years."
            }), 400

        if not (0.0 <= expected_schooling <= 30.0):
            return jsonify({
                "success": False,
                "error": "Expected years of schooling must be between 0.0 and 30.0 years."
            }), 400

        if not (0.0 <= mean_schooling <= 30.0):
            return jsonify({
                "success": False,
                "error": "Mean years of schooling must be between 0.0 and 30.0 years."
            }), 400

        if not (100.0 <= gni_per_capita <= 250000.0):
            return jsonify({
                "success": False,
                "error": "GNI per capita must be between $100 and $250,000 PPP."
            }), 400

        # 4. Logical indicator constraint validation
        if mean_schooling > expected_schooling:
            return jsonify({
                "success": False,
                "error": f"Logical inconsistency: Mean schooling ({mean_schooling} yrs) cannot exceed Expected schooling ({expected_schooling} yrs)."
            }), 400

        # 5. ML Prediction
        input_df = pd.DataFrame([{
            "Life_Expectancy": life_expectancy,
            "Expected_Years_of_Schooling": expected_schooling,
            "Mean_Years_of_Schooling": mean_schooling,
            "GNI_per_Capita": gni_per_capita
        }])
        raw_prediction = float(model.predict(input_df)[0])

        # Clip predicted HDI to valid 0.000 to 1.000 range
        hdi_prediction = round(max(0.000, min(1.000, raw_prediction)), 3)

        tier_label, tier_code = get_hdi_tier(hdi_prediction)

        return jsonify({
            "success": True,
            "hdi": hdi_prediction,
            "hdi_formatted": f"{hdi_prediction:.3f}",
            "tier": tier_label,
            "tier_code": tier_code,
            "inputs": {
                "life_expectancy": life_expectancy,
                "expected_schooling": expected_schooling,
                "mean_schooling": mean_schooling,
                "gni_per_capita": f"{int(gni_per_capita):,}"
            },
            "metrics": {
                "model_name": metrics.get("model_name", "Linear Regression"),
                "r2": metrics.get("r2", 0.9807),
                "rmse": metrics.get("rmse", 0.0202),
                "mae": metrics.get("mae", 0.0149)
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"An unexpected error occurred during prediction: {str(e)}"
        }), 500

if __name__ == "__main__":
    print("Starting HDI Insight Flask Application on http://0.0.0.0:5000 (Local & Network accessible)...")
    app.run(host="0.0.0.0", port=5000, debug=True)
