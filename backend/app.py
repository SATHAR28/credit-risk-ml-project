import os
import json
import sys
import traceback

import joblib
import pandas as pd
from flask import Flask, render_template, request

# ── paths (relative to project root) ────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_stacked_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.json")

# ── load model & features once at startup ────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    print(f"[INFO] Model loaded from {MODEL_PATH}")
except Exception as exc:
    print(f"[ERROR] Could not load model: {exc}")
    sys.exit(1)

try:
    with open(FEATURES_PATH, "r") as f:
        FEATURE_NAMES = json.load(f)
    print(f"[INFO] {len(FEATURE_NAMES)} features loaded from {FEATURES_PATH}")
except Exception as exc:
    print(f"[ERROR] Could not load features: {exc}")
    sys.exit(1)

# occupation columns extracted from the feature list
OCCUPATION_COLUMNS = [f for f in FEATURE_NAMES if f.startswith("occupation_")]
OCCUPATION_VALUES = [c.replace("occupation_", "") for c in OCCUPATION_COLUMNS]

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/developer")
def developer():
    return render_template("developer.html")


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template(
        "predict.html",
        occupations=OCCUPATION_VALUES,
        prediction=None,
        error=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ── collect & validate numeric inputs ────────────────────────────
        gender = request.form.get("gender", "").strip()
        days_employed = request.form.get("no_of_days_employed", "").strip()
        credit_limit = request.form.get("credit_limit_used", "").strip()
        credit_score = request.form.get("credit_score", "").strip()
        defaults_6m = request.form.get("default_in_last_6months", "").strip()
        occupation = request.form.get("occupation", "").strip()

        # check for missing fields
        if not all([gender, days_employed, credit_limit, credit_score, defaults_6m, occupation]):
            return render_template(
                "predict.html",
                occupations=OCCUPATION_VALUES,
                prediction=None,
                error="All fields are required. Please fill in every input.",
            )

        # convert to numbers
        try:
            gender_val = int(gender)
            days_employed_val = float(days_employed)
            credit_limit_val = float(credit_limit)
            credit_score_val = float(credit_score)
            defaults_6m_val = int(defaults_6m)
        except ValueError:
            return render_template(
                "predict.html",
                occupations=OCCUPATION_VALUES,
                prediction=None,
                error="Invalid numeric value. Please enter valid numbers.",
            )

        # ── build feature vector in training order ───────────────────────
        row = {feat: 0 for feat in FEATURE_NAMES}
        row["gender"] = gender_val
        row["no_of_days_employed"] = days_employed_val
        row["credit_limit_used(%)"] = credit_limit_val
        row["credit_score"] = credit_score_val
        row["default_in_last_6months"] = defaults_6m_val

        occ_col = f"occupation_{occupation}"
        if occ_col in row:
            row[occ_col] = 1

        df = pd.DataFrame([row], columns=FEATURE_NAMES)

        # ── predict ──────────────────────────────────────────────────────
        pred = model.predict(df)[0]
        result = "High Risk" if int(pred) == 1 else "Low Risk"

        return render_template(
            "predict.html",
            occupations=OCCUPATION_VALUES,
            prediction=result,
            error=None,
        )

    except Exception:
        traceback.print_exc()
        return render_template(
            "predict.html",
            occupations=OCCUPATION_VALUES,
            prediction=None,
            error="An unexpected error occurred. Please try again.",
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
