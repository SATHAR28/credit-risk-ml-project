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

# configurable decision thresholds for class-1 probability (default risk)
LOW_RISK_THRESHOLD = float(os.getenv("LOW_RISK_THRESHOLD", "40"))
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "60"))

if not (0 <= LOW_RISK_THRESHOLD <= HIGH_RISK_THRESHOLD <= 100):
    print("[WARN] Invalid threshold config; falling back to LOW=40, HIGH=60")
    LOW_RISK_THRESHOLD = 40.0
    HIGH_RISK_THRESHOLD = 60.0

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


def evaluate_model_health(loaded_model):
    """Run quick probe predictions to detect obviously degenerate artifacts."""
    try:
        if not OCCUPATION_VALUES:
            return {
                "healthy": False,
                "message": "No occupation features found in feature schema.",
            }

        probe_occupation = OCCUPATION_VALUES[0]

        def make_probe(gender, days, limit_used, score, defaults_6m):
            row = {feat: 0 for feat in FEATURE_NAMES}
            row["gender"] = gender
            row["no_of_days_employed"] = days
            row["credit_limit_used(%)"] = limit_used
            row["credit_score"] = score
            row["default_in_last_6months"] = defaults_6m
            row[f"occupation_{probe_occupation}"] = 1
            return pd.DataFrame([row], columns=FEATURE_NAMES)

        probes = [
            make_probe(1, 20, 98, 320, 9),
            make_probe(0, 7000, 4, 830, 0),
            make_probe(1, 1200, 55, 640, 1),
        ]

        preds = [int(loaded_model.predict(df)[0]) for df in probes]
        probas = []
        if hasattr(loaded_model, "predict_proba"):
            probas = [float(loaded_model.predict_proba(df)[0][1]) for df in probes]

        all_same_pred = len(set(preds)) == 1
        all_same_proba = len(set(round(p, 6) for p in probas)) == 1 if probas else False

        if all_same_pred and (not probas or all_same_proba):
            return {
                "healthy": False,
                "message": (
                    "Model appears degenerate (constant predictions on probe inputs). "
                    "Retrain and replace models/credit_risk_stacked_model.pkl."
                ),
            }

        return {"healthy": True, "message": None}
    except Exception as exc:
        return {"healthy": False, "message": f"Model health check failed: {exc}"}


MODEL_HEALTH = evaluate_model_health(model)
if MODEL_HEALTH["healthy"]:
    print("[INFO] Model health check passed.")
else:
    print(f"[WARN] {MODEL_HEALTH['message']}")

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)


@app.context_processor
def inject_decision_thresholds():
    return {
        "low_risk_threshold": LOW_RISK_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "model_warning": None if MODEL_HEALTH["healthy"] else MODEL_HEALTH["message"],
    }


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
        probability=None,
        error=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not MODEL_HEALTH["healthy"]:
            return render_template(
                "predict.html",
                occupations=OCCUPATION_VALUES,
                prediction=None,
                probability=None,
                error=(
                    "Prediction is temporarily disabled because the current model artifact is invalid. "
                    "Please retrain and deploy a new model file."
                ),
            )

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
                probability=None,
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
                probability=None,
                error="Invalid numeric value. Please enter valid numbers.",
            )

        # ── strict range validation ──────────────────────────────────────
        validation_errors = []
        if credit_score_val < 300 or credit_score_val > 850:
            validation_errors.append("Credit Score must be between 300 and 850.")
        if credit_limit_val < 0 or credit_limit_val > 100:
            validation_errors.append("Credit Limit Used (%) must be between 0 and 100.")
        if defaults_6m_val < 0 or defaults_6m_val > 10:
            validation_errors.append("Defaults in Last 6 Months must be between 0 and 10.")
        if days_employed_val < 0:
            validation_errors.append("Days Employed must be 0 or greater.")
        if gender_val not in (0, 1):
            validation_errors.append("Gender must be 0 (Female) or 1 (Male).")
        if occupation not in OCCUPATION_VALUES:
            validation_errors.append(f"Invalid occupation: {occupation}.")

        if validation_errors:
            return render_template(
                "predict.html",
                occupations=OCCUPATION_VALUES,
                prediction=None,
                probability=None,
                error=" ".join(validation_errors),
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

        # ── debug logging ────────────────────────────────────────────────
        print("\n[DEBUG] Input to model:")
        print(df.to_string())
        print()

        # ── probability of default ───────────────────────────────────────
        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            # class 1 = default probability
            probability = round(float(proba[1]) * 100, 2)

        # ── base class prediction fallback ───────────────────────────────
        pred = int(model.predict(df)[0])

        # ── post-processing policy for stability ─────────────────────────
        # Middle-range probability is treated as uncertain and flagged.
        if probability is None:
            result = "High Risk" if pred == 1 else "Low Risk"
        elif probability >= HIGH_RISK_THRESHOLD:
            result = "High Risk"
        elif probability <= LOW_RISK_THRESHOLD:
            result = "Low Risk"
        else:
            result = "Manual Review"

        print(
            f"[INFO] Decision policy: prob={probability}, "
            f"low={LOW_RISK_THRESHOLD}, high={HIGH_RISK_THRESHOLD}, result={result}"
        )

        return render_template(
            "predict.html",
            occupations=OCCUPATION_VALUES,
            prediction=result,
            probability=probability,
            error=None,
        )

    except Exception:
        traceback.print_exc()
        return render_template(
            "predict.html",
            occupations=OCCUPATION_VALUES,
            prediction=None,
            probability=None,
            error="An unexpected error occurred. Please try again.",
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
