import os
import json
import sys
import traceback
from io import BytesIO
from textwrap import wrap
from datetime import datetime, timedelta

import joblib
import pandas as pd
from flask import Flask, make_response, render_template, request
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

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


def _normalize_gender_label(gender_value):
    return "Male" if str(gender_value) == "1" else "Female"


def _generate_guidance(verdict):
    if verdict == "Low Risk":
        return {
            "title": "Pre-Approved: Proceed to Next Lending Step",
            "next_visit": (datetime.now() + timedelta(days=2)).strftime("%d %b %Y"),
            "summary": "The applicant has cleared automated risk screening and may proceed to documentation and underwriting review.",
            "actions": [
                "Proceed with KYC and identity verification.",
                "Collect latest salary proof or income proof.",
                "Run bureau refresh before final sanction.",
                "Schedule loan counseling and agreement signing.",
            ],
        }

    if verdict == "High Risk":
        return {
            "title": "Risk Advisory: Hold Disbursement",
            "next_visit": (datetime.now() + timedelta(days=30)).strftime("%d %b %Y"),
            "summary": "The applicant is currently high-risk for default based on model probability and should not be fast-tracked.",
            "actions": [
                "Pause loan processing and move case to senior risk officer.",
                "Request updated income continuity proof and bank statements.",
                "Ask borrower to reduce utilization and clear overdue dues.",
                "Re-evaluate after improvement window and new bureau pull.",
            ],
        }

    return {
        "title": "Manual Review Required",
        "next_visit": (datetime.now() + timedelta(days=14)).strftime("%d %b %Y"),
        "summary": "Risk is in the uncertain band and the case requires manual underwriting judgement.",
        "actions": [
            "Escalate for manual underwriting and document verification.",
            "Collect additional repayment capacity evidence.",
            "Validate employer continuity and debt obligations.",
            "Finalize decision after committee review.",
        ],
    }


def _draw_wrapped_text(pdf, text, x, y, max_width, font_name="Helvetica", font_size=10, line_gap=14):
    pdf.setFont(font_name, font_size)
    words = text.split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if pdf.stringWidth(candidate, font_name, font_size) <= max_width:
            line = candidate
        else:
            pdf.drawString(x, y, line)
            y -= line_gap
            line = word
    if line:
        pdf.drawString(x, y, line)
        y -= line_gap
    return y


def _build_decision_letter_pdf(form_data, prediction, probability):
    guidance = _generate_guidance(prediction)
    generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")
    probability_text = f"{probability:.2f}%" if probability is not None else "N/A"

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4
    margin = 48
    y = page_height - margin
    max_width = page_width - (2 * margin)

    pdf.setTitle("CreditGuard Decision Letter")

    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawString(margin, y, "CreditGuard AI - Borrower Decision Letter")
    y -= 24

    pdf.setFont("Helvetica", 10)
    pdf.drawString(margin, y, f"Generated on: {generated_at}")
    y -= 22

    pdf.setFillColorRGB(0.93, 0.95, 0.99)
    pdf.roundRect(margin, y - 52, max_width, 52, 8, fill=1, stroke=0)
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(margin + 12, y - 18, f"Model Verdict: {prediction}")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(margin + 12, y - 34, f"Default Probability: {probability_text}")
    y -= 70

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(margin, y, "Borrower Snapshot")
    y -= 18

    snapshot_rows = [
        ("Gender", _normalize_gender_label(form_data.get("gender", ""))),
        ("Occupation", form_data.get("occupation", "N/A")),
        ("Days Employed", form_data.get("no_of_days_employed", "N/A")),
        ("Credit Score", form_data.get("credit_score", "N/A")),
        ("Credit Utilization (%)", form_data.get("credit_limit_used", "N/A")),
        ("Defaults in Last 6 Months", form_data.get("default_in_last_6months", "N/A")),
    ]

    for label, value in snapshot_rows:
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(margin, y, f"{label}:")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(margin + 145, y, str(value))
        y -= 15

    y -= 8
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(margin, y, guidance["title"])
    y -= 16

    y = _draw_wrapped_text(pdf, guidance["summary"], margin, y, max_width, font_name="Helvetica", font_size=10)
    y -= 2
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(margin, y, f"Recommended Next Visit: {guidance['next_visit']}")
    y -= 22

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(margin, y, "Required Actions")
    y -= 16

    for action in guidance["actions"]:
        y = _draw_wrapped_text(pdf, f"- {action}", margin + 4, y, max_width - 4, font_name="Helvetica", font_size=10)
        y -= 2

    y -= 20
    pdf.setFont("Helvetica", 10)
    pdf.line(margin, y, margin + 210, y)
    pdf.line(page_width - margin - 210, y, page_width - margin, y)
    y -= 12
    pdf.drawString(margin, y, "Loan Officer Signature")
    pdf.drawString(page_width - margin - 210, y, "Borrower Signature")

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

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
        form_data={},
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
                form_data=request.form.to_dict(),
            )

        # ── collect & validate numeric inputs ────────────────────────────
        gender = request.form.get("gender", "").strip()
        days_employed = request.form.get("no_of_days_employed", "").strip()
        credit_limit = request.form.get("credit_limit_used", "").strip()
        credit_score = request.form.get("credit_score", "").strip()
        defaults_6m = request.form.get("default_in_last_6months", "").strip()
        occupation = request.form.get("occupation", "").strip()
        form_data = {
            "gender": gender,
            "no_of_days_employed": days_employed,
            "credit_limit_used": credit_limit,
            "credit_score": credit_score,
            "default_in_last_6months": defaults_6m,
            "occupation": occupation,
        }

        # check for missing fields
        if not all([gender, days_employed, credit_limit, credit_score, defaults_6m, occupation]):
            return render_template(
                "predict.html",
                occupations=OCCUPATION_VALUES,
                prediction=None,
                probability=None,
                error="All fields are required. Please fill in every input.",
                form_data=form_data,
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
                form_data=form_data,
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
                form_data=form_data,
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
            form_data=form_data,
        )

    except Exception:
        traceback.print_exc()
        return render_template(
            "predict.html",
            occupations=OCCUPATION_VALUES,
            prediction=None,
            probability=None,
            error="An unexpected error occurred. Please try again.",
            form_data=request.form.to_dict(),
        )


@app.route("/download-letter", methods=["POST"])
def download_letter():
    prediction = request.form.get("prediction", "").strip()
    probability_raw = request.form.get("probability", "").strip()

    if prediction not in {"Low Risk", "High Risk", "Manual Review"}:
        return "Invalid prediction payload.", 400

    probability = None
    if probability_raw:
        try:
            probability = float(probability_raw)
        except ValueError:
            probability = None

    form_data = {
        "gender": request.form.get("gender", "").strip(),
        "no_of_days_employed": request.form.get("no_of_days_employed", "").strip(),
        "credit_limit_used": request.form.get("credit_limit_used", "").strip(),
        "credit_score": request.form.get("credit_score", "").strip(),
        "default_in_last_6months": request.form.get("default_in_last_6months", "").strip(),
        "occupation": request.form.get("occupation", "").strip(),
    }

    letter_pdf = _build_decision_letter_pdf(form_data, prediction, probability)
    response = make_response(letter_pdf)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"creditguard_decision_{prediction.lower().replace(' ', '_')}_{timestamp}.pdf"
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


if __name__ == "__main__":
    app.run(debug=True, port=5000)
