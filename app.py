
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_model.pkl"


def find_index():
   
    for folder in (BASE_DIR / "templates", BASE_DIR / "static", BASE_DIR):
        candidate = folder / "index.html"
        if candidate.exists():
            return candidate
    raise SystemExit(
        
        f"(looked in: templates/, static/, and {BASE_DIR})."
    )


INDEX_PATH = find_index()

if not MODEL_PATH.exists():
    raise SystemExit(f"{MODEL_PATH.name} not found. Run `python train.py` first.")

bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
THRESHOLD = float(bundle["threshold"])
FEATURES = list(bundle["features"])
METRICS = bundle.get("metrics", {})

# kind: "num" = numeric input, "bin" = yes/no (0/1). required=False fields may be left blank
# and are filled in by the model's imputer, exactly like missing values in the training data.
FIELDS = [
    {"name": "age", "label": "Age (years)", "kind": "num", "min": 18, "max": 100, "required": True},
    {"name": "male", "label": "Sex", "kind": "bin", "required": True,
     "choices": [(1, "Male"), (0, "Female")]},
    {"name": "sysBP", "label": "Systolic blood pressure (mmHg)", "kind": "num", "min": 70, "max": 300, "required": True},
    {"name": "diaBP", "label": "Diastolic blood pressure (mmHg)", "kind": "num", "min": 40, "max": 160, "required": True},
    {"name": "totChol", "label": "Total cholesterol (mg/dL)", "kind": "num", "min": 80, "max": 700, "required": False},
    {"name": "BMI", "label": "BMI (kg/m²)", "kind": "num", "min": 12, "max": 60, "required": False},
    {"name": "heartRate", "label": "Resting heart rate (bpm)", "kind": "num", "min": 35, "max": 200, "required": False},
    {"name": "glucose", "label": "Glucose (mg/dL)", "kind": "num", "min": 30, "max": 600, "required": False},
    {"name": "currentSmoker", "label": "Current smoker", "kind": "bin", "required": True},
    {"name": "cigsPerDay", "label": "Cigarettes per day", "kind": "num", "min": 0, "max": 100, "required": False},
    {"name": "BPMeds", "label": "On blood pressure medication", "kind": "bin", "required": True},
    {"name": "prevalentStroke", "label": "History of stroke", "kind": "bin", "required": True},
    {"name": "prevalentHyp", "label": "Hypertensive", "kind": "bin", "required": True},
    {"name": "diabetes", "label": "Diabetic", "kind": "bin", "required": True},
]
assert {f["name"] for f in FIELDS} == set(FEATURES), "FIELDS and model features are out of sync"

app = Flask(__name__)


def parse_payload(data):
    """Validate raw input. Returns (row_dict, errors_dict)."""
    row, errors = {}, {}
    for f in FIELDS:
        name = f["name"]
        raw = data.get(name)
        if raw is None or str(raw).strip() == "":
            if f["required"]:
                errors[name] = "This field is required."
            else:
                row[name] = np.nan
            continue
        text = str(raw).strip().lower()
        if f["kind"] == "bin":
            mapping = {"1": 1, "0": 0, "yes": 1, "no": 0, "true": 1, "false": 0}
            if text in mapping:
                row[name] = mapping[text]
            else:
                errors[name] = "Must be 1/0 (or yes/no)."
            continue
        try:
            value = float(text)
        except ValueError:
            errors[name] = "Must be a number."
            continue
        if not np.isfinite(value) or not (f["min"] <= value <= f["max"]):
            errors[name] = f"Must be between {f['min']} and {f['max']}."
            continue
        row[name] = value
    # A non-smoker smokes zero cigarettes. Keeps inputs consistent with the training data.
    if row.get("currentSmoker") == 0 and "cigsPerDay" not in errors:
        cigs = row.get("cigsPerDay", np.nan)
        if np.isnan(cigs):
            row["cigsPerDay"] = 0.0
        elif cigs != 0:
            errors["cigsPerDay"] = "Must be 0 for a non-smoker."
    if "sysBP" in row and "diaBP" in row and "sysBP" not in errors and "diaBP" not in errors:
        if row["diaBP"] >= row["sysBP"]:
            errors["diaBP"] = "Diastolic must be lower than systolic."
    return row, errors


def predict_risk(row):
    frame = pd.DataFrame([row], columns=FEATURES)
    probability = float(MODEL.predict_proba(frame)[0, 1])
    flagged = probability >= THRESHOLD
    return {
        "risk_probability": round(probability, 4),
        "risk_percent": round(probability * 100, 1),
        "high_risk": bool(flagged),
        "risk_label": "Higher risk" if flagged else "Lower risk",
        "threshold_percent": round(THRESHOLD * 100, 1),
        "disclaimer": "Educational model, not clinically validated. Not medical advice.",
    }


@app.after_request
def add_headers(response):
    # Allow the page to call this API even when index.html is opened straight from disk (file://).
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
@app.get("/index.html")
def index():
    return send_file(INDEX_PATH, mimetype="text/html")


@app.get("/api/config")
def config():
    """Everything the page needs to build the form and show model stats."""
    fields = []
    for f in FIELDS:
        item = {k: v for k, v in f.items() if k != "choices"}
        if "choices" in f:
            item["choices"] = [{"value": v, "label": label} for v, label in f["choices"]]
        fields.append(item)
    test = METRICS.get("test", {})
    return jsonify(
        fields=fields,
        threshold=THRESHOLD,
        metrics={
            "model": METRICS.get("model"),
            "rows": METRICS.get("rows"),
            "test": {
                "roc_auc": test.get("roc_auc"),
                "recall": test.get("recall"),
                "precision": test.get("precision"),
            } if test else None,
        },
    )


@app.get("/health")
def health():
    return jsonify(status="ok", model=METRICS.get("model"), threshold=THRESHOLD)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if data is None:
        data = request.form.to_dict()
    if not isinstance(data, dict):
        return jsonify(error="Send a JSON object or form data."), 400
    row, errors = parse_payload(data)
    if errors:
        return jsonify(error="Invalid input.", fields=errors), 400
    return jsonify(predict_risk(row))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
