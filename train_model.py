
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_PATH = BASE_DIR / "heart_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

TARGET = "TenYearCHD"
NUMERIC = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
BINARY = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]
FEATURES = NUMERIC + BINARY  # 'education' is left out: it is not a clinical risk factor
RANDOM_STATE = 42

# Physiologically plausible ranges. Values outside are treated as data errors (missing).
PLAUSIBLE = {
    "age": (18, 100), "cigsPerDay": (0, 100), "totChol": (80, 700),
    "sysBP": (70, 300), "diaBP": (40, 160), "BMI": (12, 60),
    "heartRate": (35, 200), "glucose": (30, 600),
}


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found. Put the Framingham CSV at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, na_values=["NA", "?", ""])
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"heart.csv is missing expected Framingham columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates().dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    for col, (lo, hi) in PLAUSIBLE.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def make_pipeline(classifier, scale: bool) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC),
        ("binary", SimpleImputer(strategy="most_frequent"), BINARY),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def bootstrap_auc_ci(y_true, proba, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], proba[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    print(f"Rows: {len(df)} | Positive rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Probabilities are kept uncalibrated-by-weights (no class_weight) so that the output
    # is a real risk estimate; the imbalance is handled by choosing the threshold below.
    candidates = {
        "logistic_regression": make_pipeline(
            LogisticRegression(C=0.1, max_iter=5000), scale=True),
        "random_forest": make_pipeline(
            RandomForestClassifier(n_estimators=500, min_samples_leaf=10,
                                   random_state=RANDOM_STATE, n_jobs=-1), scale=False),
        "hist_gradient_boosting": make_pipeline(
            HistGradientBoostingClassifier(learning_rate=0.03, max_depth=3, max_iter=150,
                                           l2_regularization=1.0,
                                           random_state=RANDOM_STATE), scale=False),
    }

    # --- 2. model selection: repeated CV on the training split only ---
    cv_repeated = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    cv_scores = {}
    print("\nCross-validated ROC-AUC (training split only):")
    for name, pipe in candidates.items():
        s = cross_val_score(pipe, X_train, y_train, cv=cv_repeated, scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = (float(s.mean()), float(s.std()))
        print(f"  {name:24s} {s.mean():.3f} +/- {s.std():.3f}")
    best_name = max(cv_scores, key=lambda k: cv_scores[k][0])
    model = candidates[best_name]
    print(f"\nSelected model: {best_name}")

    # --- 3. threshold from out-of-fold training predictions (Youden's J) ---
    # J = sensitivity + specificity - 1 balances catching sick patients against false alarms.
    cv_single = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(model, X_train, y_train, cv=cv_single, method="predict_proba")[:, 1]
    grid = np.linspace(0.05, 0.60, 111)
    youden = []
    for t in grid:
        p = oof >= t
        sens = recall_score(y_train, p)
        spec = recall_score(1 - y_train, 1 - p.astype(int))
        youden.append(sens + spec - 1)
    threshold = float(grid[int(np.argmax(youden))])
    print(f"Decision threshold (max Youden's J on out-of-fold data): {threshold:.3f}")

    # --- 4. single evaluation on the untouched test set ---
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_test, proba)
    lo, hi = bootstrap_auc_ci(y_test, proba)
    majority_acc = 1 - y_test.mean()

    print("\n=== Held-out test results ===")
    print(f"ROC-AUC        : {auc:.3f}  (95% CI {lo:.3f} - {hi:.3f})")
    print(f"PR-AUC         : {average_precision_score(y_test, proba):.3f}  "
          f"(random baseline = {y_test.mean():.3f})")
    print(f"Accuracy       : {accuracy_score(y_test, pred):.3f}  "
          f"(always predicting 'no CHD' = {majority_acc:.3f})")
    print(f"Recall (sens.) : {recall_score(y_test, pred):.3f}")
    print(f"Precision      : {precision_score(y_test, pred, zero_division=0):.3f}")
    print("\nConfusion matrix [rows = actual 0/1, cols = predicted 0/1]:\n",
          confusion_matrix(y_test, pred))
    print("\nClassification report:\n",
          classification_report(y_test, pred, zero_division=0))

    metrics = {
        "model": best_name,
        "rows": int(len(df)),
        "positive_rate": float(y.mean()),
        "cv_roc_auc": {k: {"mean": v[0], "std": v[1]} for k, v in cv_scores.items()},
        "threshold": threshold,
        "test": {
            "roc_auc": float(auc), "roc_auc_ci95": [lo, hi],
            "pr_auc": float(average_precision_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "majority_class_accuracy": float(majority_acc),
            "recall": float(recall_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        },
    }

    # --- 5. refit on all data for the deployable model ---
    model.fit(X, y)
    joblib.dump(
        {"model": model, "threshold": threshold, "features": FEATURES, "metrics": metrics},
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model   -> {MODEL_PATH.name}")
    print(f"Saved metrics -> {METRICS_PATH.name}")
    print("Educational model, not clinically validated.")


if __name__ == "__main__":
    main()
