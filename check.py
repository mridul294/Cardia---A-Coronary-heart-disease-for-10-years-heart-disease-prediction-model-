"""Diagnose the project setup.   Run:  python check.py

Checks library versions, required files, that the model loads, and that every page and
API route responds, then prints exactly what is wrong (if anything).
"""
import importlib
import importlib.util
import sys
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent
ok_all = True


def line(ok, text):
    global ok_all
    ok_all = ok_all and ok
    print(("  OK    " if ok else "  FAIL  ") + text)


print(f"Python {sys.version.split()[0]}  |  project folder: {BASE}\n")

print("Libraries")
from importlib import metadata

for name, dist in (("flask", "flask"), ("pandas", "pandas"), ("numpy", "numpy"),
                   ("sklearn", "scikit-learn"), ("joblib", "joblib")):
    try:
        importlib.import_module(name)
        line(True, f"{dist} {metadata.version(dist)}")
    except Exception as exc:
        line(False, f"{dist} is not installed ({exc}). Run: pip install -r requirements.txt")

print("\nFiles")
index_candidates = [BASE / "templates" / "index.html", BASE / "static" / "index.html", BASE / "index.html"]
found = [p for p in index_candidates if p.exists()]
line(bool(found), f"index.html: {found[0] if found else 'NOT FOUND. Put it in a folder named templates next to app.py'}")
if found:
    text = found[0].read_text(encoding="utf-8", errors="replace")
    if "{%" in text or "{{" in text:
        line(False, "index.html is the OLD version (contains {% %} template tags). Replace it with the new one.")
    else:
        line(True, "index.html is the new static version")
line((BASE / "app.py").exists(), "app.py")
line((BASE / "heart_model.pkl").exists(),
     "heart_model.pkl" + ("" if (BASE / "heart_model.pkl").exists() else "  <- run: python train.py"))
line((BASE / "data" / "heart.csv").exists(),
     "data/heart.csv" + ("" if (BASE / "data" / "heart.csv").exists() else "  <- needed for train.py"))

print("\nApp")
try:
    sys.path.insert(0, str(BASE))
    spec = importlib.util.spec_from_file_location("heart_app", BASE / "app.py")
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    line(True, "app.py imports and the model loads")
    client = app_module.app.test_client()
    for path in ("/", "/health", "/api/config"):
        r = client.get(path)
        line(r.status_code == 200, f"GET {path} -> {r.status_code}")
    sample = dict(age=50, male=1, sysBP=130, diaBP=85, totChol=220, BMI=26, heartRate=72, glucose=90,
                  currentSmoker=0, cigsPerDay=0, BPMeds=0, prevalentStroke=0, prevalentHyp=0, diabetes=0)
    r = client.post("/predict", json=sample)
    detail = r.get_json(silent=True) or {}
    line(r.status_code == 200, f"POST /predict -> {r.status_code}"
         + (f"  (risk {detail.get('risk_percent')}%)" if r.status_code == 200 else f"  {detail}"))
except SystemExit as exc:
    line(False, f"app.py stopped: {exc}")
except Exception as exc:
    line(False, f"app.py failed to start: {type(exc).__name__}: {exc}")
    traceback.print_exc()

print("\n" + ("Everything looks good. Start the site with:  python app.py" if ok_all
              else "Fix the FAIL lines above, then run python check.py again."))
