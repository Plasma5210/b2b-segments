# model_utils.py
import re
import numpy as np
import pandas as pd
import joblib


MODEL_PATH = "segment_model_final.pkl"


def load_model(path=MODEL_PATH):
    bundle = joblib.load(path)
    return bundle


def sanitize(name):
    name = re.sub(r'[\[\]\{\}":,\.\(\)/\\\-\–\—]', '', str(name))
    name = re.sub(r'\s+', '_', name.strip())
    return name or "feature"


def safe_rename(cols):
    seen, result = {}, []
    for c in cols:
        s = sanitize(c)
        if s in seen:
            seen[s] += 1
            result.append(f"{s}__{seen[s]}")
        else:
            seen[s] = 0
            result.append(s)
    return result


def predict(bundle, company: dict) -> dict:
    model = bundle["model"]
    class_order = bundle["class_order"]
    feature_names = bundle["feature_names"]
    san_to_orig = bundle["san_to_orig"]

    row = {}
    for c in feature_names:
        orig = san_to_orig[c]
        val = company.get(orig, 0)
        row[c] = val

    row_df = pd.DataFrame([row])
    for c in row_df.columns:
        conv = pd.to_numeric(row_df[c], errors="coerce")
        if conv.isna().all():
            row_df[c] = 0.0
        else:
            row_df[c] = conv.fillna(0).astype("float64")

    proba = model.predict_proba(row_df)[0]
    pred = class_order[int(proba.argmax())]
    sorted_p = np.sort(proba)[::-1]

    return {
        "segment": pred,
        "confidence": round(float(sorted_p[0]), 3),
        "is_borderline": bool((sorted_p[0] - sorted_p[1]) < 0.15),
        "probabilities": {
            class_order[i]: round(float(p), 4) for i, p in enumerate(proba)
        },
    }