# src/ml/predict.py
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------
# PROJECT PATHS (relative to file -> project root)
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[2]        # project root
BASE_MODELS_DIR = BASE_DIR / "models"
DATA_CSV = BASE_DIR / "data" / "regional_agro_hybrid_5000.csv"

SCALER_PKL = BASE_MODELS_DIR / "scaler.pkl"
PCA_PKL = BASE_MODELS_DIR / "pca.pkl"
KMEANS_PKL = BASE_MODELS_DIR / "kmeans.pkl"
LABEL_MAP_NPY = BASE_MODELS_DIR / "label_map.npy"           # optional (dict saved as .npy)
LABEL_JSON = BASE_MODELS_DIR / "cluster_label_map.json"     # friendly labels JSON (expected)

# -------------------------
# UTIL: load friendly label JSON (once)
# -------------------------
def load_label_json_once():
    """Load friendly labels JSON. Returns dict with int keys when possible."""
    if not LABEL_JSON.exists():
        return {}
    try:
        with LABEL_JSON.open("r", encoding="utf-8-sig") as f:
            parsed = json.load(f)
        out = {}
        for k, v in parsed.items():
            try:
                out[int(k)] = v
            except Exception:
                out[str(k).strip()] = v
        return out
    except Exception:
        return {}

_LABEL_JSON_CACHE = load_label_json_once()

# -------------------------
# UTIL: feature names from CSV
# -------------------------
def get_feature_names():
    """Return numeric feature order from the CSV. Raises helpful errors if missing."""
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Data CSV not found at: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns detected in dataset CSV.")
    return numeric_cols

# -------------------------
# Load model artifacts
# -------------------------
def load_models():
    missing = [str(p) for p in (SCALER_PKL, PCA_PKL, KMEANS_PKL) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))

    try:
        scaler = joblib.load(SCALER_PKL)
        pca = joblib.load(PCA_PKL)
        kmeans = joblib.load(KMEANS_PKL)
    except Exception as e:
        raise RuntimeError(f"Failed to load model files: {e}")

    label_map_npy = None
    if LABEL_MAP_NPY.exists():
        try:
            label_map_npy = np.load(LABEL_MAP_NPY, allow_pickle=True).item()
        except Exception:
            label_map_npy = None

    friendly_labels_json = _LABEL_JSON_CACHE or None

    return scaler, pca, kmeans, label_map_npy, friendly_labels_json

# -------------------------
# Core prediction
# -------------------------
def predict_from_values(values):
    """
    values: list of numeric values in the exact order returned by get_feature_names()
    returns: dict with pred_cluster, mapped_label, friendly_label, pc_coords
    """
    scaler, pca, kmeans, label_map_npy, friendly_labels_json = load_models()

    feat_names = get_feature_names()
    if len(values) != len(feat_names):
        raise ValueError(f"Expected {len(feat_names)} numeric values, got {len(values)}")

    arr_df = pd.DataFrame([values], columns=feat_names)

    arr_scaled = scaler.transform(arr_df)
    arr_pca = pca.transform(arr_scaled)

    pred_cluster = int(kmeans.predict(arr_pca)[0])

    # --- Friendly label (JSON) ---
    friendly_label = None
    if pred_cluster in _LABEL_JSON_CACHE:
        friendly_label = _LABEL_JSON_CACHE[pred_cluster]
    elif friendly_labels_json is not None:
        friendly_label = friendly_labels_json.get(
            str(pred_cluster),
            friendly_labels_json.get(pred_cluster, None)
        )

    # -------------------------
    # FIXED MAPPED LABEL BLOCK
    # -------------------------
    mapped_label = None

    # 1) Try label_map.npy first (if exists)
    if label_map_npy is not None:
        mapped_label = label_map_npy.get(pred_cluster)
        if mapped_label is None:
            mapped_label = label_map_npy.get(str(pred_cluster))

    # 2) fallback: JSON friendly labels
    if mapped_label is None and (friendly_labels_json is not None):
        mapped_label = friendly_labels_json.get(pred_cluster)
        if mapped_label is None:
            mapped_label = friendly_labels_json.get(str(pred_cluster))

    # 3) final fallback
    if mapped_label is None:
        mapped_label = friendly_label

    return {
        "pred_cluster": pred_cluster,
        "mapped_label": mapped_label,
        "friendly_label": friendly_label,
        "pc_coords": arr_pca.flatten().tolist()
    }

# -------------------------
# CLI interface / main
# -------------------------
def main():
    try:
        feat_names = get_feature_names()
    except Exception as e:
        print("ERROR reading dataset/features:", e)
        return

    print("Feature order (use this order when entering values):")
    for i, f in enumerate(feat_names):
        print(f"{i+1}. {f}")
    print()

    if len(sys.argv) > 1:
        raw = sys.argv[1]
        if Path(raw).exists() and raw.lower().endswith(".csv"):
            tmp = pd.read_csv(raw)
            vals = tmp.select_dtypes(include=["number"]).iloc[0].values.tolist()
        else:
            try:
                vals = [float(x) for x in raw.split(",")]
            except Exception:
                print("ERROR: Could not parse numeric values from input string.")
                return

        try:
            res = predict_from_values(vals)
            print("Prediction result:", res)
        except Exception as e:
            print("ERROR during prediction:", repr(e))
        return

    try:
        df = pd.read_csv(DATA_CSV)
        sample_vals = df.select_dtypes(include=["number"]).iloc[0].values.tolist()
    except Exception as e:
        print("ERROR: Could not read sample row from dataset:", e)
        return

    print("Using first numeric row from dataset as sample input...")
    try:
        res = predict_from_values(sample_vals)
        print("Prediction result (sample):", res)
    except Exception as e:
        print("ERROR during prediction:", repr(e))

if __name__ == "__main__":
    main()
