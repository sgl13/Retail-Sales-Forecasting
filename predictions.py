# ============================================================
# Import Libraries
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
LGB_MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_model.txt")
FEATURES_PATH = os.path.join(MODEL_DIR, "features_columns.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "ensemble_weight.pkl")

# ============================================================
# FALLBACK FEATURES
# ============================================================

FALLBACK_FEATURES = [
    "store_nbr", "family", "onpromotion", "is_weekend",
    "city", "state", "store_type", "cluster",
    "dcoilwtico", "oil_roll_7", "oil_fwd_1", "oil_fwd_3",
    "is_holiday", "day", "dayofweek", "isweekend", "is_payday",
    "sales_lag_21", "sales_lag_28", "sales_roll_21_7", "promo_roll_3",
]

# ============================================================
# LOAD FEATURE ORDER
# ============================================================

if os.path.exists(FEATURES_PATH):
    FEATURES = joblib.load(FEATURES_PATH)
    print(f"✅ Feature columns loaded from features_columns.pkl ({len(FEATURES)} features)")
else:
    FEATURES = FALLBACK_FEATURES
    print(f"⚠️ Using fallback feature list ({len(FEATURES)} features)")

# ============================================================
# LOAD XGBOOST & LIGHTGBM MODELS
# ============================================================

if not os.path.exists(XGB_MODEL_PATH):
    raise FileNotFoundError(f"❌ XGBoost model missing at: {XGB_MODEL_PATH}")

xgb_model = xgb.Booster()
xgb_model.load_model(XGB_MODEL_PATH)
print(f"✅ XGBoost model loaded ({len(xgb_model.feature_names)} features)")

if not os.path.exists(LGB_MODEL_PATH):
    raise FileNotFoundError(f"❌ LightGBM model missing at: {LGB_MODEL_PATH}")

lgb_model = lgb.Booster(model_file=LGB_MODEL_PATH)
print(f"✅ LightGBM model loaded ({lgb_model.num_feature()} features)")

# ============================================================
# LOAD LABEL ENCODERS & ENSEMBLE WEIGHTS
# ============================================================

encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else {}

xgb_weight = 0.5
lgb_weight = 0.5

if os.path.exists(WEIGHTS_PATH):
    try:
        weights = joblib.load(WEIGHTS_PATH)
        if isinstance(weights, (list, tuple, np.ndarray)) and len(weights) == 2:
            lgb_weight, xgb_weight = weights[0], weights[1]
        elif isinstance(weights, dict):
            xgb_weight = weights.get("xgb_weight", weights.get("xgb", 0.5))
            lgb_weight = weights.get("lgb_weight", weights.get("lgb", 0.5))
    except Exception as e:
        print(f"⚠️ Error loading weights ({e}). Defaulting to 50/50 split.")

print(f"Ensemble weights: LightGBM={lgb_weight}, XGBoost={xgb_weight}")


# ============================================================
# PREPARE FEATURES FOR INFERENCE
# ============================================================

def prepare_features(df):
    # Create an explicit deep copy of the incoming DataFrame
    X = df.copy(deep=True)

    # Safely drop non-feature target/metadata columns without using in-place operations
    drop_cols = [col for col in ["id", "date", "sales"] if col in X.columns and col not in FEATURES]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    # Apply Label Encoding safely
    for col, encoder in encoders.items():
        if col not in X.columns:
            continue

        X[col] = X[col].fillna("unknown").astype(str)
        known_classes = set(encoder.classes_)
        X[col] = X[col].map(
            lambda val: val if val in known_classes else encoder.classes_[0]
        )
        X[col] = encoder.transform(X[col])

    # Missing feature padding
    for col in FEATURES:
        if col not in X.columns:
            X[col] = 0

    # Ensure exact column order and double precision numbers
    X = X[FEATURES]
    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    return X.astype(np.float64)


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_sales(input_df):
    X = prepare_features(input_df)

    # Model Inferences
    xgb_matrix = xgb.DMatrix(X, feature_names=FEATURES)
    xgb_log_pred = xgb_model.predict(xgb_matrix)

    lgb_array = X.to_numpy(dtype=np.float64)
    lgb_log_pred = lgb_model.predict(lgb_array)

    # Inverse log1p scaling
    xgb_pred = np.expm1(xgb_log_pred)
    lgb_pred = np.expm1(lgb_log_pred)

    # Weighted Ensemble + Clipping
    final_pred = (xgb_weight * xgb_pred) + (lgb_weight * lgb_pred)
    return np.maximum(final_pred, 0)


def predict_batch(input_df):
    # Make a clean deep copy to preserve original column formatting
    result = input_df.copy(deep=True)

    # Generate predictions array
    predictions = predict_sales(input_df)

    # Assign predictions directly as a column
    result["sales"] = predictions
    return result