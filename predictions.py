import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
LABEL_ENCODERS_PATH = os.path.join(BASE_DIR, "models", "label_encoders.pkl")

# Load model and booster
model = joblib.load(MODEL_PATH)
booster = model.get_booster() if hasattr(model, "get_booster") else model

# Extract expected features directly from booster
expected_features = getattr(booster, "feature_names", None) or getattr(model, "feature_names_in_", None)

# Load Label Encoders if present
label_encoders = joblib.load(LABEL_ENCODERS_PATH) if os.path.exists(LABEL_ENCODERS_PATH) else {}


def prepare_features(df):
    df_clean = df.copy()

    # Drop non-feature ID / target / date columns
    ignore_cols = ["id", "sales", "date"]
    for col in ignore_cols:
        if col in df_clean.columns and (expected_features is None or col not in expected_features):
            df_clean = df_clean.drop(columns=[col])

    # 1. Apply Label Encoders if present
    if label_encoders:
        for col, le in label_encoders.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str)
                known_classes = set(le.classes_)
                df_clean[col] = df_clean[col].apply(lambda x: x if x in known_classes else le.classes_[0])
                df_clean[col] = le.transform(df_clean[col])

    # 2. Convert string/categorical columns into numeric factorized codes
    for col in df_clean.columns:
        if df_clean[col].dtype == "object" or str(df_clean[col].dtype) == "category":
            df_clean[col] = pd.factorize(df_clean[col])[0]

    # 3. Ensure all expected features exist and order them strictly
    if expected_features:
        for col in expected_features:
            if col not in df_clean.columns:
                df_clean[col] = 0.0

        df_clean = df_clean[expected_features]

    # Force ALL columns to standard float64 (removes pandas category metadata)
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0).astype(np.float64)

    return df_clean


def predict_sales(input_df):
    processed_df = prepare_features(input_df)

    # Pass pure numeric matrix without enable_categorical flag
    dmatrix = xgb.DMatrix(processed_df)
    return booster.predict(dmatrix)


def predict_batch(input_df):
    processed_df = prepare_features(input_df)

    dmatrix = xgb.DMatrix(processed_df)
    preds = booster.predict(dmatrix)

    result_df = input_df.copy()
    result_df["sales"] = preds
    return result_df