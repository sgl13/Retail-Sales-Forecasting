import os
import numpy as np
import pandas as pd

MISSING_REFERENCE_FILES = []

# Load reference datasets safely
try:
    STORES = pd.read_csv("data/stores.csv").rename(columns={"type": "store_type"})
except Exception as e:
    STORES = pd.DataFrame()
    MISSING_REFERENCE_FILES.append("data/stores.csv")

try:
    OIL_RAW = pd.read_csv("data/oil.csv", parse_dates=["date"])
except Exception as e:
    OIL_RAW = pd.DataFrame()
    MISSING_REFERENCE_FILES.append("data/oil.csv")

try:
    HOLIDAYS = pd.read_csv("data/holidays_events.csv", parse_dates=["date"])
except Exception as e:
    HOLIDAYS = pd.DataFrame()
    MISSING_REFERENCE_FILES.append("data/holidays_events.csv")

try:
    HISTORY = pd.read_csv("data/train_recent.csv", parse_dates=["date"])
except Exception as e:
    try:
        HISTORY = pd.read_csv("data/train.csv", parse_dates=["date"])
    except Exception as e2:
        HISTORY = pd.DataFrame()
        MISSING_REFERENCE_FILES.append("data/train_recent.csv")


def build_oil_features(oil_raw: pd.DataFrame) -> pd.DataFrame:
    if oil_raw.empty:
        return pd.DataFrame()

    oil = (
        oil_raw.set_index("date")
        .resample("D")
        .mean()
        .interpolate(limit_direction="both")
        .reset_index()
    )
    oil["oil_fwd_1"] = oil["dcoilwtico"].shift(-1).bfill()
    oil["oil_fwd_3"] = oil["dcoilwtico"].shift(-3).bfill()
    oil["oil_roll_7"] = oil["dcoilwtico"].rolling(7, min_periods=1).mean()
    return oil


def apply_holiday(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_holiday"] = 0

    if holidays.empty:
        return df

    # Vectorized National Holidays
    nat_dates = set(holidays[holidays["locale"] == "National"]["date"])
    df.loc[df["date"].isin(nat_dates), "is_holiday"] = 1

    # Regional Holidays (matching state)
    if "state" in df.columns:
        reg_holidays = holidays[holidays["locale"] == "Regional"][["date", "locale_name"]].drop_duplicates()
        merged_reg = df[["date", "state"]].merge(
            reg_holidays, left_on=["date", "state"], right_on=["date", "locale_name"], how="inner"
        )
        if not merged_reg.empty:
            reg_keys = set(zip(merged_reg["date"], merged_reg["state"]))
            df_keys = list(zip(df["date"], df["state"]))
            df.loc[[k in reg_keys for k in df_keys], "is_holiday"] = 1

    # Local Holidays (matching city)
    if "city" in df.columns:
        loc_holidays = holidays[holidays["locale"] == "Local"][["date", "locale_name"]].drop_duplicates()
        merged_loc = df[["date", "city"]].merge(
            loc_holidays, left_on=["date", "city"], right_on=["date", "locale_name"], how="inner"
        )
        if not merged_loc.empty:
            loc_keys = set(zip(merged_loc["date"], merged_loc["city"]))
            df_keys = list(zip(df["date"], df["city"]))
            df.loc[[k in loc_keys for k in df_keys], "is_holiday"] = 1

    # Override for transferred work days
    work_days = set(holidays[holidays["type"] == "work day"]["date"])
    df.loc[df["date"].isin(work_days), "is_holiday"] = 0

    return df


def engineer_batch_features(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    df = uploaded_df.copy()
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])

    # 1. Merge Store Metadata & Oil
    if not STORES.empty and "store_nbr" in df.columns:
        df = df.merge(STORES, on="store_nbr", how="left")

    if not OIL_RAW.empty:
        oil = build_oil_features(OIL_RAW)
        df = df.merge(oil, on="date", how="left")

    # 2. Holiday Flag
    df = apply_holiday(df, HOLIDAYS)

    # 3. Calendar Features (Deduplicated flags)
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["isweekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_weekend"] = df["isweekend"]
    df["is_payday"] = ((df["day"] == 1) | (df["date"].dt.is_month_end)).astype(int)

    # 4. Lag & Rolling Features
    if not HISTORY.empty and "store_nbr" in df.columns and "family" in df.columns:
        history_slim = HISTORY[["date", "store_nbr", "family", "onpromotion", "sales"]].copy()
        upload_slim = df[["date", "store_nbr", "family", "onpromotion"]].copy()
        upload_slim["sales"] = np.nan

        combined = pd.concat([history_slim, upload_slim], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "store_nbr", "family"], keep="last")
        combined = combined.sort_values(["store_nbr", "family", "date"])

        combined["sales_lag_21"] = combined.groupby(["store_nbr", "family"])["sales"].shift(21)
        combined["sales_lag_28"] = combined.groupby(["store_nbr", "family"])["sales"].shift(28)
        combined["sales_roll_21_7"] = combined.groupby(["store_nbr", "family"])["sales_lag_21"].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        combined["promo_roll_3"] = combined.groupby(["store_nbr", "family"])["onpromotion"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )

        lag_cols = [
            "date",
            "store_nbr",
            "family",
            "sales_lag_21",
            "sales_lag_28",
            "sales_roll_21_7",
            "promo_roll_3",
        ]
        df = df.merge(combined[lag_cols], on=["date", "store_nbr", "family"], how="left")

    # Fill NaN for engineered numeric features (excluding target 'sales' if present)
    feature_num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "sales"]
    df[feature_num_cols] = df[feature_num_cols].fillna(0)

    return df