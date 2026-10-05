import pandas as pd

FULL_PATH = "data/train.csv"
OUTPUT_PATH = "data/train_recent.csv"

# How many days of history to keep.
# Your lag features need at most 28 days lookback + a 7-day rolling window
# on top of that = 35 days minimum. We keep 60 for a safe buffer.
DAYS_TO_KEEP = 60

print(f"Reading {FULL_PATH} (this is the big file, may take a moment)...")
df = pd.read_csv(FULL_PATH, parse_dates=["date"])
print(f"Full file: {len(df):,} rows")

max_date = df["date"].max()
cutoff = max_date - pd.Timedelta(days=DAYS_TO_KEEP)
recent = df[df["date"] >= cutoff].copy()

# Keep only the columns actually used by feature_engineering.py
needed_cols = [c for c in ["date", "store_nbr", "family", "onpromotion", "sales"] if c in recent.columns]
recent = recent[needed_cols]

# Optimize dtypes to shrink memory further
if "store_nbr" in recent.columns:
    recent["store_nbr"] = recent["store_nbr"].astype("int16")
if "family" in recent.columns:
    recent["family"] = recent["family"].astype("category")
if "onpromotion" in recent.columns:
    recent["onpromotion"] = recent["onpromotion"].astype("int32")
if "sales" in recent.columns:
    recent["sales"] = recent["sales"].astype("float32")

recent.to_csv(OUTPUT_PATH, index=False)

print(f"Recent slice ({DAYS_TO_KEEP} days): {len(recent):,} rows")
print(f"Saved to {OUTPUT_PATH}")

import os
full_size = os.path.getsize(FULL_PATH) / (1024 * 1024)
new_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"\nSize comparison: {full_size:.1f} MB -> {new_size:.2f} MB")