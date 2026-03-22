"""
Shared utilities for KKBox churn prediction.
"""

import pandas as pd


RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"


def load_raw(filename: str, **kwargs) -> pd.DataFrame:
    """Load a CSV from data/raw/."""
    path = f"{RAW_DATA_PATH}/{filename}"
    print(f"Loading {path}...")
    return pd.read_csv(path, **kwargs)


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save a dataframe to data/processed/."""
    path = f"{PROCESSED_DATA_PATH}/{filename}"
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df):,} rows)")


BASELINE_FEATURES = [
    "txn_count",
    "has_txn_history",
    "auto_renew_last",
    "payment_plan_days_last",
    "plan_list_price_last",
    "actual_amount_paid_last",
    "cancel_count",
    "has_cancelled",
    "days_since_registration",
]

TARGET = "is_churn"
