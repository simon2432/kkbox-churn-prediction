"""
Helpers for notebooks / ad-hoc loads.

Uses DEFAULT_PIPELINE_VERSION from src.pipeline (see load_pipeline).
"""

from pathlib import Path

import pandas as pd

from src.pipeline import load_pipeline

_p = load_pipeline()

FEATURES_V3 = _p.FEATURES_V3
PROCESSED_DATA_DIR = _p.PROCESSED_DATA_DIR
PROJECT_ROOT = _p.PROJECT_ROOT
RAW_DATA_DIR = _p.RAW_DATA_DIR
TARGET = _p.TARGET

BASELINE_FEATURES = FEATURES_V3[:9]


def load_raw(filename: str, **kwargs) -> pd.DataFrame:
    path = RAW_DATA_DIR / filename
    print(f"Loading {path}...")
    return pd.read_csv(path, **kwargs)


def save_processed(df: pd.DataFrame, filename: str) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df):,} rows)")


def project_root() -> Path:
    return PROJECT_ROOT
