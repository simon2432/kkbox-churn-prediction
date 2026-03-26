"""
Pipeline v2 — v1 core path + extra log metrics, 10d/30d windows, interaction flags.

Usage thresholds: median of total_secs_sum is fit on a stratified 80% calibration split
(same random_state) to limit leakage when the table is later CV-split.

Writes a single processed CSV: OUT_FINAL (msno, is_churn, feature columns).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import pipeline_v1 as v1

PIPELINE_VERSION = 2

PROJECT_ROOT = v1.PROJECT_ROOT
RAW_DATA_DIR = v1.RAW_DATA_DIR
PROCESSED_DATA_DIR = v1.PROCESSED_DATA_DIR
TRANSACTION_CUTOFF = v1.TRANSACTION_CUTOFF

FILE_TRAIN = v1.FILE_TRAIN
FILE_MEMBERS = v1.FILE_MEMBERS
FILE_TRANSACTIONS = v1.FILE_TRANSACTIONS
FILE_USER_LOGS = v1.FILE_USER_LOGS

OUT_FINAL = "df_model_v2.csv"
TARGET = v1.TARGET

# v1.log block without completion_rate (v2 keeps completion_ratio only — same |corr| to is_churn on sample data)
FEATURES_LOGS_V2 = [c for c in v1.FEATURES_LOGS if c != "completion_rate"]

FEATURES_V2_EXTRA_LOGS = [
    "completion_ratio",
    "avg_secs_per_active_day",
    "std_usage",
    "secs_first_10d",
    "secs_last_10d",
    "secs_10d_tail_head_ratio",
    "plays_first_10d",
    "plays_last_10d",
    "plays_10d_tail_head_ratio",
    "secs_first_30d",
    "secs_last_30d",
    "secs_30d_tail_head_ratio",
    "plays_first_30d",
    "plays_last_30d",
    "plays_30d_tail_head_ratio",
]

FEATURES_V2_INTERACTIONS = [
    "low_usage_no_autorenew",
    "high_usage_autorenew",
    "recent_activity_recent_txn",
    "low_usage_no_recent_txn",
]

FEATURES_V3 = (
    v1.FEATURES_V1
    + v1.FEATURES_TXN_V2
    + FEATURES_LOGS_V2
    + FEATURES_V2_EXTRA_LOGS
    + FEATURES_V2_INTERACTIONS
)

LOG_FEATURE_COLUMNS = [c for c in v1.LOG_FEATURE_COLUMNS if c != "completion_rate"] + FEATURES_V2_EXTRA_LOGS

INTERACTION_CALIBRATION_FRACTION = 0.8
INTERACTION_RANDOM_STATE = 42


def _calibration_indices(n: int, y: np.ndarray, *, train_fraction: float, random_state: int) -> np.ndarray:
    """Row indices used to compute usage median (~ stratified fraction per class, NumPy only)."""
    idx = np.arange(n, dtype=np.int64)
    if n < 2 or train_fraction >= 1.0:
        return idx
    rng = np.random.RandomState(random_state)
    parts: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        if cls_idx.size == 0:
            continue
        perm = rng.permutation(cls_idx)
        n_take = max(1, int(np.floor(perm.size * train_fraction)))
        n_take = min(n_take, perm.size)
        parts.append(perm[:n_take])
    if not parts:
        return idx
    return np.sort(np.concatenate(parts))


def usage_median_for_split(df: pd.DataFrame) -> float:
    """Median total_secs_sum on calibration subset only (stratified by is_churn)."""
    y = df[TARGET].to_numpy()
    cal_idx = _calibration_indices(
        len(df),
        y,
        train_fraction=INTERACTION_CALIBRATION_FRACTION,
        random_state=INTERACTION_RANDOM_STATE,
    )
    secs = df.iloc[cal_idx]["total_secs_sum"].fillna(0.0)
    m = float(secs.median()) if len(secs) else 0.0
    return 0.0 if pd.isna(m) else m


def build_logs_aggregates_v2(user_logs: pd.DataFrame) -> pd.DataFrame:
    logs = user_logs.copy()
    logs["date"] = pd.to_datetime(logs["date"], format="%Y%m%d", errors="coerce")
    song_cols = ["num_25", "num_50", "num_75", "num_985", "num_100"]
    logs["total_songs"] = logs[song_cols].sum(axis=1)
    reference_date = logs["date"].max()

    span = (
        logs.groupby("msno", sort=False)["date"]
        .agg(user_first_log="min", user_last_log="max")
        .reset_index()
    )
    logs = logs.merge(span, on="msno", how="left")
    logs["days_from_first"] = (logs["date"] - logs["user_first_log"]).dt.days
    logs["days_before_last"] = (logs["user_last_log"] - logs["date"]).dt.days

    def _window_aggs(days_from_first_max: int, days_before_last_max: int, suffix: str):
        em = logs["days_from_first"] <= days_from_first_max
        lm = logs["days_before_last"] <= days_before_last_max
        e = (
            logs.loc[em]
            .groupby("msno", sort=False)
            .agg(
                **{
                    f"secs_first_{suffix}": ("total_secs", "sum"),
                    f"plays_first_{suffix}": ("total_songs", "sum"),
                }
            )
            .reset_index()
        )
        l = (
            logs.loc[lm]
            .groupby("msno", sort=False)
            .agg(
                **{
                    f"secs_last_{suffix}": ("total_secs", "sum"),
                    f"plays_last_{suffix}": ("total_songs", "sum"),
                }
            )
            .reset_index()
        )
        return e, l

    early10, late10 = _window_aggs(9, 9, "10d")
    early30, late30 = _window_aggs(29, 29, "30d")

    log_features = (
        logs.groupby("msno", sort=False)
        .agg(
            total_secs_sum=("total_secs", "sum"),
            total_secs_mean=("total_secs", "mean"),
            num_unq_sum=("num_unq", "sum"),
            num_unq_mean=("num_unq", "mean"),
            num_100_sum=("num_100", "sum"),
            active_days=("date", "nunique"),
            last_log_date=("date", "max"),
            total_plays_sum=("total_songs", "sum"),
            peak_day_secs=("total_secs", "max"),
            activity_std=("total_secs", "std"),
        )
        .reset_index()
    )
    for part in (early10, late10, early30, late30):
        log_features = log_features.merge(part, on="msno", how="left")

    log_features["days_since_last_log"] = (
        reference_date - log_features["last_log_date"]
    ).dt.days
    log_features["has_recent_activity"] = (
        log_features["days_since_last_log"] <= 7
    ).astype(int)
    log_features["std_usage"] = log_features["activity_std"]
    log_features["activity_cv"] = log_features["activity_std"] / (
        log_features["total_secs_mean"] + 1e-6
    )
    log_features["avg_secs_per_active_day"] = log_features["total_secs_sum"] / (
        log_features["active_days"] + 1e-6
    )
    log_features["completion_ratio"] = log_features["num_100_sum"] / (
        log_features["total_plays_sum"] + 1e-6
    )
    log_features["completion_ratio"] = log_features["completion_ratio"].clip(upper=1.0)

    window_pairs = (
        ("10d", "secs_first_10d", "secs_last_10d", "plays_first_10d", "plays_last_10d", "secs_10d_tail_head_ratio", "plays_10d_tail_head_ratio"),
        ("30d", "secs_first_30d", "secs_last_30d", "plays_first_30d", "plays_last_30d", "secs_30d_tail_head_ratio", "plays_30d_tail_head_ratio"),
    )
    for _, sf, sl, pf, pl, rsecs, rplays in window_pairs:
        for c in (sf, sl, pf, pl):
            log_features[c] = log_features[c].fillna(0)
        log_features[rsecs] = (log_features[sl] / (log_features[sf] + 1e-6)).clip(upper=50.0)
        log_features[rplays] = (log_features[pl] / (log_features[pf] + 1e-6)).clip(upper=50.0)

    drop_cols = ["last_log_date", "activity_std", "total_plays_sum"]
    return log_features.drop(columns=drop_cols)


def align_logs_to_msno_list(
    log_aggregates: pd.DataFrame,
    msno: pd.Series,
    *,
    fill_missing_with_zero: bool = True,
) -> pd.DataFrame:
    base = pd.DataFrame({"msno": pd.unique(msno)})
    out = base.merge(log_aggregates, on="msno", how="left")
    if fill_missing_with_zero:
        out[LOG_FEATURE_COLUMNS] = out[LOG_FEATURE_COLUMNS].fillna(0)
    return out[["msno"] + LOG_FEATURE_COLUMNS]


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    low/high_usage use median(total_secs_sum) on stratified calibration subset only.
    """
    out = df.copy()
    med = usage_median_for_split(out)
    secs = out["total_secs_sum"].fillna(0.0)
    renew = pd.to_numeric(out["auto_renew_last"], errors="coerce").fillna(0)
    renew = (renew >= 0.5).astype(int)
    low = (secs <= med).astype(int)
    high = (secs > med).astype(int)
    har = pd.to_numeric(out["has_recent_txn"], errors="coerce").fillna(0).astype(int)
    hra = pd.to_numeric(out["has_recent_activity"], errors="coerce").fillna(0).astype(int)

    out["low_usage_no_autorenew"] = ((low == 1) & (renew == 0)).astype(int)
    out["high_usage_autorenew"] = ((high == 1) & (renew == 1)).astype(int)
    out["recent_activity_recent_txn"] = ((hra == 1) & (har == 1)).astype(int)
    out["low_usage_no_recent_txn"] = ((low == 1) & (har == 0)).astype(int)
    return out


def training_feature_matrix(df: pd.DataFrame, *, fillna_value: float = 0.0) -> pd.DataFrame:
    return df[FEATURES_V3].fillna(fillna_value)


def build_final_dataframe(raw_dir: Path) -> pd.DataFrame:
    train = pd.read_csv(raw_dir / FILE_TRAIN)
    members = pd.read_csv(raw_dir / FILE_MEMBERS)
    transactions = pd.read_csv(raw_dir / FILE_TRANSACTIONS)
    user_logs = pd.read_csv(raw_dir / FILE_USER_LOGS)

    baseline = v1.build_baseline_dataset(train, members, transactions)
    log_agg = build_logs_aggregates_v2(user_logs)
    log_wide = align_logs_to_msno_list(log_agg, baseline["msno"], fill_missing_with_zero=True)
    df = baseline.merge(log_wide, on="msno", how="left", validate="one_to_one")
    df = v1.add_v2_v3_derived_features(df)
    df = add_interaction_features(df)
    X = training_feature_matrix(df, fillna_value=0.0)
    return pd.concat([df[["msno", TARGET]], X], axis=1)


def run_full_pipeline(raw_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final_df = build_final_dataframe(raw_dir)
    path = out_dir / OUT_FINAL
    final_df.to_csv(path, index=False)
    print(
        f"[pipeline v2] Wrote {path} ({len(final_df):,} rows, {final_df.shape[1]} cols; "
        f"{len(FEATURES_V3)} features + msno + {TARGET})"
    )
