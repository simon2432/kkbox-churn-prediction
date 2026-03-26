"""
Pipeline v1 — same transformations as notebooks 02 → 03 → 06, in one pass in memory.

Writes a single processed CSV: OUT_FINAL (msno, is_churn, 26 training features).
For experiments, add pipeline_v2.py and register it in pipeline.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PIPELINE_VERSION = 1

# =============================================================================
# Paths & file names
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TRANSACTION_CUTOFF = "2017-02-28"

FILE_TRAIN = "train_v2.csv"
FILE_MEMBERS = "members_v3.csv"
FILE_TRANSACTIONS = "transactions_v2.csv"
FILE_USER_LOGS = "user_logs_v2.csv"

# Single artefact from this pipeline (no intermediate CSVs on disk)
OUT_FINAL = "df_model_v1.csv"

# =============================================================================
# Feature column lists (training matrix)
# =============================================================================

FEATURES_V1 = [
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

FEATURES_TXN_V2 = [
    "days_since_last_txn",
    "has_recent_txn",
    "txn_per_day",
    "avg_payment",
    "price_discount",
    "plan_consistency",
]

FEATURES_LOGS = [
    "total_secs_sum",
    "total_secs_mean",
    "active_days",
    "num_100_sum",
    "num_unq_sum",
    "num_unq_mean",
    "completion_rate",
    "days_since_last_log",
    "has_recent_activity",
    "peak_day_secs",
    "activity_cv",
]

FEATURES_V3 = FEATURES_V1 + FEATURES_TXN_V2 + FEATURES_LOGS

LOG_FEATURE_COLUMNS = [
    "total_secs_sum",
    "total_secs_mean",
    "num_unq_sum",
    "num_unq_mean",
    "num_100_sum",
    "active_days",
    "days_since_last_log",
    "has_recent_activity",
    "completion_rate",
    "peak_day_secs",
    "activity_cv",
]

TARGET = "is_churn"


def aggregate_transactions_before_cutoff(
    transactions: pd.DataFrame,
    base_msno: pd.Series,
    cutoff: str | pd.Timestamp = TRANSACTION_CUTOFF,
) -> pd.DataFrame:
    t = transactions[transactions["msno"].isin(base_msno)].copy()
    for col in ("transaction_date", "membership_expire_date"):
        t[col] = pd.to_datetime(t[col], format="%Y%m%d", errors="coerce")

    cutoff_ts = pd.Timestamp(cutoff)
    before = t[t["transaction_date"] <= cutoff_ts]

    txn_features = (
        before.groupby("msno", sort=False)
        .agg(
            txn_count=("transaction_date", "count"),
            txn_last_date=("transaction_date", "max"),
            txn_first_date=("transaction_date", "min"),
            cancel_count=("is_cancel", "sum"),
            has_cancelled=("is_cancel", "max"),
            auto_renew_last=("is_auto_renew", "last"),
            payment_plan_days_last=("payment_plan_days", "last"),
            plan_list_price_last=("plan_list_price", "last"),
            actual_amount_paid_last=("actual_amount_paid", "last"),
            payment_plan_days_median=("payment_plan_days", "median"),
            actual_amount_paid_median=("actual_amount_paid", "median"),
        )
        .reset_index()
    )
    txn_features["days_between_first_last_txn"] = (
        txn_features["txn_last_date"] - txn_features["txn_first_date"]
    ).dt.days
    return txn_features


def build_baseline_dataset(
    train: pd.DataFrame,
    members: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp = TRANSACTION_CUTOFF,
) -> pd.DataFrame:
    members = members.copy()
    members["registration_init_time"] = pd.to_datetime(
        members["registration_init_time"], format="%Y%m%d", errors="coerce"
    )
    base = train.merge(members, on="msno", how="left")
    txn_features = aggregate_transactions_before_cutoff(transactions, base["msno"], cutoff=cutoff)
    df_model = base.merge(txn_features, on="msno", how="left")

    df_model["has_txn_history"] = df_model["txn_count"].notnull().astype(int)
    df_model["txn_count"] = df_model["txn_count"].fillna(0)
    df_model["cancel_count"] = df_model["cancel_count"].fillna(0)
    df_model["has_cancelled"] = df_model["has_cancelled"].fillna(0)

    ref = pd.Timestamp(cutoff)
    df_model["days_since_registration"] = (
        ref - df_model["registration_init_time"]
    ).dt.days
    df_model.loc[(df_model["bd"] < 0) | (df_model["bd"] > 100), "bd"] = pd.NA
    df_model["gender"] = df_model["gender"].fillna("unknown")
    return df_model


def build_logs_aggregates(user_logs: pd.DataFrame) -> pd.DataFrame:
    logs = user_logs.copy()
    logs["date"] = pd.to_datetime(logs["date"], format="%Y%m%d", errors="coerce")
    song_cols = ["num_25", "num_50", "num_75", "num_985", "num_100"]
    logs["total_songs"] = logs[song_cols].sum(axis=1)
    reference_date = logs["date"].max()

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
            num_100_mean=("num_100", "mean"),
            total_songs_mean=("total_songs", "mean"),
            peak_day_secs=("total_secs", "max"),
            activity_std=("total_secs", "std"),
        )
        .reset_index()
    )
    log_features["days_since_last_log"] = (
        reference_date - log_features["last_log_date"]
    ).dt.days
    log_features["has_recent_activity"] = (
        log_features["days_since_last_log"] <= 7
    ).astype(int)
    log_features["completion_rate"] = (
        log_features["num_100_mean"] / (log_features["total_songs_mean"] + 1e-6)
    ).clip(upper=1.0)
    log_features["activity_cv"] = log_features["activity_std"] / (
        log_features["total_secs_mean"] + 1e-6
    )
    drop_cols = ["last_log_date", "num_100_mean", "total_songs_mean", "activity_std"]
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


def add_v2_v3_derived_features(
    df: pd.DataFrame,
    *,
    cutoff: str | pd.Timestamp = TRANSACTION_CUTOFF,
) -> pd.DataFrame:
    out = df.copy()
    cutoff_ts = pd.Timestamp(cutoff)
    out["txn_last_date"] = pd.to_datetime(out["txn_last_date"])
    out["txn_first_date"] = pd.to_datetime(out["txn_first_date"])

    days_since_last = (cutoff_ts - out["txn_last_date"]).dt.days
    max_valid = days_since_last.max()
    if pd.isna(max_valid):
        max_valid = 0
    out["days_since_last_txn"] = days_since_last.fillna(max_valid + 1)
    out["has_recent_txn"] = (out["days_since_last_txn"] < 30).astype(int)
    out["txn_per_day"] = out["txn_count"] / (
        out["days_between_first_last_txn"].fillna(0) + 1
    )
    out["avg_payment"] = out["actual_amount_paid_median"].fillna(0)
    out["price_discount"] = (
        out["plan_list_price_last"].fillna(0) - out["actual_amount_paid_last"].fillna(0)
    ).clip(lower=0)
    out["plan_consistency"] = (
        out["payment_plan_days_last"].fillna(-1)
        == out["payment_plan_days_median"].fillna(-2)
    ).astype(int)
    return out


def training_feature_matrix(df: pd.DataFrame, *, fillna_value: float = 0.0) -> pd.DataFrame:
    return df[FEATURES_V3].fillna(fillna_value)


def build_final_dataframe(raw_dir: Path) -> pd.DataFrame:
    """
    Load raw CSVs from raw_dir and return the model-ready frame (no write).

    Order: baseline (train+members+txn) → merge log aggregates → v2/v3 txn features → feature columns.
    """
    train = pd.read_csv(raw_dir / FILE_TRAIN)
    members = pd.read_csv(raw_dir / FILE_MEMBERS)
    transactions = pd.read_csv(raw_dir / FILE_TRANSACTIONS)
    user_logs = pd.read_csv(raw_dir / FILE_USER_LOGS)

    baseline = build_baseline_dataset(train, members, transactions)
    log_agg = build_logs_aggregates(user_logs)
    log_wide = align_logs_to_msno_list(
        log_agg, baseline["msno"], fill_missing_with_zero=True
    )
    df = baseline.merge(log_wide, on="msno", how="left", validate="one_to_one")
    df = add_v2_v3_derived_features(df)
    X = training_feature_matrix(df, fillna_value=0.0)
    return pd.concat([df[["msno", TARGET]], X], axis=1)


def run_full_pipeline(raw_dir: Path, out_dir: Path) -> None:
    """End-to-end v1: read raw → transform → write a single CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final_df = build_final_dataframe(raw_dir)
    path = out_dir / OUT_FINAL
    final_df.to_csv(path, index=False)
    print(
        f"[pipeline v1] Wrote {path} ({len(final_df):,} rows, {final_df.shape[1]} cols; "
        f"{len(FEATURES_V3)} features + msno + {TARGET})"
    )
