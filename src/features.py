"""
Feature engineering functions for KKBox churn prediction.

These functions mirror the logic developed in:
  notebooks/02_eda_transactions_features.ipynb

Usage:
    from src.features import build_transaction_features, build_member_features, build_model_dataset
"""

import pandas as pd


TRANSACTION_CUTOFF = "2017-02-28"


def build_transaction_features(transactions: pd.DataFrame, base_msno: pd.Series) -> pd.DataFrame:
    """
    Aggregate transaction features per user, applying the temporal cutoff
    to avoid data leakage.

    Parameters
    ----------
    transactions : pd.DataFrame
        Raw transactions_v2 dataframe. Must have columns:
        msno, transaction_date, auto_renew, payment_plan_days,
        plan_list_price, actual_amount_paid, is_cancel
    base_msno : pd.Series
        Series of msno values present in the base (train + members) dataset.

    Returns
    -------
    pd.DataFrame
        One row per user with aggregated transaction features.
    """
    transactions = transactions[transactions["msno"].isin(base_msno)].copy()
    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"], format="%Y%m%d"
    )

    before_cutoff = transactions[transactions["transaction_date"] <= TRANSACTION_CUTOFF]

    features = (
        before_cutoff.groupby("msno")
        .agg(
            txn_count=("transaction_date", "count"),
            txn_last_date=("transaction_date", "max"),
            txn_first_date=("transaction_date", "min"),
            auto_renew_last=("auto_renew", "last"),
            payment_plan_days_last=("payment_plan_days", "last"),
            plan_list_price_last=("plan_list_price", "last"),
            actual_amount_paid_last=("actual_amount_paid", "last"),
            cancel_count=("is_cancel", "sum"),
            has_cancelled=("is_cancel", "max"),
        )
        .reset_index()
    )

    return features


def build_member_features(members: pd.DataFrame, reference_date: str = TRANSACTION_CUTOFF) -> pd.DataFrame:
    """
    Derive features from the members table.

    Parameters
    ----------
    members : pd.DataFrame
        Raw members_v3 dataframe.
    reference_date : str
        Date to compute days_since_registration relative to. Default: cutoff date.

    Returns
    -------
    pd.DataFrame
        Members dataframe with derived features added.
    """
    members = members.copy()
    members["registration_init_time"] = pd.to_datetime(
        members["registration_init_time"], format="%Y%m%d"
    )
    members["days_since_registration"] = (
        pd.Timestamp(reference_date) - members["registration_init_time"]
    ).dt.days

    members.loc[(members["bd"] < 0) | (members["bd"] > 100), "bd"] = None
    members["gender"] = members["gender"].fillna("unknown")

    return members


def build_model_dataset(
    train: pd.DataFrame,
    members: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full model-ready dataset by merging train labels, member features,
    and aggregated transaction features.

    Parameters
    ----------
    train : pd.DataFrame
        train_v2 with columns msno, is_churn.
    members : pd.DataFrame
        Raw members_v3 dataframe.
    transactions : pd.DataFrame
        Raw transactions_v2 dataframe.

    Returns
    -------
    pd.DataFrame
        Merged dataset ready for feature selection and model training.
    """
    members_enriched = build_member_features(members)
    base = train.merge(members_enriched, on="msno", how="left")

    txn_features = build_transaction_features(transactions, base["msno"])
    df_model = base.merge(txn_features, on="msno", how="left")

    df_model["has_txn_history"] = df_model["txn_count"].notnull().astype(int)
    df_model["txn_count"] = df_model["txn_count"].fillna(0)
    df_model["cancel_count"] = df_model["cancel_count"].fillna(0)
    df_model["has_cancelled"] = df_model["has_cancelled"].fillna(0)

    return df_model
