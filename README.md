# KKBox Churn Prediction

Predicting subscriber churn for KKBox, Taiwan's leading music streaming service.  
This project is based on the [WSDM - KKBox's Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) dataset from Kaggle.

---

## Problem Statement

Given a user's subscription and transaction history, predict whether they will churn (not renew their subscription) in the following month.

- **Target variable:** `is_churn` from `train_v2.csv`
- **Positive class (1):** user churns — does not renew their subscription
- **Negative class (0):** user renews their subscription

---

## Project Structure

```
kkbox-ml/
├── data/
│   ├── raw/              # Original Kaggle CSVs (not tracked in Git)
│   └── processed/        # Intermediate and model-ready datasets (not tracked in Git)
├── notebooks/
│   ├── 01_eda_train_members.ipynb          # EDA on train_v2 + members_v3
│   ├── 02_eda_transactions_features.ipynb  # EDA on transactions_v2 + feature engineering
│   ├── 03_eda_user_logs.ipynb              # EDA on user_logs_v2
│   └── 04_baseline_model.ipynb            # Logistic Regression baseline
├── docs/
│   └── iterations/
│       └── baseline_v1.md   # Full documentation for the baseline iteration
├── src/
│   ├── __init__.py
│   ├── features.py          # Feature engineering functions
│   └── utils.py             # Shared utilities
├── .gitignore
└── README.md
```

---

## Datasets

All raw data comes from the Kaggle competition. Files are stored in `data/raw/` and are excluded from version control due to their size.

| File | Description | Size |
|---|---|---|
| `train_v2.csv` | Labels: msno + is_churn | ~44 MB |
| `members_v3.csv` | User demographics and registration info | ~408 MB |
| `transactions_v2.csv` | Subscription transaction history | ~110 MB |
| `user_logs_v2.csv` | Daily listening activity logs | ~1.3 GB |
| `transactions.csv` | Full historical transactions (larger) | ~1.6 GB |
| `user_logs.csv` | Full historical user logs | ~28 GB |

To run the notebooks, download the data from Kaggle and place all CSV files in `data/raw/`.

---

## Iterations

| Version | Description | ROC AUC | Notes |
|---|---|---|---|
| `baseline_v1` | Logistic Regression on 9 transaction + member features | 0.567 | Cutoff 2017-02-28. Class imbalance unaddressed. |

---

## Baseline Results (v1)

- **Model:** Logistic Regression (`max_iter=1000`)
- **Train/Val split:** 80/20, stratified
- **ROC AUC:** 0.567
- **Accuracy:** 0.914 *(misleading — driven by class imbalance)*
- **F1-score (churn class):** 0.12 at threshold 0.5

The model struggles to identify churners. `has_txn_history` and `auto_renew_last` are the most predictive features. See [`docs/iterations/baseline_v1.md`](docs/iterations/baseline_v1.md) for full analysis.

---

## How to Run

### 1. Set up the environment

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Or with conda:

```bash
conda create -n kkbox python=3.10
conda activate kkbox
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 2. Get the data

Download the competition data from:  
https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data

Place all CSV files inside `data/raw/`.

### 3. Run the notebooks in order

```
notebooks/01_eda_train_members.ipynb
notebooks/02_eda_transactions_features.ipynb   ← generates data/processed/df_model_baseline_v1.csv
notebooks/03_eda_user_logs.ipynb
notebooks/04_baseline_model.ipynb
```

---

## Class Imbalance

The dataset is heavily imbalanced:

- **No churn (0):** ~91%
- **Churn (1):** ~9%

This inflates accuracy metrics and suppresses recall on the minority class. Handling this imbalance is a priority for the next iteration.

---

## Next Steps

- [ ] Address class imbalance (SMOTE, class weights, undersampling)
- [ ] Add user log features (listening behavior)
- [ ] Try tree-based models (Random Forest, XGBoost, LightGBM)
- [ ] Engineer richer transaction features (renewal gaps, plan change patterns)
- [ ] Proper cross-validation setup
- [ ] Threshold tuning based on business cost matrix

---

## References

- [WSDM KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)
- [WSDMChurnLabeller.scala](data/raw/WSDMChurnLabeller.scala) — official label generation logic from KKBox
