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
│   ├── raw/              # Original Kaggle CSVs (gitignored; .gitkeep keeps the folder)
│   └── processed/        # Intermediate CSVs (gitignored; .gitkeep keeps the folder)
├── notebooks/
│   ├── 01_eda_train_members.ipynb           # EDA on train_v2 + members_v3
│   ├── 02_eda_transactions_features.ipynb   # Transactions EDA + df_model_baseline_v1.csv
│   ├── 03_eda_user_logs.ipynb               # User logs EDA + log_features_march2017.csv
│   ├── 04_baseline_model.ipynb              # Logistic regression baseline
│   ├── 05_iteration_v2_features_and_trees.ipynb  # v2 features + tree models
│   └── 06_iteration_v3_user_logs.ipynb      # Log features + RF / LightGBM
├── docs/
│   └── iterations/
│       ├── baseline_v1.md
│       ├── iteration_v2.md
│       └── v3_user_logs.md
├── src/
│   ├── __init__.py
│   ├── features.py         # Feature engineering helpers
│   └── utils.py            # Shared utilities
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Datasets

All raw data comes from the Kaggle competition. Files are stored in `data/raw/` and are excluded from version control due to their size.

| File                  | Description                             | Size    |
| --------------------- | --------------------------------------- | ------- |
| `train_v2.csv`        | Labels: msno + is_churn                 | ~44 MB  |
| `members_v3.csv`      | User demographics and registration info | ~408 MB |
| `transactions_v2.csv` | Subscription transaction history        | ~110 MB |
| `user_logs_v2.csv`    | Daily listening activity logs           | ~1.3 GB |
| `transactions.csv`    | Full historical transactions (larger)   | ~1.6 GB |
| `user_logs.csv`       | Full historical user logs               | ~28 GB  |

To run the notebooks, download the data from Kaggle and place all CSV files in `data/raw/`.

---

## Iterations

| Version | Description | Val ROC AUC (best model in notebook) | Notes |
| ------- | ----------- | ------------------------------------ | ----- |
| `baseline_v1` | Logistic regression, 9 transaction + member features | 0.567 | Cutoff 2017-02-28. No class weights. |
| `iteration_v2` | +6 transaction features; DT + RF | **0.673** | RF `class_weight='balanced'`. Recall ~0.84 at threshold 0.4 (see v2 doc). |
| `iteration_v3` | +11 log aggregates (March 2017); RF + **LightGBM** | **0.776** | Single stratified 80/20 split. See threshold table in v3 doc. |

---

## Best results so far (iteration v3)

- **Model:** LightGBM (`n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `class_weight='balanced'`, early stopping on validation)
- **Train/validation split:** 80/20, stratified (`random_state=42`)
- **ROC AUC (validation):** **0.776** (vs. 0.673 for the best v2 Random Forest on the same comparison table in notebook 06)
- **At threshold 0.5:** recall (churn) **0.57**, precision **0.28** — both higher than the v2 RF at 0.5 in that notebook
- **Top signal:** log recency (`days_since_last_log`) leads Random Forest feature importance on v3 features; log-related columns account for ~54% of total RF importance in that run

Write-ups: [`docs/iterations/baseline_v1.md`](docs/iterations/baseline_v1.md), [`docs/iterations/iteration_v2.md`](docs/iterations/iteration_v2.md), [`docs/iterations/v3_user_logs.md`](docs/iterations/v3_user_logs.md).

---

## How to Run

### 1. Set up the environment

From the repo root:

```bash
pip install -r requirements.txt
```

`lightgbm` is included for `06_iteration_v3_user_logs.ipynb`; if you skip that notebook you can omit it.

With conda:

```bash
conda create -n kkbox python=3.10
conda activate kkbox
pip install -r requirements.txt
```

### 2. Get the data

Download the competition data from:  
https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data

Place all CSV files inside `data/raw/`.

### 3. Run the notebooks in order

```
notebooks/01_eda_train_members.ipynb
notebooks/02_eda_transactions_features.ipynb   → data/processed/df_model_baseline_v1.csv
notebooks/03_eda_user_logs.ipynb                 → data/processed/log_features_march2017.csv
notebooks/04_baseline_model.ipynb
notebooks/05_iteration_v2_features_and_trees.ipynb
notebooks/06_iteration_v3_user_logs.ipynb        # needs log_features_march2017.csv + baseline CSV
```

---

## Class imbalance

Roughly **91%** non-churn vs. **9%** churn. That inflates accuracy and makes naive models shy about flagging churners. From iteration v2 onward, tree models use **`class_weight='balanced'`** so the minority class matters in training; threshold tuning (see the iteration docs) still controls precision vs. recall at deployment time.

---

## Next steps

- [x] Class-aware training (`class_weight='balanced'`) and tree models — see v2/v3
- [x] User log aggregates merged into the modelling frame — see v3
- [x] Random Forest and LightGBM on the expanded feature set — see v3
- [ ] **Cross-validation** (replace single holdout metrics)
- [ ] **Richer transaction features** (renewal gaps, plan-change sequences)
- [ ] **Threshold / cost-sensitive** decisions tied to a real retention budget
- [ ] **Submission pipeline** (test predictions, Kaggle-format CSV)

---

## References

- [WSDM KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)
- [WSDMChurnLabeller.scala](data/raw/WSDMChurnLabeller.scala) — official label generation logic from KKBox
