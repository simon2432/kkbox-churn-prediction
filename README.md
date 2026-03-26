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
│       └── v1.md            # notebooks 01–06 + pipeline v1 (single write-up)
├── scripts/
│   └── build_processed_datasets.py   # raw → processed (pipeline v1 / v2)
├── src/
│   ├── __init__.py
│   ├── pipeline.py         # Version registry (load_pipeline) + DEFAULT_PIPELINE_VERSION
│   ├── pipeline_v1.py      # Pipeline v1 — frozen notebook parity (26 features)
│   ├── pipeline_v2.py      # Extended features (windows, interactions, …)
│   └── utils.py            # Notebook helpers; follows DEFAULT_PIPELINE_VERSION
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

Full write-up (data treatment, notebooks 01–06, models, pipeline v1): **[`docs/iterations/v1.md`](docs/iterations/v1.md)**.

| Stage (in v1 doc) | Description                                          | Val ROC AUC (best model in notebook) | Notes                                                          |
| ----------------- | ---------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------- |
| Baseline (nb 04)  | Logistic regression, 9 transaction + member features | ~0.567                               | Cutoff 2017-02-28. No class weights.                           |
| +txn v2 (nb 05)   | +6 transaction features; DT + RF                     | **~0.673**                           | RF `class_weight='balanced'`. High recall at lower thresholds. |
| +logs v3 (nb 06)  | +11 log aggregates; RF + **LightGBM**                | **~0.776**                           | Single stratified 80/20 split; see thresholds in v1 doc.       |

---

## Best results so far (notebook 06 / v3 feature set)

- **Model:** LightGBM (`n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `class_weight='balanced'`, early stopping on validation)
- **Train/validation split:** 80/20, stratified (`random_state=42`)
- **ROC AUC (validation):** **0.776** (vs. 0.673 for the best v2 Random Forest on the same comparison table in notebook 06)
- **At threshold 0.5:** recall (churn) **0.57**, precision **0.28** — both higher than the v2 RF at 0.5 in that notebook
- **Top signal:** log recency (`days_since_last_log`) leads Random Forest feature importance on v3 features; log-related columns account for ~54% of total RF importance in that run

Write-up: [`docs/iterations/v1.md`](docs/iterations/v1.md).

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

### 4. Reproducible pipeline (recommended)

**One command** reads all required raw CSVs, runs every transformation in memory, and writes **a single file**:

| Output                           | Contents                                                                                                                                                                                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/processed/df_model_v1.csv` | `msno`, `is_churn`, 26 features (NaNs → 0) — notebook parity                                                                                                                                                                                                              |
| `data/processed/df_model_v2.csv` | v1 + 10d/30d log windows, `completion_ratio` (drops `completion_rate`), `avg_secs_per_active_day`, `std_usage`; usage-based interactions use median on an 80% stratified calibration split (`INTERACTION_RANDOM_STATE`); extra interaction flags — `--pipeline-version 2` |

```bash
# From the repository root (requires train_v2, members_v3, transactions_v2, user_logs_v2 in data/raw/)
python scripts/build_processed_datasets.py
python scripts/build_processed_datasets.py --pipeline-version 2
```

**Pipeline versions:** `--pipeline-version 1` → `src/pipeline_v1.py` (default). **`--pipeline-version 2`** → `src/pipeline_v2.py`. Compare outputs with different `--out-dir` folders. Default for `src/utils.py`: **`DEFAULT_PIPELINE_VERSION`** in `src/pipeline.py`.

Custom paths: `--raw-dir` / `--out-dir`

**Note:** Notebooks 02–03 may still save intermediate CSVs for teaching/EDA; the CLI pipeline does **not** create those. For modelling, you can load **`df_model_v1.csv`** directly (e.g. simplify notebook 06 to a single `read_csv`).

Implementation: **`src/pipeline_v1.py`** / **`src/pipeline_v2.py`** (`build_final_dataframe`, `run_full_pipeline`); **`src/pipeline.py`** — version registry (`load_pipeline`).

---

## Class imbalance

Roughly **91%** non-churn vs. **9%** churn. That inflates accuracy and makes naive models shy about flagging churners. From notebook 05 onward, tree models use **`class_weight='balanced'`** so the minority class matters in training; threshold tuning (see [`docs/iterations/v1.md`](docs/iterations/v1.md)) still controls precision vs. recall at deployment time.

---

## Next steps

- [x] Class-aware training (`class_weight='balanced'`) and tree models — see [`docs/iterations/v1.md`](docs/iterations/v1.md)
- [x] User log aggregates merged into the modelling frame — see v1 doc
- [x] Random Forest and LightGBM on the expanded feature set — see v1 doc
- [ ] **Cross-validation** (replace single holdout metrics)
- [ ] **Richer transaction features** (renewal gaps, plan-change sequences)
- [ ] **Threshold / cost-sensitive** decisions tied to a real retention budget
- [ ] **Submission pipeline** (test predictions, Kaggle-format CSV)

---

## References

- [WSDM KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)
- [WSDMChurnLabeller.scala](data/raw/WSDMChurnLabeller.scala) — official label generation logic from KKBox
