# KKBox Churn Prediction & Customer Risk Ranking

Churn prediction project built on the [WSDM 2018 KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge). The goal goes beyond binary classification — the output is a ranked list of users by churn probability so a retention team can prioritize who to contact when they can't reach everyone.

---

## The problem

KKBox is a music streaming subscription service. A user churns if they don't renew within 30 days of their plan expiring. The label (`is_churn`) is already built correctly in `train_v2.csv` — no need to reconstruct it. Important: `is_cancel` does not define churn, it just flags whether the user explicitly cancelled before expiry.

With ~9% churn rate across ~970k users, a binary classifier that just predicts the majority class gets 91% accuracy and is useless. The useful question is: **given a limited outreach budget, which users should we contact first?** That's what the model is actually optimized for.

The final LightGBM model, at threshold 0.5 on the validation set, captures 57.5% of churners while contacting only 23.6% of users. More usefully — ranking by predicted probability, the top 13% of users by risk captures 51% of churners.

---

## Dataset

| File                  | What it contains                                  |
| --------------------- | ------------------------------------------------- |
| `train_v2.csv`        | User IDs and churn labels                         |
| `members_v3.csv`      | Demographics: city, age, gender, registration     |
| `transactions_v2.csv` | Payment history: plan, price, dates, cancellation |
| `user_logs_v2.csv`    | Daily listening logs (March 2017)                 |

~970k users total.

---

## Temporal handling

Transaction features are computed using only records up to **2017-02-28** — anything after that would be post-expiry data for a chunk of users, which leaks the target.

The listening logs (`user_logs_v2.csv`) cover March 2017 and are used intentionally: activity in the period right before a subscription expires is a strong behavioral signal. Users with no log records in this window get features filled with 0, which turns out to be informative on its own — silent users churn more.

---

## Features

Two pipeline versions:

**Pipeline v1** (26 features): transaction recency/frequency, last payment snapshot, auto-renewal and cancellation flags, member demographics, user log aggregates (listening time, active days, completion rate, log recency).

**Pipeline v2** (44 features): everything in v1, plus price-per-day and payment deviation from the user's own median, plan switches, 10-day and 30-day activity windows from the logs, completion ratio, and a few binary interaction flags (e.g. low usage + no auto-renew).

Both are reproducible from raw data:

```bash
python scripts/build_processed_datasets.py                       # v1 → df_model_v1.csv
python scripts/build_processed_datasets.py --pipeline-version 2  # v2 → df_model_v2.csv
```

---

## Models

Evaluated on a single 80/20 stratified holdout (seed=42, ~194k validation users):

| Model                             | ROC AUC   | Avg Precision |
| --------------------------------- | --------- | ------------- |
| **LightGBM**                      | **0.778** | **0.476**     |
| XGBoost                           | 0.779     | 0.476         |
| HistGradientBoosting (sklearn)    | 0.779     | 0.474         |
| GradientBoosting (200k subsample) | 0.772     | 0.464         |
| Random Forest                     | 0.768     | 0.453         |

LightGBM and XGBoost are essentially tied. LightGBM was chosen for the final model — faster to train and slightly easier to configure for early stopping. The gap over Random Forest (~+0.01 AUC) is consistent across runs.

The baseline logistic regression from the first iteration sat at ~0.62 AUC. The biggest single improvement came from adding the log features (+~0.10 AUC jump going from notebook 05 to 06).

---

## Results (validation set)

| Metric            | Value |
| ----------------- | ----- |
| ROC AUC           | 0.778 |
| Average Precision | 0.476 |
| Precision @ 0.5   | 0.286 |
| Recall @ 0.5      | 0.575 |
| F1 @ 0.5          | 0.382 |

Ranking by predicted probability:

| Users contacted | Share of population | Churners captured |
| --------------- | ------------------- | ----------------- |
| 5,000           | 2.6%                | 23.8%             |
| 10,000          | 5.1%                | 36.1%             |
| 25,000          | 12.9%               | 51.1%             |
| 50,000          | 25.7%               | 64.1%             |
| 100,000         | 51.5%               | 80.2%             |

A random list of 25,000 users would capture ~2.6% of churners. The model gets to 51% with the same budget.

---

## Iteration history

| Notebook                                   | Model               | ROC AUC | What changed                                     |
| ------------------------------------------ | ------------------- | ------- | ------------------------------------------------ |
| `04_baseline_model.ipynb`                  | Logistic Regression | ~0.620  | 9 transaction/member features                    |
| `05_iteration_v2_features_and_trees.ipynb` | Random Forest       | ~0.673  | +6 transaction features, switched to trees       |
| `06_iteration_v3_user_logs.ipynb`          | LightGBM            | ~0.776  | +11 log activity features                        |
| `07_pipeline_v2_boosting_ranking.ipynb`    | LightGBM            | 0.778   | Full pipeline v2, added XGBoost, ranking metrics |

Detailed write-ups: `docs/iterations/v1.md` (notebooks 01–06) and `docs/iterations/v2_final.md` (notebook 07 + final model).

---

## Repo structure

```
kkbox-ml/
├── data/
│   ├── raw/              # Kaggle CSVs (gitignored)
│   └── processed/        # Pipeline outputs (gitignored)
├── docs/
│   └── iterations/
│       ├── v1.md         # EDA through notebook 06
│       └── v2_final.md   # Pipeline v2 and final model
├── notebooks/
│   ├── 01_eda_train_members.ipynb
│   ├── 02_eda_transactions_features.ipynb
│   ├── 03_eda_user_logs.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_iteration_v2_features_and_trees.ipynb
│   ├── 06_iteration_v3_user_logs.ipynb
│   ├── 07_pipeline_v2_boosting_ranking.ipynb
│   └── 08_final_model_and_ranking.ipynb
├── reports/
│   ├── metrics/          # final_metrics.json + figures
│   └── top_users/        # top 1/5/10% by risk score
├── scripts/
│   └── build_processed_datasets.py
├── src/
│   ├── pipeline_v1.py
│   ├── pipeline_v2.py
│   ├── pipeline.py
│   └── utils.py
├── requirements.txt
└── .gitignore
```

---

## How to run

```bash
pip install -r requirements.txt
```

Put the Kaggle CSVs in `data/raw/`, then build the dataset:

```bash
python scripts/build_processed_datasets.py --pipeline-version 2
```

Open `notebooks/08_final_model_and_ranking.ipynb` — this trains the final model, generates the churn ranking, and saves everything to `reports/`.

The notebooks 01–07 are meant to be read in sequence. They document the decisions at each step, not just the final numbers.

---

## What's missing / next steps

- No cross-validation — all metrics are from a single split. The numbers are stable across seeds but CV would tighten the error bars.
- No uplift modeling — the risk score tells you who's likely to churn, not who's likely to respond to a retention action. Those aren't the same thing.
- Threshold hasn't been tuned to a business cost function. At 0.5 the precision/recall tradeoff is arbitrary.
- SHAP would be useful for explaining individual predictions to stakeholders.
