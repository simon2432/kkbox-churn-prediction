# Iteration v3 — User log features

**Date:** March 2026  
**Status:** Complete  
**Notebook:** `notebooks/06_iteration_v3_user_logs.ipynb`  
**Inputs:** `data/processed/df_model_baseline_v1.csv` (from notebooks 02 + 04 pipeline) and `data/processed/log_features_march2017.csv` (from `notebooks/03_eda_user_logs.ipynb`)

---

## Objective

Push churn prediction beyond iteration v2 by adding **listening behaviour** from `user_logs_v2.csv`. The working hypothesis was simple: whether someone actually uses the product in the label month should carry signal that pure subscription metadata does not capture.

Iteration v2 had already lifted ROC AUC to **0.673** with transaction recency and tree models. This iteration asks whether **March 2017 log aggregates**, merged per user, improve ranking and calibration further.

---

## Data and leakage guardrails

- **Transaction / member features** use the same **cutoff date `2017-02-28`** as earlier iterations (no post-cutoff transactions in features).
- **Log features** are built from **March 2017** activity in `user_logs_v2.csv`, aligned with the competition’s churn horizon (March behaviour vs. the `is_churn` label). Details of the aggregation window and column definitions are in `03_eda_user_logs.ipynb`.
- In the merged training frame used in notebook 06, **all labelled users had non-null log aggregates** (100% coverage in that run), so there was no separate “no logs” slice to compare within this notebook.

---

## Feature set

### Inherited from v1 (9)

`txn_count`, `has_txn_history`, `auto_renew_last`, `payment_plan_days_last`, `plan_list_price_last`, `actual_amount_paid_last`, `cancel_count`, `has_cancelled`, `days_since_registration`.

### Added in v2 (6)

`days_since_last_txn`, `has_recent_txn`, `txn_per_day`, `avg_payment`, `price_discount`, `plan_consistency`.

### New log-derived features (11)

| Feature                        | Role (intuition)                                       |
| ------------------------------ | ------------------------------------------------------ |
| `total_secs_sum`               | Total listening time in the window                     |
| `total_secs_mean`              | Average daily listening time                           |
| `active_days`                  | Count of days with any activity                        |
| `num_100_sum`                  | Sum of “100% of song played” counts (engagement proxy) |
| `num_unq_sum` / `num_unq_mean` | Unique tracks — breadth of listening                   |
| `completion_rate`              | Share of plays that completed                          |
| `days_since_last_log`          | Recency of last log row in the window                  |
| `has_recent_activity`          | Short-horizon activity flag                            |
| `peak_day_secs`                | Concentration of listening on the heaviest day         |
| `activity_cv`                  | Variability of daily activity                          |

**Total:** 26 features for v3.

---

## Modelling setup

- **Split:** 80% train / 20% validation, stratified on `is_churn`, `random_state=42`.
- **Class imbalance:** ~9% churn; tree models use `class_weight='balanced'` where applicable (same spirit as v2).
- **Models compared in the notebook:** Random Forest (v3 feature set, depth 10), deeper Random Forest (depth 15, 200 trees), **LightGBM** (500 estimators, learning rate 0.05, max depth 6, early stopping on validation log-loss). Earlier iterations (LR, shallow trees, v2-only RF) are kept in the comparison table for context.

---

## Results (validation set)

All numbers below come from the notebook outputs on that single stratified split.

### ROC AUC — main comparison

| Model                                    | ROC AUC    |
| ---------------------------------------- | ---------- |
| **LightGBM** (n=500, lr=0.05, depth=6)   | **0.7761** |
| Random Forest — deeper (depth=15, n=200) | 0.7675     |
| Random Forest — v3 (depth=10, balanced)  | 0.7651     |
| Random Forest — v2 only (depth=10)       | 0.6727     |
| Logistic Regression — v1 features        | 0.5673     |

**Lift vs. v2 best RF:** 0.7761 vs. 0.6727 → about **+10.3 points** absolute AUC (~**+15.3%** relative to 0.6727).

### Precision / recall at default threshold 0.5 (LightGBM)

| Metric            | v2 RF (0.6727 AUC) | LightGBM v3 |
| ----------------- | ------------------ | ----------- |
| Recall (churn)    | 0.4875             | **0.5696**  |
| Precision (churn) | 0.1611             | **0.2842**  |

So at the same 0.5 cutoff, recall and precision both improve materially—not only ranking (AUC) but also the operating point used by default `evaluate()`.

---

## Threshold tuning (LightGBM, validation probabilities)

| Threshold | Precision | Recall | F1     | Users flagged as churn |
| --------- | --------- | ------ | ------ | ---------------------- |
| 0.5       | 0.2842    | 0.5696 | 0.3792 | 35,001                 |
| 0.4       | 0.1941    | 0.6820 | 0.3022 | 61,379                 |
| 0.3       | 0.1128    | 0.9124 | 0.2007 | 141,311                |
| 0.2       | 0.0963    | 0.9829 | 0.1754 | 178,253                |
| 0.1       | 0.0905    | 0.9991 | 0.1660 | 192,777                |

**Practical read:** F1 peaks at **0.5** on this table; if the business goal is to catch almost all churners and cheap interventions are acceptable, **0.3–0.4** gives recall **0.68–0.91** at the cost of many false positives (precision **0.11–0.19**). There is no single “best” threshold without a cost matrix.

---

## Feature importance (deeper Random Forest on v3 features)

The notebook uses the **deeper RF** (`rf_deep`) for the importance plot. Top features:

1. **`days_since_last_log`** (~22.7% of total importance) — log recency dominates.
2. **`days_since_last_txn`** — transaction recency still strong, now second.
3. **`has_recent_activity`**, **`days_since_registration`**, **`txn_count`**, **`txn_per_day`**, etc.

**Group sums (RF importance):** logs **54.2%**, baseline (v1) **24.2%**, v2 transaction extensions **21.5%**. Listening aggregates contribute more than half of the model’s split gain in this forest, which supports the iteration hypothesis.

`auto_renew_last` remains in the mix (~2.0% importance in this run) but is no longer the single dominant signal the way it was under logistic regression in v1.

---

## Key takeaways

1. **Engagement features carry strong signal** when combined with subscription history; log recency and activity flags rank at the top.
2. **Gradient boosting (LightGBM) edges out RF** on the same feature matrix on this split—worth keeping in the stack for tabular churn.
3. **Precision improved alongside AUC**, which is not automatic when adding features; the model is less aggressively “always predict majority” at 0.5 than the v2 RF was.

---

## Limitations

- **Single train/validation split** — metrics have sampling variance; no k-fold CV yet.
- **No temporal decay** inside the log window — all March days weighted equally in the aggregates.
- **Aggregates only** — no sequence or session-level patterns, no cross-feature interactions beyond what trees learn.
- **LightGBM is optional** — the notebook guards imports; reproducibility requires installing `lightgbm` in the environment.
- **Competition gap** — public leaderboards for this challenge are much higher (~0.89–0.91 AUC); this repo is a learning track, not a full leaderboard submission pipeline.

---

## Next steps

- [ ] **Cross-validation** for stable AUC and threshold choice.
- [ ] **Time-decayed** log aggregates (recent days weigh more).
- [ ] **Explicit interaction** features (e.g. low logs × auto-renew off).
- [ ] **Calibration** (Platt / isotonic) if probabilities are used for budgeting.
- [ ] **Test-set predictions** and Kaggle-format submission once happy with CV.

---

## How to reproduce

1. Build `df_model_baseline_v1.csv` via `02_eda_transactions_features.ipynb` (and labels via the train pipeline in `04_baseline_model.ipynb` as documented in `baseline_v1.md`).
2. Run `03_eda_user_logs.ipynb` to produce `data/processed/log_features_march2017.csv`.
3. Run `06_iteration_v3_user_logs.ipynb` with **pandas**, **scikit-learn**, **matplotlib**, **seaborn**, and optionally **lightgbm** installed.
