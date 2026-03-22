# Iteration: baseline_v1

**Date:** 2026-03  
**Status:** Complete  
**Notebook:** `notebooks/04_baseline_model.ipynb`  
**Dataset produced:** `data/processed/df_model_baseline_v1.csv`

---

## Objective

Establish a first working baseline for the churn prediction task. The goal was not to maximize performance but to:

1. Validate the data pipeline end to end (load → merge → features → model → evaluate).
2. Understand the data structure, class distribution, and basic feature signal.
3. Set a concrete metric floor (ROC AUC) to beat in future iterations.

---

## Datasets Used

| File | Role |
|---|---|
| `data/raw/train_v2.csv` | Labels: `msno` + `is_churn` target |
| `data/raw/members_v3.csv` | User demographics: `registration_init_time`, `bd`, `gender`, `city` |
| `data/raw/transactions_v2.csv` | Transaction history: subscription events, payment amounts, cancellation flags |

User logs (`user_logs_v2.csv`) were explored in `03_eda_user_logs.ipynb` but not used in this baseline.

---

## Temporal Cutoff

**Cutoff date: `2017-02-28`**

Transactions were filtered to include only records where `transaction_date <= 2017-02-28`.  
This prevents data leakage: the churn label corresponds to March 2017 behavior, so any transaction data from March 2017 or later would constitute future information bleeding into training features.

```python
cutoff = '2017-02-28'
transactions_before_cutoff = transactions_base[
    transactions_base['transaction_date'] <= cutoff
]
```

---

## Feature Engineering

All features were derived from `transactions_v2` (pre-cutoff) and `members_v3`.

### Transaction features (aggregated per user)

| Feature | Description | Aggregation |
|---|---|---|
| `txn_count` | Number of transactions before cutoff | count |
| `has_txn_history` | Whether the user had any transaction before cutoff | binary flag |
| `auto_renew_last` | Auto-renew flag on the user's last transaction | last value |
| `payment_plan_days_last` | Plan duration in days on last transaction | last value |
| `plan_list_price_last` | Listed plan price on last transaction | last value |
| `actual_amount_paid_last` | Actual amount paid on last transaction | last value |
| `cancel_count` | Number of cancellation events | count |
| `has_cancelled` | Whether the user ever cancelled | binary flag |

### Member features

| Feature | Description |
|---|---|
| `days_since_registration` | Days between `registration_init_time` and `2017-02-28` |

### Notes on preprocessing

- Missing transaction features (`txn_count`, `cancel_count`, `has_cancelled`) were filled with `0` — these users had no transaction history.
- Invalid ages (`bd < 0` or `bd > 100`) were set to `NaN`.
- Missing `gender` was filled with `'unknown'`.
- Remaining NaN features were filled with `0` before model training.

---

## EDA Highlights

### Train + Members (notebooks/01, 02)

- **Total users:** 970,960 (from `train_v2.csv`)
- **Churn rate:** ~9% positive class (`is_churn = 1`)
- **Class imbalance:** severe — ~91% no-churn, ~9% churn
- `registration_init_time` spans a wide range; `days_since_registration` captures account tenure
- `bd` (age) had many invalid values (negatives, 0, >100) requiring cleaning
- Most users had at least one transaction record before the cutoff

### Transactions (notebook/02)

- `transactions_v2.csv` covers transactions up to ~mid-2017
- Strong signal: users who have no transaction history are far more likely to churn
- `auto_renew_last = 1` is a strong negative predictor of churn (renewing users stay)
- Cancel events are relatively rare but informative

### User Logs (notebook/03)

- `user_logs_v2.csv` covers March 2017 only — this is post-cutoff listening behavior
- **Not used in this baseline** to avoid leakage
- Will be considered in future iterations using pre-cutoff user logs

---

## Model

**Algorithm:** Logistic Regression  
**Library:** `scikit-learn`  
**Hyperparameters:** `max_iter=1000`, all other defaults

### Train/Validation Split

- 80/20 split, stratified by `is_churn`
- Validation set: 194,192 rows (~18,000 churners)

---

## Results

### Core Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.914 |
| ROC AUC | **0.567** |
| Precision (churn) | 0.781 |
| Recall (churn) | 0.065 |
| F1-score (churn) | 0.120 |

### Threshold Analysis

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.5 | 0.781 | 0.065 | 0.120 |
| 0.4 | 0.702 | 0.081 | 0.146 |
| 0.3 | 0.676 | 0.086 | 0.152 |
| 0.2 | 0.642 | 0.104 | 0.179 |
| 0.1 | 0.172 | 0.181 | 0.177 |

Lowering the threshold improves recall at the cost of precision. F1 peaks around 0.2 but remains very low throughout — the model has poor discriminative power for the minority class.

### Feature Coefficients

| Feature | Coefficient | Interpretation |
|---|---|---|
| `has_txn_history` | +4.10 | No history → strong churn signal |
| `has_cancelled` | +2.17 | Ever cancelled → more likely to churn |
| `payment_plan_days_last` | +0.21 | Longer plans → slightly more likely to churn |
| `txn_count` | +0.08 | More transactions → slightly more likely to churn |
| `actual_amount_paid_last` | +0.004 | Negligible |
| `days_since_registration` | +0.00003 | Negligible |
| `plan_list_price_last` | -0.045 | Higher priced plans → slightly less likely to churn |
| `cancel_count` | -1.16 | Counterintuitively negative — likely collinear with `has_cancelled` |
| `auto_renew_last` | -3.93 | Auto-renew enabled → strong retention signal |

---

## Limitations

1. **ROC AUC of 0.567 is barely above random (0.5).** The model has very limited discriminative power.
2. **Class imbalance is not addressed.** No resampling, no `class_weight='balanced'`. The model overwhelmingly predicts no-churn.
3. **Features are sparse.** Only 9 features, all from transactions. No listening behavior, no plan change history, no temporal trends.
4. **No scaling applied.** Logistic Regression is sensitive to feature scale; features like `actual_amount_paid_last` and `days_since_registration` are on very different scales than binary flags.
5. **User logs excluded.** Pre-cutoff listening data (`user_logs.csv`) could provide strong behavioral signals.
6. **No cross-validation.** Single train/val split — no confidence intervals on metrics.
7. **Linear model.** Interaction effects and non-linear relationships are not captured.

---

## Next Steps

### Immediate improvements (v2)

- [ ] Apply `class_weight='balanced'` or use SMOTE/undersampling
- [ ] Normalize/scale numerical features before Logistic Regression
- [ ] Add pre-cutoff user log features (total listening time, unique tracks, active days)
- [ ] Enrich transaction features: days since last transaction, plan change frequency, gap between renewal and expiry

### Model upgrades (v3+)

- [ ] Random Forest or Gradient Boosting (LightGBM / XGBoost)
- [ ] Proper k-fold cross-validation
- [ ] Hyperparameter search
- [ ] Target metric: ROC AUC > 0.85 (competition top entries reached ~0.89–0.91)

### Engineering

- [ ] Migrate feature engineering logic from notebook to `src/features.py`
- [ ] Add a reproducible pipeline script (`src/pipeline.py`)
