# Iteration v2 — Feature engineering + tree models

**Date:** March 2026  
**Status:** Complete  
**Notebook:** `notebooks/05_iteration_v2_features_and_trees.ipynb`  
**Input dataset:** `data/processed/df_model_baseline_v1.csv`

---

## Objective

Improve ROC AUC over baseline v1 (0.567) on two fronts at once:

1. **New features** — use columns that already existed in the dataset but were not yet in the model.
2. **Non-linear models** — move beyond logistic regression to decision trees and random forests.

---

## Features used

### Carried over from baseline v1

| Feature                   | Description                              |
| ------------------------- | ---------------------------------------- |
| `txn_count`               | Number of transactions before the cutoff |
| `has_txn_history`         | At least one transaction before cutoff   |
| `auto_renew_last`         | Auto-renew on the last transaction       |
| `payment_plan_days_last`  | Plan length (days) on last transaction   |
| `plan_list_price_last`    | List price on last transaction           |
| `actual_amount_paid_last` | Amount paid on last transaction          |
| `cancel_count`            | Historical cancellation count            |
| `has_cancelled`           | Ever cancelled                           |
| `days_since_registration` | Days from registration to cutoff         |

### New features (v2)

| Feature               | Concept              | Construction                                                   |
| --------------------- | -------------------- | -------------------------------------------------------------- |
| `days_since_last_txn` | **Recency**          | `CUTOFF - txn_last_date` in days; no history → max + 1         |
| `has_recent_txn`      | Recency flag         | `days_since_last_txn < 30`                                     |
| `txn_per_day`         | **Frequency**        | `txn_count / (days_between_first_last_txn + 1)`                |
| `avg_payment`         | Financial engagement | `actual_amount_paid_median`                                    |
| `price_discount`      | Perceived discount   | `plan_list_price_last - actual_amount_paid_last`, clipped at 0 |
| `plan_consistency`    | Plan stability       | `payment_plan_days_last == payment_plan_days_median`           |

---

## Models trained

Same split for all: 80% train / 20% validation, stratified on `is_churn`.  
**All reported metrics are on the validation set.**

| Model                    | Configuration                                                 |
| ------------------------ | ------------------------------------------------------------- |
| Logistic Regression (v2) | `max_iter=1000` — internal reference with new features        |
| Decision Tree            | `max_depth=5`                                                 |
| Random Forest            | `n_estimators=100`, `max_depth=10`, `class_weight='balanced'` |

---

## Results

```
Model                                      ROC AUC  Accuracy  Precision  Recall    F1
Random Forest (n=100, depth=10, balanced)   0.6727    0.7256     0.1611  0.4875  0.2421
Decision Tree (max_depth=5)                 0.6038    0.9200     0.7674  0.1588  0.2632
Logistic Regression (baseline v1)           0.5673    0.9143     0.7808  0.0649  0.1198
Logistic Regression (features v2)         0.5635    0.9147     0.7900  0.0704  0.1293
```

**Best model:** Random Forest with ROC AUC **0.6727** (~**+18.5%** relative to baseline v1).

---

## Feature importance — Random Forest

```
Feature                    Importance
days_since_last_txn           24.5%   ← new, #1
days_since_registration       24.0%
txn_count                     14.0%
txn_per_day                   11.2%   ← new
avg_payment                    6.1%   ← new
actual_amount_paid_last        4.5%
plan_list_price_last           4.0%
payment_plan_days_last         3.7%
auto_renew_last                3.3%
has_txn_history                2.6%
has_recent_txn                 1.2%   ← new
plan_consistency               0.6%   ← new
cancel_count                   0.2%
has_cancelled                  0.1%
price_discount                 0.04%  ← new
```

Recency (`days_since_last_txn`) became the top feature, matching the intuition that inactive subscribers are more likely to churn.

---

## Threshold analysis — Random Forest

```
threshold  precision  recall    f1
     0.5     0.1611  0.4875  0.2421
     0.4     0.1088  0.8435  0.1927
     0.3     0.0949  0.9785  0.1729
     0.2     0.0922  0.9926  0.1688
     0.1     0.0900  1.0000  0.1651
```

At threshold **0.4**, recall on churners is **~84%**. The tradeoff is many false positives (roughly eight non-churners flagged per true churner at that setting). For low-cost retention nudges, that may still be acceptable.

---

## Main findings

**1. Recency is the strongest feature.**  
`days_since_last_txn` captures the “drifting away” pattern: long gaps since the last paid period line up with churn. The forest assigned it ~24.5% of total importance.

**2. Trees spread importance; logistic regression did not.**  
In v1 LR, `auto_renew_last` and `has_txn_history` dominated (coefficients around -3.93 and +4.10). In the RF, those drop to roughly 9th and 10th by importance. The forest uses interactions and non-linear splits across many variables.

**3. New features did not help plain logistic regression.**  
LR with v2 features reached ROC AUC **0.5635**, slightly _below_ v1 (0.5673). Likely causes: very different scales on new numeric columns without scaling, plus a `ConvergenceWarning` at 1000 iterations.

**4. `class_weight='balanced'` changes the operating point.**  
Without reweighting, the RF would lean heavily on the ~91% majority class. Balanced weights make the minority class matter in the loss, which explains recall ~0.49 at 0.5 threshold vs. ~0.07 on the baseline LR.

---

## Limitations

1. **ROC AUC ~0.67 is still modest** compared with strong competition solutions (~0.89–0.91).
2. **No user logs** in v2 — listening behaviour was still unused.
3. **No cross-validation** — one split only.
4. **Low precision at 0.5** (~0.16) — many false churn flags.
5. **LR without scaling** — `StandardScaler` was not tried for a fair linear comparison.

---

## Next steps (from v2; several done in v3)

- [x] Add **user log** aggregates and retrain (see `v3_user_logs.md`).
- [x] Try **LightGBM** on the expanded feature set (notebook 06).
- [ ] Apply **StandardScaler** to LR and recompare.
- [ ] **K-fold cross-validation** for more stable metrics.
- [ ] **SHAP** or similar for deeper interpretation of the best tree model.
