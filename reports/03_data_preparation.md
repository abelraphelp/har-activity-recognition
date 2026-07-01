# Phase 3 — Data Preparation

*CRISP-DM Phase 3. Turning raw files into model-ready inputs and locking down an
honest evaluation setup. Analysis: [`notebooks/01_har_analysis.ipynb`](../notebooks/01_har_analysis.ipynb).*

## Two model-ready inputs

| Representation | Train | Test | Used by |
|---|---|---|---|
| ① Pre-computed features | `X_train` (7352, 561) | `X_test` (2947, 561) | Baseline (Random Forest / SVM) |
| ② Raw inertial signals | `X_train_raw` (7352, 128, 9) | `X_test_raw` (2947, 128, 9) | Deep model (CNN / LSTM) |

Labels: `y_train` (7352), `y_test` (2947), values 1–6 mapped to activity names.

## Subject-independent split — proven, not assumed

The train (21) and test (9) subject sets were checked for overlap programmatically:

```
Overlap (people in BOTH sets): set()   -> PASSED
```

An `assert` guards against **data leakage**: if any person appeared in both sets the
notebook would fail loudly. This guarantees the model is evaluated on **people it has
never seen**, so reported performance reflects generalization to new users — not
memorization of individuals.

## Scaling decisions

- **Features (①):** already normalized by the dataset authors to **[-1, 1]**
  (verified in Phase 2). Random Forest is scale-invariant; Logistic Regression / SVM
  are scale-sensitive but the features are already bounded — so **no additional scaling
  is applied** to the baseline inputs.
- **Raw signals (②):** in physical units, range ≈ **[-6, 6]**, differing per channel.
  These will be **standardized per channel** (subtract mean, divide by std) for the
  neural network. The mean/std are computed **from the training set only** and applied
  to both train and test (no leakage). Implemented at the start of Phase 4.

## Validation strategy (Phase 4)

The 9-subject test set is touched **once, at the very end**. For tuning the deep model,
a **subject-wise validation split** will be carved out of the 21 training subjects
(hold out a few whole subjects), so hyper-parameter choices never peek at the test set.

## Status

Both representations are prepared and the evaluation protocol is fixed. Ready for
Phase 4 — Modeling.
