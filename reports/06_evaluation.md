# Phase 5 — Evaluation

*CRISP-DM Phase 5. Consolidated comparison of all models, robustness analysis, and the
key findings. Analysis: [`notebooks/02_modeling.ipynb`](../notebooks/02_modeling.ipynb).*

## Model comparison (fixed subject-independent split)

| Model | Test accuracy | Macro-F1 |
|---|---:|---:|
| Random Forest (561 features) | 92.87% | 0.927 |
| 1-D CNN (raw signals) | 93.35% | 0.934 |
| **Ensemble (RF + CNN, soft vote)** | **94.91%** | **0.950** |

The **ensemble is the best model**, improving ~2 points over the baseline.

## Per-class recall vs. Phase-1 targets

![Per-class recall by model](figures/model_recall_comparison.png)

| Activity | RF | CNN | Ensemble | Target | Best meets? |
|---|---:|---:|---:|---:|:--:|
| WALKING | 0.976 | 0.978 | 0.992 | ≥0.90 | ✅ |
| WALKING_UPSTAIRS | 0.915 | 0.962 | 0.975 | ≥0.90 | ✅ |
| WALKING_DOWNSTAIRS | 0.857 | 0.974 | 0.993 | ≥0.90 | ✅ |
| STANDING | 0.921 | 0.921 | 0.940 | ≥0.88 | ✅ |
| SITTING | 0.886 | 0.823 | 0.833 | ≥0.85 | ❌ |
| LAYING | 1.000 | 0.950 | 0.968 | ≥0.85 | ✅ |

The best model (ensemble) meets **5 of 6** targets. **SITTING (0.833) is the sole miss.**

![Ensemble confusion matrix](figures/ensemble_confusion_matrix.png)

## Key findings

1. **Complementary specialists.** The RF (hand-crafted features) is a *posture specialist*;
   the CNN (raw signals) is a *motion specialist*. The CNN lifted WALKING_DOWNSTAIRS recall
   from 0.857 → 0.974 by learning the ascent/descent temporal asymmetry; the RF kept the
   static postures cleaner. Combining them beat either alone.

2. **Ensembling fixes complementary errors, not shared ones.** DOWNSTAIRS improved because
   the models *disagreed* (confident CNN overruled weak RF). SITTING did **not** recover
   because **both** models make the *same* mistake (confusing it with STANDING), so
   averaging reinforces rather than corrects it.

3. **SITTING↔STANDING is a fundamental sensor limit, not a modeling flaw.** Two motionless
   upright postures produce near-identical gravity signatures on a waist sensor (shown in
   Phase 2). No model here exceeds ~0.89 recall on SITTING — the information simply is not
   in the signal. This was predicted from physics in Phase 1.

4. **Error structure is benign.** All models are essentially block-diagonal (motion vs.
   static never confused), with one minor exception: the CNN/ensemble introduce a small
   LAYING→UPSTAIRS leak (~3–5%) absent from the RF.

## Robustness: subject-wise cross-validation

To check the fixed-split result is not an artifact of one particular test group, a
**subject-wise 5-fold GroupKFold** CV (all 30 subjects; each subject confined to one
fold — no leakage) was run on the Random Forest:

- **Accuracy: 0.9382 ± 0.0224** | **Macro-F1: 0.9356 ± 0.0237**
- Per-fold accuracy ranged **0.908 – 0.966** (~6-point spread depending on which people
  are held out).

The fixed-split RF result (0.9287) falls within this range — slightly below the mean,
indicating the official test subjects are marginally harder than average. **Conclusion:
the reported results are representative, and HAR performance is meaningfully
subject-dependent** (±2.2% from subject composition alone).

*Note:* only **subject-wise** CV is valid here. Ordinary random k-fold would place windows
from the same person in both train and validation folds, causing subject leakage and
inflated scores.
