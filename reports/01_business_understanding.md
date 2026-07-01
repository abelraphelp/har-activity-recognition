# Phase 1 — Business Understanding

*CRISP-DM Phase 1. This document is written before any data is loaded: it defines
the problem, its relevance, and — critically — how success will be measured.*

## 1. The problem

Given a short window of body-worn motion-sensor signals (tri-axial accelerometer +
gyroscope), **classify which of six physical activities** a person is performing:

`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`.

This is a **supervised, multi-class classification** task. It is *activity
recognition* (what is happening now), not *event detection* (e.g. did a fall occur) —
a related but distinct problem. HAR is the foundational layer that event-detection
systems are built on.

## 2. Why it matters

The project is motivated by assistive-systems research (elderly care, rehabilitation,
healthy ageing). HAR underpins:

- **Fall detection & elderly care** — abnormal events can only be flagged once normal
  activity is reliably recognised.
- **Rehabilitation tracking** — verifying a patient performs prescribed movement
  (walking, stair climbing) and measuring change over recovery.
- **Healthy ageing / behavioural monitoring** — detecting long-term trends such as
  increasing sedentary behaviour.

**Scope honesty:** the UCI HAR dataset contains only these six ordinary activities —
no falls or adverse events. This project therefore demonstrates the *methodology and
foundation* that real assistive systems require, not fall detection itself.

## 3. How we define success

Overall accuracy is **not** the primary metric. A model can score high on accuracy
while failing on an activity that matters for the application. Errors do not carry
equal cost across activities, so we evaluate **per-class**.

### 3.1 Per-class recall targets (tiered by clinical relevance)

| Activity            | Recall target | Reason |
|---------------------|:-------------:|--------|
| WALKING             | ≥ 0.90        | Core daily mobility; missing it under-counts a person's activity level. |
| WALKING_UPSTAIRS    | ≥ 0.90        | Stairs demand balance/effort; clinically important, common fall location. |
| WALKING_DOWNSTAIRS  | ≥ 0.90        | High relevance for balance and fall risk. |
| STANDING            | ≥ 0.88        | Upright posture; confusing with sitting/laying affects mobility monitoring. |
| SITTING             | ≥ 0.85        | Physically hard to separate from standing (see 3.4). |
| LAYING              | ≥ 0.85        | Usually easiest (distinct device orientation). |

**Recall** = of all times the person was truly doing activity X, the fraction we
caught. It is the "did we miss it?" metric — the one that matters when a miss has a
cost.

### 3.2 Headline / model-comparison targets

| Metric | Target | Rationale |
|--------|:------:|-----------|
| Baseline (Random Forest / SVM on features) accuracy | ≥ 0.90 | Reliably achievable; below this signals a problem. |
| Deep model (CNN/LSTM on raw signals) accuracy | ≥ 0.92 | Should ideally beat the baseline — if it can't, that is itself a finding. |
| Macro-averaged F1 | reported | Unweighted mean across classes: treats every activity equally regardless of frequency. Formalises the "no blind spots" fairness criterion. |

### 3.3 The confusion matrix is a primary tool, not an afterthought

We distinguish **expected** confusions from **pathological** ones:

- *Expected:* `SITTING ↔ STANDING`, `WALKING_UPSTAIRS ↔ WALKING_DOWNSTAIRS`.
- *Pathological (would indicate a real problem):* e.g. `LAYING ↔ WALKING` — physically
  dissimilar activities should never be confused.

### 3.4 Why the hard confusions are hard (physical explanation)

- **SITTING ↔ STANDING:** with a waist-mounted sensor and no motion, the accelerometer
  mainly measures the **gravity vector**. Both postures keep the trunk vertical with
  near-zero dynamic motion, so the signals are almost identical — an *information* limit
  of the sensor, not merely a model weakness.
- **UPSTAIRS ↔ DOWNSTAIRS:** both are periodic gait at similar cadence, differing mainly
  in the **sign and energy of vertical acceleration** (raising vs. lowering the body).

### 3.5 Precision matters too (error direction)

Recall guards against *missing* activity (under-counting). Precision guards against
*false positives* — e.g. calling something "walking" when it is not — which, in a
downstream alerting system, causes **alert fatigue**. Recall leads for this project,
but precision on the walking classes will also be monitored.

## 4. Methodological guardrail

These targets are **pre-registered here, before any results are seen.** The UCI test
set contains *different subjects* than the training set (evaluated in Phase 3), and it
will be scored **once, at the end**. The model is never tuned by peeking at test
performance — setting the bar before the experiment is what makes the final numbers
trustworthy on new, unseen people.

## 5. Limitations seed (revisited in Phase 6)

- No fall/adverse-event class: "laying" could mean *resting* or *fallen and unable to
  rise* — this dataset cannot distinguish them.
- Data is collected in controlled lab conditions with a fixed device placement;
  real-world performance will differ.
