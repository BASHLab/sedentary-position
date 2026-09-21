# Posture baseline — dataset, features, models

Companion to `scripts/posture_baseline.py`. Everything here describes the
windowed dataset in `analysis/posture/windows_10s.parquet` and the five models
scored in `analysis/posture/baseline_loso_*.csv`.

---

## 1. Dataset

### Source

**LittleBeats**, a chest-worn infant recorder carried in a shirt pocket, logging
audio (16 kHz), ECG (1000 Hz) and a 9-channel IMU (70 Hz) simultaneously.
Cohort **BRP1**, at `/work/hdd/bebr/Data/LB/cleaned/BRP1/`, organised
`<family_id>/{Visit,Home}/<session>/`. 182 sessions across 11 infants; every
session carries an `_imu_sync.txt`.

The chest placement matters more than any other single fact about this data:
trunk orientation relative to gravity *is* most of the posture signal. Measured
across the corpus, accelerometer magnitude sits at **9.74 ± 0.39 m/s²** — the
dynamic component is roughly 0.04 g. This is a near-static orientation problem,
not the dynamic activity recognition that wrist-worn HAR corpora contain.

### IMU channel layout

Channels are unlabelled in the raw files; these assignments were established
from per-channel magnitude statistics, not documentation.

| channels | signal | units | evidence |
|---|---|---|---|
| 0–2 | accelerometer | m/s² | per-sample magnitude 9.74 ± 0.39 |
| 3–5 | magnetometer | ~Gauss | magnitude ≈ 0.90, near-constant |
| 6–8 | gyroscope | deg/s | range ±323, mean magnitude 18.2 |

**The magnetometer is deliberately excluded from every model.** Indoor
hard/soft-iron distortion varies by room and by home, so it behaves as a
subject- and location-identity feature — precisely the thing leave-one-subject-out
is meant to guard against.

### Labels

Human behavioural coding from synchronised GoPro video, exported from ELAN as
tab-delimited files (`*_EDITED.txt`, 9 columns, one row per annotation) in
`/work/hdd/bebr/Data/Behavioral_Coding/BRP1_EMAannotations/`. 88 files, 14,266
annotations, 11 tiers; this work uses the `posture` tier only — 2,525 segments,
75.83 h, across 6 infants (25003–25008). The remaining 5 infants (1101–1104,
25001) have recordings but no annotations: no video existed to code against.

### Clock alignment

The annotation clock starts at a GoPro tone, **not** at the start of the LB
recording. `BRP1_EMAfiles_ToneTimes_tracking.xlsx` supplies the offset per
observation, so `LB_time = annotation_time + LBoffset_tonetime_secs`. Seven
observations have no tone and cannot be placed on the IMU clock at all, leaving
**71.69 h of the 75.83 h (94.5%) alignable**. Two further observations have
offsets that overrun their recording by more than an hour and should be treated
as mis-mapped — see the README's alignment-problems table.

### Windowing

10 s windows, 5 s hop, keeping a window only when a single posture label covers
**≥80%** of it — so windows straddling a posture transition are discarded rather
than given an arbitrary label. `recline forward` is dropped (3 segments in the
whole corpus). Result: **48,135 windows, 66.9 h, 6 infants, 7 classes.**

Because the hop is half the window, adjacent windows share half their samples.
Independent time is 66.9 h; do not read 48,135 as 48,135 independent
observations. Overlap is within-subject, so it cannot leak across folds.

| class | windows | share | | class | windows | share |
|---|---|---|---|---|---|---|
| on back | 12,628 | 26.2% | | stand | 3,931 | 8.2% |
| on stomach | 11,345 | 23.6% | | on side | 2,588 | 5.4% |
| recline back | 10,890 | 22.6% | | crawl | 511 | 1.1% |
| sit | 6,242 | 13.0% | | | | |

Class support is very unevenly distributed *across infants*, which is the
dominant limitation of this dataset: `crawl` occurs in only 3 of 6 infants, and
25004 contributes 27 `on back` windows against 2,236 `stand` while every other
infant is back/stomach-dominant.

---

## 2. Features

19 features per window, in three nested sets. `acc` and `gyr` are the (3, 700)
accelerometer and gyroscope slices; `amag`/`gmag` are their per-sample vector
magnitudes.

### Set A — orientation (3 features)

The mean acceleration vector over the window, normalised to unit length. With a
near-static chest sensor this *is* the gravity direction in sensor coordinates,
i.e. which way the trunk is pointing. Normalising discards magnitude so the
feature encodes direction only.

| feature | description |
|---|---|
| `grav_x`, `grav_y`, `grav_z` | unit-normalised mean accel vector; dimensionless, ‖·‖ = 1 |

### Set B — A + linear-acceleration magnitude and dispersion (8 more, 11 total)

Separates *how still* the infant is from *which way they face*. A supine infant
and a supine-but-kicking infant share set A and differ here.

| feature | description |
|---|---|
| `amag_mean` | mean of ‖acc‖; ≈ 9.81 when still, departs under acceleration |
| `amag_std` | SD of ‖acc‖ — overall movement intensity |
| `amean_norm` | ‖mean acc‖ **before** normalising. A stillness measure: ≈ 9.81 when the orientation holds, collapses toward 0 when orientation changes within the window |
| `ax_std`, `ay_std`, `az_std` | per-axis SD — movement intensity, with direction |
| `amag_jerk` | mean absolute first difference of ‖acc‖ — high-frequency jitter vs. smooth motion |
| `amag_ptp` | peak-to-peak range of ‖acc‖ — sensitive to single impacts a SD would average away |

### Set C — B + angular rate (8 more, 19 total)

Rotation that linear acceleration barely registers: a seated infant twisting,
the rocking of supported standing, the limb-driven rotation of crawling.

| feature | description |
|---|---|
| `gmag_mean`, `gmag_std`, `gmag_max` | mean / SD / max of ‖gyr‖ — overall, variability, and peak rotation rate |
| `gx_absmean`, `gy_absmean`, `gz_absmean` | mean absolute rate per axis — which rotation axis dominates |
| `gx_std`, `gy_std` | per-axis rate variability |

> **Known wart:** `gz_std` is absent. The code takes `gyr.std(axis=1)[:2]`,
> which was an arbitrary truncation to keep set C at 8 features, not a
> considered exclusion. Adding it is a one-character change and worth doing
> before anyone builds on these features.

---

## 3. Models

Five configurations — two model classes over the nested feature sets, so the
feature effect and the model-class effect can be read separately.

| # | features | classifier | description |
|---|---|---|---|
| 1 | A (3) | logistic regression | The floor. Multinomial, L2, `C=1.0`, `max_iter=2000`, on standardised inputs. Linear decision boundaries in gravity-direction space — answers "how far does a purely linear read of trunk orientation get you?" |
| 2 | B (11) | logistic regression | As above with movement-intensity features. Isolates what stillness/dispersion adds under a linear model. |
| 3 | C (19) | logistic regression | As above plus angular rate. |
| 4 | A (3) | gradient boosting | `HistGradientBoostingClassifier`, `max_iter=300`, `random_state=0`, defaults otherwise. Same 3 inputs as #1, so the gap is purely non-linearity — orientation classes occupy curved regions of the unit sphere that a linear boundary cannot carve. |
| 5 | C (19) | gradient boosting | Best performer. All 19 features, non-linear. |

Reference point: predicting the majority class (`on back`) always gives 0.262
accuracy, 0.059 macro-F1.

### harnet10 — the pretrained encoder

**harnet10** is the 10-second variant of Oxford's `ssl-wearables` family: a 1D
ResNet-V2 (11.0 M parameters, 10.5 M of them in the feature extractor)
self-supervised on roughly 700,000 person-days of unlabelled UK Biobank wrist
accelerometry via multi-task transformation prediction, then released as a
transfer backbone for human activity recognition. Its native input happens to be
exactly our window — 3-channel accelerometer in g at 30 Hz for 10 s, `(B, 3,
300)` — so `build_raw_windows.py` resamples the same 48,135 windows from 70 Hz
to 30 Hz and converts m/s² to g, and the deep model is scored on the identical
rows, in the identical order, under the identical leave-one-subject-out split as
the GBDT. Two transfer regimes answer different questions: **probe** freezes the
encoder and fits a linear head (is the pretrained representation *already*
linearly separable for infant trunk posture?) and **full** trains everything at
`lr=1e-4` (what does transfer buy at best?). Epoch selection uses a 10% random
split of the training windows — subject-overlapping, so mildly optimistic for
picking an epoch, but never touching the held-out infant. The result is a
domain-shift story: the frozen representation is clearly *worse* than 19
hand-built features (probe: 0.740 accuracy, 0.607 macro-F1, κ 0.673 — below even
the 3-feature gravity GBDT), because UK Biobank is adult wrist-worn dynamic
activity and this is infant chest-worn near-static orientation. Fine-tuned, it
edges past the baseline (**0.860 accuracy, 0.751 macro-F1, κ 0.824** vs. 0.854 /
0.729 / 0.817 for C+GBDT) — a real but small gain, which says the pretrained
weights are a useful initialisation rather than a useful representation.

### Evaluation

**Leave-one-subject-out, grouped by `family_id`** — 6 folds, each holding out one
infant entirely. There is no fixed test set: every window is predicted exactly
once while its infant is held out, and the pooled 48,135 predictions form the
reported test set. Fold sizes run 6,165–9,760 windows (13–20%).

Grouping by infant is not optional here. With 6 subjects and overlapping windows,
a random split would place windows sharing raw samples on both sides and let any
model score subject identity rather than posture.

Three metrics, because they disagree in informative ways:

- **accuracy** — flattered by the 26% majority class; reported for continuity only.
- **macro-F1** — unweighted mean over classes, so `crawl` (1.1% of windows)
  carries the same weight as `on back`. The metric to use for "does this work on
  every posture".
- **Cohen's κ** — chance-corrected agreement, weighted by prevalence. The metric
  to use for "does this agree with human coders well enough to be usable", and
  the convention in behavioural coding. Computed with `labels=` pinned to all 7
  classes so folds missing a class are not scored against a smaller matrix.

`kappa` in the results sheet is computed on pooled out-of-fold predictions;
`mean_fold_kappa` averages the six folds. They differ (0.817 vs 0.802) because
fold sizes and class priors differ by infant.

### Why macro-F1 always lands below κ

Every model in these tables reports a macro-F1 well under its κ — harnet10
(full) is 0.751 against 0.824, C+GBDT is 0.729 against 0.817. That is not a
contradiction; the two metrics are answering different questions, and there is
also one avoidable artefact mixed in. `scripts/harnet_confusion.py` separates
them.

**1 — the class-weighting difference (the real effect).** κ is chance-corrected
*accuracy*: with p_o = 0.860 and p_e = 0.205, κ = (0.860 − 0.205)/(1 − 0.205) =
0.824. Accuracy is carried by whatever is common, and `on back` + `on stomach` +
`recline back` are 72.4% of all windows at a mean F1 of 0.943. macro-F1 divides
its budget evenly over seven classes, so `crawl` (1.1% of windows, F1 0.462) and
`stand` (8.2%, F1 0.407) each count for a seventh. The same seven per-class F1
values give 0.751 averaged flat and 0.854 averaged by support — the whole gap
between macro-F1 and κ is that re-weighting. Read this way the two numbers are
a useful pair: **κ says the model agrees with the coders about as well as a
second coder would; macro-F1 says it still cannot do the upright postures.**

**2 — the macro-F1 divisor (an artefact worth fixing).** `f1_score(...,
average="macro")` with no `labels=` averages over the classes present in
`y_true ∪ y_pred`. The model predicts all 7 classes in every fold, so the
divisor is always 7 — but **three of the six infants (25003, 25007, 25008) were
never coded for `crawl`**, so those folds are charged a zero for a class their
infant does not have. κ is computed with `labels=` pinned and is unaffected.
Correcting the divisor to the classes the infant was actually coded for moves
those folds by **+0.107 to +0.113 macro-F1**:

| held out | classes coded | accuracy | macro-F1 as reported | macro-F1, correct divisor | κ |
|---|---|---|---|---|---|
| 25004 | 7 | 0.679 | 0.674 | 0.674 | **0.577** |
| 25003 | 6 | 0.874 | 0.663 | **0.774** | 0.827 |
| 25008 | 6 | 0.882 | 0.644 | **0.752** | 0.841 |
| 25006 | 7 | 0.879 | 0.754 | 0.754 | 0.846 |
| 25005 | 7 | 0.902 | 0.751 | 0.751 | 0.865 |
| 25007 | 6 | 0.933 | 0.676 | **0.788** | 0.905 |

**25004 is the one fold where κ falls *below* macro-F1** (0.577 vs 0.674), and
it is not a divisor effect — 25004 has all 7 classes. It is the infant whose
posture distribution is unlike everyone else's (29% `stand`, 0.3% `on back`), so
its accuracy collapses to 0.679 while its per-class scores stay middling and
evenly spread. κ punishes the low accuracy; macro-F1 does not notice.

`analysis/posture/harnet10_f1_vs_kappa.csv` carries all three divisors
(`macro_f1_union`, `macro_f1_pinned`, `macro_f1_present`) alongside p_e per
fold. The pooled numbers in `harnet10_loso_results.csv` and
`baseline_loso_results.csv` are **not** affected — pooled predictions contain
every class — so the headline table stands; only the per-fold macro-F1 column
understates three of the six folds.
