# Tummy time from IMU + ECG, without human coding

Companion to `scripts/tummy_time.py`, `scripts/ecg_sleep_infer.py`,
`scripts/ecg_sleep_eval.py`, `scripts/tummy_time_metrics.py` and
`scripts/tummy_time_confusion.py`.

---

## 1. What is being measured

**Tummy time = the infant is awake AND on its stomach.** It is a conjunction of
two things neither sensor can supply alone: posture comes from the chest IMU,
wake/sleep from the ECG. Two pretrained models are combined, each already
fine-tuned on its own task, and the result is checked against human behavioural
coding.

The wake/sleep vocabulary is taken **verbatim** from the ECG model's own
annotation parser
(`ecg_foundation_model/single_stream_downstream/dataset.py::_parse_sleep_segments`)
so that the model's predictions and the human reference mean the same thing:

| | labels |
|---|---|
| **wake** | quiet alert, active alert, active, crying |
| **sleep** | drowsy, drowsy unsure, light sleep, deep sleep |

`drowsy` counting as *sleep* is the single most debatable line in the
definition. `--drowsy-awake` flips it; it moves the total by **1.9%** (11.90 h →
12.13 h), so the choice does not carry the result.

## 2. The window grid

Everything rides on the grid the rest of this project already uses: **48,135
windows of 10 s on a 5 s hop**, drawn from `windows_raw_30hz_meta.parquet`,
keyed by `(family_id, obs, lb_session, t_start_s)` on the LittleBeats clock.
6 infants, 66.85 h of covered time.

Because windows overlap by half, **each window is credited one hop (5 s)**, not
its full 10 s. That makes the hour totals a true partition of covered time, and
the same rule is applied to the reference and to every prediction, so the
columns are directly comparable.

Human `state` coding covers **47,819 windows (99.3%)** at the same >=80%-coverage
rule used for posture: 32,578 wake, 15,241 sleep.

## 3. Posture — harnet10 (IMU)

The fine-tuned harnet10 from `docs/METHODS.md`, used at its **leave-one-subject-out
out-of-fold** predictions, so every window was predicted while its own infant was
held out. On-stomach is its strongest class (F1 0.979): 11,523 windows predicted
on-stomach against 11,345 coded.

## 4. Wake/sleep — the infant ECG foundation model

From `/work/nvme/bebr/mkhan14/ecg_foundation_model`: each 30 s ECG window is a
graph of heartbeats (per-beat CNN → +RR features → transformer → GNN →
attention pool), self-supervised on the BCP cohort and fine-tuned on BRP for
sleep vs wake. Checkpoint `single_stream_downstream/results_sleep_v5_linear/best_model.pt`
(best validation macro-F1 of the 18 sleep runs present). 1000 Hz, 30 s, <=80 beats.

Three things in that repo had to be worked around, all recorded in
`scripts/ecg_sleep_infer.py`:

1. **Dead imports.** `single_stream_downstream/dataset.py` imports
   `ibi_graph_model.signal_utils` and `dual_stream_pretraining.signal_utils`,
   neither of which is still in the checkout, and reads per-recording IBI
   `.npz` files that are also gone. All four functions survive in
   `single_stream_pretraining/signal_utils.py`, and the IBI they want is just
   the R-peak differences, so the window tensors are rebuilt directly from the
   `.wav` + `_peaks.npy` pair.
2. **A time-base offset their pipeline does not correct.** The `ecg_v2` wavs do
   not start at the recording start — for `BRP1_25008_2025-10-09-09-39-05` the
   wav is 4180.6 s against the cleaned recording's 4214.0 s, its first sample
   ~4.35 s late. Their dataset indexes the wav at `t_start * sr` with `t_start`
   on the annotation clock, so every training window carried that per-recording
   shift. Here the offset is measured per recording from the ECG parquet's
   wall-clock `datetime` against the LittleBeats recording start, and
   subtracted.
3. **Coverage.** `ecg_v2` holds no recording for **25007** at all, and covers
   65 of 79 recordings overall → **38,304 of 48,135 windows (79.6%)**.

## 5. How the two are merged

There is no fusion model and no learned combination — it is a **hard logical AND
on a shared grid**. The only real design decision is how a 30 s ECG prediction
is attached to a 10 s posture window.

Rather than predict on an independent ECG grid and interval-join, one 30 s ECG
window is placed **centred on each posture window** (`ecg_sleep_infer.py`):

```python
centre = r["t_start_s"] + WIN_S / 2.0              # midpoint of the 10 s window
t_wav  = centre - cfg.window_sec / 2.0 - info["offset_s"]
```

so the correspondence is 1:1 by construction and the join is exact rather than
approximate. Consecutive ECG windows overlap by 25 s — redundant compute, but it
removes any interval-assignment ambiguity. Then (`tummy_time.py`):

```python
key = ["family_id", "obs", "t_start_s"]
e   = md[key].merge(e, on=key, how="left")     # left join onto the IMU grid
ecg_wake = (e["ecg_state"] == "wake").to_numpy()
cols["predicted"] = pred_stomach & ecg_wake    # the conjunction
```

The `left` join keeps all 48,135 rows and leaves `ecg_state` NaN where no ECG
exists. **`NaN == "wake"` is `False`, so a missing ECG becomes "not tummy time"
rather than "unknown"** — which is exactly why 25007 reports 0.00 h against
1.53 h of coded tummy time, and why every metric is reported twice: over all
windows, and restricted to the windows where the ECG model can actually speak.

Three pipelines are scored, in increasing order of how much is modelled:

| name | posture | state |
|---|---|---|
| `annotated` | coded | coded | ← the reference |
| `imu_only` | harnet10 | coded | isolates the IMU half |
| `ecg_only` | coded | ECG | isolates the ECG half |
| `predicted` | harnet10 | ECG | nothing human in the loop |

---

## 6. Results

### Hours of tummy time

| infant | covered h | annotated | imu_only | predicted | ECG cov. |
|---|---|---|---|---|---|
| 25003 | 10.05 | 2.11 | 2.15 | 2.20 | 99% |
| 25004 | 10.73 | 1.36 | 1.45 | 1.63 | 100% |
| 25005 | 8.56 | 1.60 | 1.58 | 1.52 | 100% |
| 25006 | 11.12 | 1.15 | 1.13 | 1.10 | 100% |
| 25007 | 13.56 | 1.53 | 1.56 | — | **0%** |
| 25008 | 12.83 | 4.14 | 4.12 | 4.12 | 100% |
| **ALL** | **66.85** | **11.90** | **12.00** | **10.57** | 80% |

**11.90 h of tummy time in 66.85 h of coded recording — 17.8% of covered time**,
with a 3.6x spread across infants (25006 1.15 h vs 25008 4.14 h).

### Detection metrics, per window

| pipeline | scope | precision | recall | F1 | kappa |
|---|---|---|---|---|---|
| `imu_only` | ECG-covered | 0.979 | 0.984 | 0.982 | **0.977** |
| `ecg_only` | ECG-covered | 0.964 | 0.980 | 0.972 | **0.965** |
| `predicted` | all windows | 0.945 | 0.839 | 0.889 | **0.866** |
| `predicted` | ECG-covered | 0.945 | 0.964 | 0.954 | **0.943** |
| `predicted` | **HELD-OUT 25008** | 0.989 | 0.984 | 0.986 | **0.979** |

Each model costs ~0.012-0.034 kappa on its own and 0.034 together, so **the two
error sources are close to independent rather than compounding**.

### The ECG half on its own

| scope | acc | balanced acc | macro-F1 | kappa | AUROC | recall wake | recall sleep |
|---|---|---|---|---|---|---|---|
| training infants | 0.913 | 0.911 | 0.906 | 0.812 | 0.965 | 0.918 | 0.904 |
| **HELD-OUT 25008** | 0.884 | 0.794 | 0.827 | **0.656** | 0.861 | 0.976 | **0.612** |

Its failure mode is **over-calling wake**: on the held-out infant it recovers
97.6% of coded wake but only 61.2% of coded sleep. Note that macro-F1 barely
moves out of sample (0.906 → 0.827) while kappa drops hard (0.812 → 0.656) —
kappa is the metric doing real work here.

### Why the conjunction is more robust than its weakest part

Tummy time on the held-out infant scores **better** (kappa 0.979) than on the
ECG model's own training subjects (0.921). The conjunction is the reason: a
sleep window mislabelled wake only becomes a false positive if the infant is
*also* on its stomach, and sleeping infants are mostly on their backs. Of 1,866
ECG wake errors in the covered set, only 422 survive into tummy time.

That is a real effect, but **do not read 0.979 as the expected value for a new
infant**. 25008 is 74.6% coded-wake, so the over-call-wake failure mode has
little to bite on; the worst tummy-time infant is 25004 (kappa 0.861, 227 false
positives), which is *in-sample* for the ECG model.

---

## 7. Limitations

- **The ECG model's split runs through this cohort.** It trained on 25003-25006
  and used 25008 as validation, so four of six infants are in-sample and the
  fifth was used for checkpoint selection. There is no infant here that is
  clean of both. Treat the held-out row as an upper bound.
- **25007 has no ECG** in `ecg_v2` — 9,760 windows, 20% of the corpus, and the
  reason the all-windows recall (0.839) understates the pipeline.
- **The pipeline over-reports.** 422 false positives against 267 false
  negatives on the covered set, i.e. +0.22 h on a 10.36 h reference (+2%),
  inherited from the ECG model's wake bias.
- **`drowsy` counts as sleep.** Flipping it changes the total by 1.9%.
- **Windows overlap**, so 48,135 is not 48,135 independent observations; the
  hour totals use the 5 s hop and are correct, the per-window counts are not
  independent samples.

## 8. Outputs

| file | contents |
|---|---|
| `analysis/posture/tummy_time_by_subject.csv` | hours per infant per pipeline + ECG coverage |
| `analysis/posture/tummy_time_windows.parquet` | per-window: coded/predicted posture, coded state, all three tummy flags |
| `analysis/posture/tummy_time_metrics.csv` | precision/recall/F1/kappa by pipeline, scope and infant |
| `analysis/posture/tummy_time_confusion.csv` | the six 2x2 matrices with kappa and F1 |
| `analysis/posture/ecg_sleep_predictions.parquet` | per-window `ecg_state`, `p_wake`, `p_sleep` |
| `analysis/posture/ecg_sleep_eval_by_subject.csv` | ECG sleep/wake scored per infant, split-aware |
| `figures/tummy_time_confusion.png` | tummy time and the ECG half, across all / covered / held-out |
| `*_drowsy_awake.*` | every table again with drowsy counted as awake |
