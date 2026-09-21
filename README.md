# IMU_pretraining

## Posture distribution

```bash
python scripts/posture_distribution.py          # ~25 s
sbatch slurm/posture_distribution.sbatch        # same thing as a batch job
```

### Inputs

| What | Where |
|---|---|
| IMU / audio / ECG recordings | `/work/hdd/bebr/Data/LB/cleaned/BRP1/<id>/{Visit,Home}/<session>/` |
| EMA behavioural annotations | `/work/hdd/bebr/Data/Behavioral_Coding/BRP1_EMAannotations/*_EDITED.txt` |
| Annotation→recording offsets | `BRP1_EMAfiles_ToneTimes_tracking.xlsx` |

Annotation files are ELAN tab-delimited exports, 9 columns, one row per
annotation: `tier | <blank> | begin_hms | begin_s | end_hms | end_s | dur_hms |
dur_s | label`. Tiers present: `INTERACTION`, `state`, `posture`, `feeding`,
`location`, `sound_played`, `Missing`, `Sleep`, `is_paused`.

**The annotation clock starts at the GoPro tone, not at the start of the LB
recording.** `LBoffset_tonetime_secs` in the tracking sheet converts one to the
other, so a posture segment only lands on the IMU stream when that offset is
known.

Two tier-name quirks are normalised in the loader: `position` is used instead of
`posture` in `BRP1_25008_Obs1A`, and `INTERACTIONS` instead of `INTERACTION` in
`BRP1_25003_Obs12A`. Labels are whitespace-stripped and lowercased (the raw files
carry trailing-space variants such as `"sit "`, `"crawl  "`).

### Outputs — `analysis/posture/`

| File | Contents |
|---|---|
| `posture_segments.csv` | every posture segment + LB-clock times + alignability |
| `posture_by_class.csv` | per-class counts, duration, share, segment stats |
| `posture_by_subject_seconds.csv` | subject × class duration matrix |
| `posture_by_observation.csv` | per-observation coverage + IMU linkage |
| `annotation_coverage.csv` | recordings vs. annotations per subject |
| `posture_by_class.png` | ranked duration per class |
| `posture_by_subject.png` | per-subject composition (100% stacked) |
| `posture_segment_durations.png` | segment-length spread per class |

## Documentation

- **`docs/METHODS.md`** — dataset description, all 19 features, the 5 models,
  and the evaluation protocol, in prose.
- **`analysis/posture/feature_dictionary.csv`** — the same feature table,
  machine-readable: `feature, feature_set, group, units, definition, rationale,
  in_set_A/B/C`. Row order matches the feature column order in
  `windows_10s.parquet`.

## Baseline probe — how much is just gravity?

```bash
python scripts/posture_baseline.py            # ~2 min from cache, ~8 min cold
sbatch slurm/posture_baseline.sbatch
python scripts/posture_baseline.py --rebuild  # force re-windowing
```

Windows the 71.7 h of IMU-alignable posture at 10 s / 5 s hop, keeping only
windows where one label covers ≥80% (48,135 windows), then fits nested feature
sets under **leave-one-subject-out** CV. `recline forward` is dropped (3
segments corpus-wide); the magnetometer is deliberately unused — indoor
hard/soft-iron distortion makes it a room-identity feature, not a posture one.

| features | model | acc | macro-F1 |
|---|---|---|---|
| majority class | — | 0.262 | 0.059 |
| A: gravity only (3) | logreg | 0.799 | 0.591 |
| A: gravity only (3) | GBDT | 0.836 | 0.676 |
| C: + accel + gyro (19) | GBDT | **0.854** | **0.729** |

Per-class F1: on stomach 0.97, on back 0.94, recline back 0.91, on side 0.88 —
then **sit 0.63, stand 0.46, crawl 0.32**. Lying postures are solved by trunk
orientation alone; all remaining headroom is in upright/transitional classes.

Outputs: `windows_10s.parquet` (feature cache), `baseline_loso_results.csv`,
`baseline_confusion.csv`, `baseline_confusion.png`.

### Known alignment problems

Nine observations run past the end of their LB recording. Seven overrun by
1.8–24 s (benign — the GoPro outlived LittleBeats) but **two are real
mismatches** and their labels should not be trusted:

| observation | session | issue |
|---|---|---|
| 25005 Obs4A | `BRP1_25005_REC456_2025-10-06-09-40-48` | 2.2 h annotation against a 1.1 h recording |
| 25003 Obs4B | `BRP1_25003_REC456_2025-01-29-10-49-55` | tone at 15880 s, needs 22292 s of a 19621 s recording |

Together they contribute 1,415 windows (2.9%).
