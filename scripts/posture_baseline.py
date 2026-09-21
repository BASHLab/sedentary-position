#!/usr/bin/env python3
"""Gravity-only floor for posture classification from the LittleBeats trunk IMU.

The question this answers: how much of the posture signal is just "which way is
the chest pointing", before any learned representation is involved?

Windows the IMU at 10 s / 5 s hop, labels each window from the posture tier
(requires one label to cover >=80% of the window), and fits three nested feature
sets under leave-one-subject-out CV:

    A  gravity      3 feats   unit-normalised mean accel vector -- orientation only
    B  + accel      11 feats  A + accel magnitude/per-axis dispersion
    C  + gyro       19 feats  B + angular-rate summaries

LOSO is mandatory here: 6 subjects, and a within-subject split would let any
model memorise subject identity instead of posture.

IMU layout (measured, constant across the corpus): 70 Hz, 9 channels --
0-2 accel m/s^2, 3-5 magnetometer, 6-8 gyro deg/s. The magnetometer is
deliberately unused: indoor hard/soft-iron distortion makes it a subject- and
room-identity feature, not a posture feature.
"""

from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJ = Path("/work/hdd/bebr/Projects/IMU_pretraining")
LB_DIR = Path("/work/hdd/bebr/Data/LB/cleaned/BRP1")
OUT_DIR = PROJ / "analysis" / "posture"
CACHE = OUT_DIR / "windows_10s.parquet"

FS = 70.0
WIN_S = 10.0
HOP_S = 5.0
MIN_PURITY = 0.80
DROP_CLASSES = {"recline forward"}  # 3 segments corpus-wide

ACC = slice(0, 3)
GYR = slice(6, 9)


def session_index() -> dict[str, Path]:
    return {
        s.name: s
        for fid in LB_DIR.iterdir() if fid.is_dir()
        for ctx in fid.iterdir() if ctx.is_dir()
        for s in ctx.iterdir() if s.is_dir()
    }


def audio_duration(base: Path) -> float:
    with wave.open(f"{base}_audio_sync.wav") as w:
        return w.getnframes() / w.getframerate()


# ------------------------------------------------------------------ features
def window_features(acc: np.ndarray, gyr: np.ndarray) -> list[float]:
    """acc, gyr: (3, n) slices of one window."""
    amag = np.linalg.norm(acc, axis=0)
    gmag = np.linalg.norm(gyr, axis=0)
    mean_a = acc.mean(axis=1)
    norm = np.linalg.norm(mean_a)
    grav = mean_a / norm if norm > 1e-6 else np.zeros(3)

    return [
        # --- A: orientation only (3)
        *grav,
        # --- B: + magnitude & dispersion (8)
        amag.mean(), amag.std(), norm,
        *acc.std(axis=1),
        np.abs(np.diff(amag)).mean(),
        amag.max() - amag.min(),
        # --- C: + angular rate (8)
        gmag.mean(), gmag.std(), gmag.max(),
        *np.abs(gyr).mean(axis=1),
        *gyr.std(axis=1)[:2],
    ]


FEAT_NAMES = (
    ["grav_x", "grav_y", "grav_z"]
    + ["amag_mean", "amag_std", "amean_norm", "ax_std", "ay_std", "az_std",
       "amag_jerk", "amag_ptp"]
    + ["gmag_mean", "gmag_std", "gmag_max", "gx_absmean", "gy_absmean",
       "gz_absmean", "gx_std", "gy_std"]
)
SET_A = FEAT_NAMES[:3]
SET_B = FEAT_NAMES[:11]
SET_C = FEAT_NAMES


# ------------------------------------------------------------------ windowing
def build_windows() -> pd.DataFrame:
    seg = pd.read_csv(OUT_DIR / "posture_segments.csv")
    seg = seg[seg["imu_alignable"] & ~seg["posture"].isin(DROP_CLASSES)]
    sessions = session_index()

    classes = sorted(seg["posture"].unique())
    cls_idx = {c: i for i, c in enumerate(classes)}

    win_n, hop_n = int(WIN_S * FS), int(HOP_S * FS)
    rows, overruns = [], []
    groups = list(seg.groupby(["family_id", "obs", "lb_session"], sort=True))

    for k, ((fid, obs, lb_session), g) in enumerate(groups, 1):
        sess = sessions.get(lb_session)
        if sess is None:
            print(f"  ! no session dir for {lb_session}", flush=True)
            continue
        base = sess / sess.name
        imu = np.loadtxt(f"{base}_imu_sync.txt")
        n = imu.shape[1]

        # Per-sample label track on the LB clock (-1 = uncoded).
        labels = np.full(n, -1, dtype=np.int8)
        want_end = 0.0
        for _, r in g.iterrows():
            want_end = max(want_end, r["lb_end_s"])
            i0 = max(int(round(r["lb_begin_s"] * FS)), 0)
            i1 = min(int(round(r["lb_end_s"] * FS)), n)
            if i1 > i0:
                labels[i0:i1] = cls_idx[r["posture"]]
        if want_end * FS > n:
            overruns.append((lb_session, want_end - n / FS))

        acc_all, gyr_all = imu[ACC], imu[GYR]
        for start in range(0, n - win_n + 1, hop_n):
            lab = labels[start:start + win_n]
            coded = lab[lab >= 0]
            if coded.size < win_n * MIN_PURITY:
                continue
            vals, counts = np.unique(coded, return_counts=True)
            if counts.max() < win_n * MIN_PURITY:
                continue  # window straddles a posture transition
            rows.append(
                [fid, obs, lb_session, start / FS, classes[vals[counts.argmax()]]]
                + window_features(acc_all[:, start:start + win_n],
                                  gyr_all[:, start:start + win_n])
            )
        if k % 10 == 0 or k == len(groups):
            print(f"  [{k:3d}/{len(groups)}] {len(rows):6d} windows", flush=True)

    df = pd.DataFrame(
        rows, columns=["family_id", "obs", "lb_session", "t_start_s", "posture"]
        + FEAT_NAMES,
    )
    if overruns:
        tot = sum(o for _, o in overruns)
        print(f"  {len(overruns)} observations ran past the IMU end "
              f"(total {tot:.0f} s, max {max(o for _, o in overruns):.0f} s) — truncated")
    return df


# ----------------------------------------------------------------- evaluation
def loso(df: pd.DataFrame, feats: list[str], model_fn, name: str) -> dict:
    X, y, gr = df[feats].to_numpy(), df["posture"].to_numpy(), df["family_id"].to_numpy()
    pred = np.empty_like(y)
    per_f1, per_kappa = {}, {}
    labels = np.unique(y)
    for s in np.unique(gr):
        te = gr == s
        m = model_fn().fit(X[~te], y[~te])
        pred[te] = m.predict(X[te])
        per_f1[int(s)] = f1_score(y[te], pred[te], average="macro", zero_division=0)
        # labels= pins the full class set: a fold missing a class (crawl is in
        # only 3 of 6 subjects) would otherwise score against a smaller matrix.
        per_kappa[int(s)] = cohen_kappa_score(y[te], pred[te], labels=labels)
    return {
        "model": name,
        "n_feats": len(feats),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y, pred, labels=labels),
        "mean_fold_kappa": float(np.mean(list(per_kappa.values()))),
        "worst_subject_f1": min(per_f1.values()),
        "worst_subject_kappa": min(per_kappa.values()),
        "per_subject_f1": per_f1,
        "per_subject_kappa": per_kappa,
        "pred": pred,
        "y": y,
    }


SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def fig_confusion(cmn: np.ndarray, classes: list[str], title: str) -> None:
    """Magnitude over a grid -> heatmap, one hue light->dark."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
    n = len(classes)
    fig, ax = plt.subplots(figsize=(1.05 * n + 3.2, 1.0 * n + 2.2), facecolor=SURFACE)
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(n), classes, rotation=35, ha="right", color=INK, fontsize=10)
    ax.set_yticks(range(n), classes, color=INK, fontsize=10)
    ax.set_xlabel("predicted", color=INK_2, fontsize=10.5, labelpad=8)
    ax.set_ylabel("true", color=INK_2, fontsize=10.5, labelpad=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.0)  # 2px surface gap
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            v = cmn[i, j]
            if v < 0.5:
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if v > 55 else INK_2,
                    fontweight="bold" if i == j else "normal")

    ax.set_title(title, color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=14)
    fig.text(0.008, 0.008, "row-normalised %, leave-one-subject-out",
             color="#8a8985", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT_DIR / "baseline_confusion.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE.exists() and "--rebuild" not in sys.argv:
        df = pd.read_parquet(CACHE)
        print(f"Loaded {len(df)} cached windows from {CACHE.name}")
    else:
        print(f"Windowing IMU at {WIN_S:.0f}s / {HOP_S:.0f}s hop, "
              f"purity>={MIN_PURITY:.0%} …")
        t0 = time.time()
        df = build_windows()
        df.to_parquet(CACHE, index=False)
        print(f"Built {len(df)} windows in {time.time() - t0:.0f}s -> {CACHE.name}")

    print(f"\n{len(df)} windows = {len(df) * HOP_S / 3600:.1f} h of stride-covered time, "
          f"{df['family_id'].nunique()} subjects, {df['posture'].nunique()} classes")
    dist = df["posture"].value_counts()
    print("\nWindow class balance:")
    for c, v in dist.items():
        print(f"  {c:16s} {v:6d}  {v / len(df) * 100:5.1f}%")

    majority = dist.max() / len(df)
    print(f"\nMajority-class accuracy: {majority:.3f}  "
          f"(macro-F1 {f1_score(df['posture'], [dist.idxmax()] * len(df), average='macro', zero_division=0):.3f})")

    runs = [
        (SET_A, lambda: make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=2000, C=1.0)),
         "A gravity only      (logreg)"),
        (SET_B, lambda: make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=2000, C=1.0)),
         "B + accel dispersion(logreg)"),
        (SET_C, lambda: make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=2000, C=1.0)),
         "C + gyro            (logreg)"),
        (SET_A, lambda: HistGradientBoostingClassifier(max_iter=300, random_state=0),
         "A gravity only      (GBDT)"),
        (SET_C, lambda: HistGradientBoostingClassifier(max_iter=300, random_state=0),
         "C + gyro            (GBDT)"),
    ]

    print("\nLeave-one-subject-out …")
    results = []
    for feats, fn, name in runs:
        t0 = time.time()
        r = loso(df, feats, fn, name)
        results.append(r)
        print(f"  {name:30s} acc={r['accuracy']:.3f}  macroF1={r['macro_f1']:.3f}  "
              f"kappa={r['kappa']:.3f}  worst-subj F1={r['worst_subject_f1']:.3f}/"
              f"k={r['worst_subject_kappa']:.3f}  ({time.time() - t0:.0f}s)")

    best = max(results, key=lambda r: r["macro_f1"])
    classes = sorted(df["posture"].unique())
    cm = confusion_matrix(best["y"], best["pred"], labels=classes)
    cmn = cm / cm.sum(axis=1, keepdims=True) * 100

    print(f"\nPer-class F1 — {best['model'].strip()}")
    f1s = f1_score(best["y"], best["pred"], average=None, labels=classes,
                   zero_division=0)
    for c, f in zip(classes, f1s):
        print(f"  {c:16s} F1={f:.3f}  (n={int((best['y'] == c).sum())})")

    print(f"\nConfusion (row = true, % of row) — {best['model'].strip()}")
    head = "".join(f"{c[:9]:>10s}" for c in classes)
    print(f"{'':17s}{head}")
    for i, c in enumerate(classes):
        print(f"  {c:15s}" + "".join(f"{v:9.1f} " for v in cmn[i]))

    n_test = df.groupby("family_id").size().to_dict()
    print(f"\nPer-fold scores — {best['model'].strip()}")
    print(f"  {'held out':<10s}{'n_test':>8s}{'macro-F1':>10s}{'kappa':>8s}")
    for s, f in sorted(best["per_subject_f1"].items()):
        print(f"  {s:<10d}{n_test[s]:8d}{f:10.3f}{best['per_subject_kappa'][s]:8.3f}")
    print(f"  {'pooled':<10s}{len(df):8d}{best['macro_f1']:10.3f}{best['kappa']:8.3f}")

    summary = pd.DataFrame(
        [{k: v for k, v in r.items()
          if k not in ("pred", "y", "per_subject_f1", "per_subject_kappa")}
         for r in results]
    )
    summary["majority_baseline_acc"] = majority
    summary.to_csv(OUT_DIR / "baseline_loso_results.csv", index=False)

    pd.DataFrame([
        {"model": r["model"].strip(), "held_out": s, "n_test": n_test[s],
         "macro_f1": r["per_subject_f1"][s], "kappa": r["per_subject_kappa"][s]}
        for r in results for s in sorted(r["per_subject_f1"])
    ]).to_csv(OUT_DIR / "baseline_loso_per_fold.csv", index=False)

    # Same numbers pivoted subject x model, for reading in a spreadsheet.
    wide = pd.DataFrame({"held_out": sorted(n_test), 
                         "n_test": [n_test[s] for s in sorted(n_test)]})
    for r in results:
        k = "_".join(r["model"].split()).replace("(", "").replace(")", "")
        wide[f"kappa__{k}"] = [r["per_subject_kappa"][s] for s in wide.held_out]
        wide[f"macroF1__{k}"] = [r["per_subject_f1"][s] for s in wide.held_out]
    wide.loc[len(wide)] = (["pooled", len(df)]
                           + [v for r in results for v in (r["kappa"], r["macro_f1"])])
    wide.to_csv(OUT_DIR / "baseline_loso_by_subject.csv", index=False)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        OUT_DIR / "baseline_confusion.csv")
    fig_confusion(cmn, classes,
                  f"Posture confusion — hand-crafted features + GBDT "
                  f"(acc {best['accuracy']:.3f}, macro-F1 {best['macro_f1']:.3f})")
    print(f"\nWrote {OUT_DIR / 'baseline_loso_results.csv'} + baseline_confusion.png")


if __name__ == "__main__":
    main()
