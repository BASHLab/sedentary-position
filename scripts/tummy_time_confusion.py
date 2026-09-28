#!/usr/bin/env python3
"""Confusion matrices + kappa for the fully-automated tummy-time detector.

Two levels, because tummy time is a conjunction and either half can be the one
that fails:

  row 1  **tummy time** -- harnet10 posture AND ECG wake, against the human
         coding, as a 2x2 detection problem
  row 2  **ECG sleep/wake alone**, the half that is doing the damage

Each panel is scoped so the ECG model's own split stays visible: 25003-25006 are
in its training set, 25008 is the only held-out infant, and 25007 has no ECG
recording at all (its windows can only ever be false negatives, which is why the
all-windows panel and the ECG-covered panel differ so much).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

PROJ = Path("/work/hdd/bebr/Projects/IMU_pretraining")
OUT_DIR = PROJ / "analysis" / "posture"
FIG_DIR = PROJ / "figures"
ECG_TRAIN = {25003, 25004, 25005, 25006}
ECG_HELDOUT = {25008}

# same house tokens as the other figures in this project
SURFACE, INK, INK_2, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", [SURFACE, *BLUE_RAMP])


def panel(ax, y, yhat, labels, title, subtitle):
    C = confusion_matrix(y, yhat, labels=[0, 1])
    row = C.sum(1, keepdims=True)
    P = np.divide(C * 100.0, np.maximum(row, 1))
    ax.imshow(P, cmap=SEQ, vmin=0, vmax=100, aspect="equal")

    k = cohen_kappa_score(y, yhat, labels=[0, 1])
    f1 = f1_score(y, yhat, zero_division=0)
    # Header stacked bottom-up so the two-line subtitle cannot land on the
    # title: metrics sit just above the matrix, subtitle above that, title on top.
    ax.text(0, 1.03, f"κ {k:.3f}    F1 {f1:.3f}    n={len(y):,}",
            transform=ax.transAxes, color=INK, fontsize=9.5,
            fontweight="bold", va="bottom")
    ax.text(0, 1.11, subtitle, transform=ax.transAxes, color=INK_2,
            fontsize=8.8, va="bottom", linespacing=1.45)
    ax.text(0, 1.30, title, transform=ax.transAxes, color=INK,
            fontsize=11.5, fontweight="bold", va="bottom")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([f"pred\n{labels[0]}", f"pred\n{labels[1]}"], fontsize=9)
    ax.set_yticklabels([f"coded\n{labels[0]}", f"coded\n{labels[1]}"], fontsize=9)
    for lb in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        lb.set_color(INK_2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([-0.5, 0.5, 1.5], minor=True)
    ax.set_yticks([-0.5, 0.5, 1.5], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3.0)
    ax.tick_params(which="minor", length=0)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{C[i, j]:,}\n{P[i, j]:.1f}%", ha="center",
                    va="center", fontsize=10.5,
                    color="#ffffff" if P[i, j] >= 55 else INK_2,
                    fontweight="bold" if i == j else "normal")
    return {"kappa": k, "f1": f1, "tn": int(C[0, 0]), "fp": int(C[0, 1]),
            "fn": int(C[1, 0]), "tp": int(C[1, 1]), "n": int(len(y))}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    w = pd.read_parquet(OUT_DIR / "tummy_time_windows.parquet")
    e = pd.read_parquet(OUT_DIR / "ecg_sleep_predictions.parquet")
    key = ["family_id", "obs", "t_start_s"]
    w = w.merge(e[key + ["ecg_state"]], on=key, how="left")
    cov = w["ecg_state"].notna().to_numpy()

    ref = w["tummy_annotated"].to_numpy().astype(int)
    pred = w["tummy_predicted"].to_numpy().astype(int)
    held = w["family_id"].isin(ECG_HELDOUT).to_numpy()
    train = w["family_id"].isin(ECG_TRAIN).to_numpy()

    # ECG sleep/wake alone, where a coded state exists
    has_state = w["coded_state"].isin(["wake", "sleep"]).to_numpy() & cov
    sy = (w["coded_state"] == "wake").to_numpy().astype(int)
    sp = (w["ecg_state"] == "wake").to_numpy().astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(14.6, 9.6), facecolor=SURFACE)
    rows = []
    TT = ("not tummy", "tummy")
    SW = ("sleep", "wake")

    specs = [
        (axes[0, 0], ref, pred, np.ones(len(w), bool), TT,
         "Tummy time — all windows",
         "harnet10 posture AND ECG wake; 25007 has no ECG so its\n"
         "tummy windows are forced false negatives"),
        (axes[0, 1], ref, pred, cov, TT,
         "Tummy time — ECG-covered only",
         "the 38,304 windows where the ECG model can actually speak\n"
         "(65 of 79 recordings, 5 of 6 infants)"),
        (axes[0, 2], ref, pred, cov & held, TT,
         "Tummy time — HELD-OUT 25008",
         "the only infant outside the ECG model's training set\n"
         "(still its validation set, so selection-optimistic)"),
        (axes[1, 0], sy, sp, has_state, SW,
         "ECG sleep/wake — all covered",
         "the ECG half on its own, against the coded state tier"),
        (axes[1, 1], sy, sp, has_state & train, SW,
         "ECG sleep/wake — training infants",
         "25003–25006: in the ECG model's training set"),
        (axes[1, 2], sy, sp, has_state & held, SW,
         "ECG sleep/wake — HELD-OUT",
         "the honest generalisation view of the ECG half"),
    ]
    for ax, y, yh, m, labs, title, sub in specs:
        r = panel(ax, y[m], yh[m], labs, title, sub)
        r["panel"] = title
        rows.append(r)

    fig.text(0.008, 0.977,
             "Tummy time = awake AND on stomach — where the fully-automated "
             "pipeline agrees with human coding",
             color=INK, fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(0.008, 0.945,
             "Cells show window counts and row %. The top row is the conjunction; "
             "the bottom row is the ECG half alone, which is the weaker of the two.",
             color=INK_2, fontsize=10, ha="left", va="top")
    fig.text(0.008, 0.008,
             "10 s windows, 5 s hop. harnet10 posture is leave-one-subject-out "
             "out-of-fold; the ECG model trained on 25003–25006.",
             color=INK_MUTED, fontsize=9, ha="left")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.800, bottom=0.055,
                        wspace=0.38, hspace=0.72)
    fig.savefig(FIG_DIR / "tummy_time_confusion.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    R = pd.DataFrame(rows)[["panel", "n", "tn", "fp", "fn", "tp", "kappa", "f1"]]
    R.to_csv(OUT_DIR / "tummy_time_confusion.csv", index=False)
    print(R.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote tummy_time_confusion.csv to {OUT_DIR}")
    print(f"Wrote tummy_time_confusion.png to {FIG_DIR}")


if __name__ == "__main__":
    main()
