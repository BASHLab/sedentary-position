#!/usr/bin/env python3
"""Score the ECG sleep/wake predictions against the human coding, per infant.

The point of splitting by infant is that the ECG model's own split runs straight
through this cohort:

    BRP_train.csv  25003 (18 files), 25004 (39), 25005 (27), 25006 (36)
    BRP_test.csv   25008 (42)

so four of the six infants here are **in the model's training set** and only
**25008 is held out**. A pooled accuracy over all of them is not an estimate of
anything -- it is mostly a training-set score. This script reports the two
groups separately.

One further caveat on the held-out number: the checkpoint in use
(`results_sleep_v5_linear/best_model.pt`) was selected as the best epoch *on
25008*, because that repo uses BRP_test as its validation set. So even 25008 is
held out from fitting but not from model selection, and its score here is
optimistic relative to a genuinely untouched infant. Treat it as an upper bound.

25007 has no `ecg_v2` recording at all and cannot be scored.

Reference labels are the `state` tier mapped by the ECG model's own vocabulary,
already materialised per window by scripts/tummy_time.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             roc_auc_score)

OUT_DIR = Path("/work/hdd/bebr/Projects/IMU_pretraining/analysis/posture")

# from BRP_train.csv / BRP_test.csv in the ECG repo
ECG_TRAIN = {25003, 25004, 25005, 25006}
ECG_HELDOUT = {25008}


def scores(y_true: np.ndarray, y_pred: np.ndarray, p_wake: np.ndarray) -> dict:
    """y_* are 1 = wake, 0 = sleep."""
    out = {
        "n": len(y_true),
        "wake_frac_coded": float(y_true.mean()),
        "wake_frac_pred": float(y_pred.mean()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred, labels=[0, 1]),
    }
    out["auroc"] = (roc_auc_score(y_true, p_wake)
                    if len(np.unique(y_true)) == 2 else float("nan"))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    # per-class recall: how much of each coded class the model finds
    out["recall_wake"] = tp / (tp + fn) if tp + fn else float("nan")
    out["recall_sleep"] = tn / (tn + fp) if tn + fp else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path,
                    default=OUT_DIR / "tummy_time_windows.parquet")
    ap.add_argument("--ecg-pred", type=Path,
                    default=OUT_DIR / "ecg_sleep_predictions.parquet")
    args = ap.parse_args()

    w = pd.read_parquet(args.windows)
    e = pd.read_parquet(args.ecg_pred)
    key = ["family_id", "obs", "t_start_s"]
    df = w[key + ["coded_state"]].merge(
        e[key + ["ecg_state", "p_wake"]], on=key, how="inner")

    df = df[(df["coded_state"].isin(["wake", "sleep"]))
            & df["ecg_state"].notna()]
    df["y"] = (df["coded_state"] == "wake").astype(int)
    df["yhat"] = (df["ecg_state"] == "wake").astype(int)
    print(f"{len(df):,} windows with both a coded state and an ECG prediction")

    rows = []
    for fid, g in df.groupby("family_id"):
        split = ("train (in-sample)" if fid in ECG_TRAIN else
                 "HELD OUT" if fid in ECG_HELDOUT else "not in ECG split")
        rows.append({"family_id": int(fid), "ecg_split": split,
                     **scores(g["y"].to_numpy(), g["yhat"].to_numpy(),
                              g["p_wake"].to_numpy())})

    for name, mask in (("ECG train subjects (pooled)",
                        df["family_id"].isin(ECG_TRAIN)),
                       ("ECG HELD-OUT (25008)",
                        df["family_id"].isin(ECG_HELDOUT))):
        g = df[mask]
        if len(g):
            rows.append({"family_id": name, "ecg_split": "",
                         **scores(g["y"].to_numpy(), g["yhat"].to_numpy(),
                                  g["p_wake"].to_numpy())})

    R = pd.DataFrame(rows)
    R.to_csv(OUT_DIR / "ecg_sleep_eval_by_subject.csv", index=False)

    show = ["family_id", "ecg_split", "n", "accuracy", "balanced_acc",
            "macro_f1", "kappa", "auroc", "recall_wake", "recall_sleep",
            "wake_frac_coded", "wake_frac_pred"]
    print("\n" + "=" * 118)
    print("ECG SLEEP/WAKE vs HUMAN CODING")
    print("=" * 118)
    print(R[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    ho = df[df["family_id"].isin(ECG_HELDOUT)]
    if len(ho):
        cm = confusion_matrix(ho["y"], ho["yhat"], labels=[0, 1])
        print("\nHeld-out infant 25008 -- confusion (rows coded, cols predicted):")
        print(f"              pred sleep   pred wake")
        print(f"  coded sleep {cm[0,0]:>10,} {cm[0,1]:>11,}")
        print(f"  coded wake  {cm[1,0]:>10,} {cm[1,1]:>11,}")
        s = scores(ho["y"].to_numpy(), ho["yhat"].to_numpy(),
                   ho["p_wake"].to_numpy())
        print(f"\n  accuracy {s['accuracy']:.3f} | balanced {s['balanced_acc']:.3f} "
              f"| macro-F1 {s['macro_f1']:.3f} | kappa {s['kappa']:.3f} "
              f"| AUROC {s['auroc']:.3f}")
        print("\n  The ECG repo reports macro-F1 0.858 / acc 0.897 / AUROC 0.892 on\n"
              "  this same infant, over its own 30 s windows inside coded state\n"
              "  segments. This run scores every posture window instead, so the\n"
              "  windows differ; the numbers should be close, not identical.")

    print(f"\nWrote ecg_sleep_eval_by_subject.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
