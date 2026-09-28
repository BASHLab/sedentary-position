#!/usr/bin/env python3
"""Detection metrics for tummy time, broken out by infant and by ECG split.

Works off the artefacts `tummy_time.py` and `ecg_sleep_infer.py` already wrote,
so it is cheap to re-run and does not re-parse annotations.

Tummy time is a conjunction -- awake AND on stomach -- so a window is a true
positive only when both halves are right. Three pipelines are scored against the
human coding:

    imu_only    harnet10 posture + coded state    isolates the IMU model
    predicted   harnet10 posture + ECG state      nothing human in the loop
    ecg_only    coded posture + ECG state         isolates the ECG model

Splitting `predicted` by the ECG model's own train/test split matters: 25003-6
are in its training set, 25008 is held out, and 25007 has no ECG recording at
all. Pooling them reports mostly a training-set score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("/work/hdd/bebr/Projects/IMU_pretraining/analysis/posture")
ECG_TRAIN = {25003, 25004, 25005, 25006}
ECG_HELDOUT = {25008}
HOP_S = 5.0


def prf(ref: np.ndarray, pred: np.ndarray) -> dict:
    tp = int((ref & pred).sum()); fp = int((~ref & pred).sum())
    fn = int((ref & ~pred).sum()); tn = int((~ref & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if (p == p and r == r and p + r) else float("nan")
    # Cohen's kappa for a binary detector
    n = tp + fp + fn + tn
    po = (tp + tn) / n if n else float("nan")
    pe = (((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)) if n else float("nan")
    k = (po - pe) / (1 - pe) if pe == pe and pe != 1 else float("nan")
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f1, "kappa": k,
            "pred_h": (tp + fp) * HOP_S / 3600.0,
            "ref_h": (tp + fn) * HOP_S / 3600.0,
            "err_h": (tp + fp) * HOP_S / 3600.0 - (tp + fn) * HOP_S / 3600.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path,
                    default=OUT_DIR / "tummy_time_windows.parquet")
    ap.add_argument("--ecg-pred", type=Path,
                    default=OUT_DIR / "ecg_sleep_predictions.parquet")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    w = pd.read_parquet(args.windows)
    e = pd.read_parquet(args.ecg_pred)
    key = ["family_id", "obs", "t_start_s"]
    w = w.merge(e[key + ["ecg_state"]], on=key, how="left")
    w["ecg_covered"] = w["ecg_state"].notna()

    ref = w["tummy_annotated"].to_numpy()
    stomach_pred = (w["pred_posture"] == "on stomach").to_numpy()
    stomach_coded = (w["posture"] == "on stomach").to_numpy()
    ecg_wake = (w["ecg_state"] == "wake").to_numpy()

    pipes = {
        "imu_only": w["tummy_imu_only"].to_numpy(),
        "predicted": w["tummy_predicted"].to_numpy()
        if "tummy_predicted" in w else stomach_pred & ecg_wake,
        "ecg_only": stomach_coded & ecg_wake,
    }

    cov = w["ecg_covered"].to_numpy()
    rows = []
    for name, pred in pipes.items():
        rows.append({"pipeline": name, "scope": "all windows", **prf(ref, pred)})
        rows.append({"pipeline": name, "scope": "ECG-covered only",
                     **prf(ref[cov], pred[cov])})

    # per infant, on the fully-predicted pipeline
    for fid, g in w.groupby("family_id"):
        m = g.index
        split = ("ECG train" if fid in ECG_TRAIN else
                 "ECG HELD OUT" if fid in ECG_HELDOUT else "no ECG")
        for name in ("imu_only", "predicted"):
            sel = pipes[name][m]
            if name == "predicted" and not cov[m].any():
                continue
            idx = m if name == "imu_only" else m[cov[m]]
            rows.append({"pipeline": name, "scope": f"{int(fid)} ({split})",
                         **prf(ref[idx], pipes[name][idx])})

    # grouped by ECG split, fully-predicted only
    for label, ids in (("ECG train subjects", ECG_TRAIN),
                       ("ECG HELD OUT (25008)", ECG_HELDOUT)):
        m = w.index[w["family_id"].isin(ids) & w["ecg_covered"]]
        if len(m):
            rows.append({"pipeline": "predicted", "scope": label,
                         **prf(ref[m], pipes["predicted"][m])})

    R = pd.DataFrame(rows)
    R.to_csv(OUT_DIR / f"tummy_time_metrics{args.tag}.csv", index=False)

    show = ["pipeline", "scope", "n", "tp", "fp", "fn", "precision", "recall",
            "f1", "kappa", "pred_h", "ref_h", "err_h"]
    for pipe in ("imu_only", "predicted", "ecg_only"):
        sub = R[R["pipeline"] == pipe]
        if sub.empty:
            continue
        print("\n" + "=" * 112)
        print(f"TUMMY TIME -- {pipe}")
        print("=" * 112)
        print(sub[show].drop(columns="pipeline")
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote tummy_time_metrics{args.tag}.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
