#!/usr/bin/env python3
"""Rebuild LOSO metric tables from saved out-of-fold predictions.

Every model in this project saves its pooled out-of-fold predictions as one
`.npy` in window order, so the metric tables are a pure function of that array
plus the window metadata. That makes them recoverable when a run is killed after
the predictions are written but before the CSVs are -- which is exactly what the
4 h wall clock did to the LIMU-BERT probe stages on 2026-09-27.

    python metrics_from_predictions.py limu_acc_mag_probe_predictions.npy \
        --model "limu-bert (probe, acc_mag)"

Metrics match scripts/finetune_harnet.py exactly, including `labels=` pinned to
all seven classes so a fold missing a class is not scored against a smaller
matrix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

OUT_DIR = Path("/work/hdd/bebr/Projects/IMU_pretraining/analysis/posture")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="+",
                    help="*_predictions.npy under analysis/posture/")
    ap.add_argument("--meta", default="windows_raw_30hz_meta.parquet",
                    help="window metadata; any of the meta parquets, they are "
                         "asserted identical row-for-row")
    ap.add_argument("--model", default=None,
                    help="model name for the table; defaults to the filename")
    ap.add_argument("--out-prefix", default=None,
                    help="write <prefix>_loso_results.csv / _loso_per_fold.csv")
    args = ap.parse_args()

    md = pd.read_parquet(OUT_DIR / args.meta)
    classes = sorted(md["posture"].unique())
    labels = np.arange(len(classes))
    y = md["posture"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int64)
    gr = md["family_id"].to_numpy()

    rows, per_fold = [], []
    for pth in args.predictions:
        p = Path(pth)
        if not p.is_absolute():
            p = OUT_DIR / p
        pred = np.load(p)
        if pred.shape != y.shape:
            print(f"! {p.name}: {pred.shape} vs meta {y.shape}; skipped")
            continue
        name = args.model or p.stem.replace("_predictions", "")

        for s in np.unique(gr):
            te = gr == s
            per_fold.append({
                "model": name, "held_out": int(s), "n_test": int(te.sum()),
                "macro_f1": f1_score(y[te], pred[te], average="macro",
                                     zero_division=0),
                "kappa": cohen_kappa_score(y[te], pred[te], labels=labels),
                "best_epoch": np.nan,   # not recoverable from predictions
            })
        mine = [r for r in per_fold if r["model"] == name]
        rows.append({
            "model": name,
            "n_feats": np.nan,
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(y, pred, labels=labels),
            "mean_fold_kappa": float(np.mean([r["kappa"] for r in mine])),
            "worst_subject_f1": min(r["macro_f1"] for r in mine),
            "worst_subject_kappa": min(r["kappa"] for r in mine),
        })
        per_class = f1_score(y, pred, average=None, labels=labels, zero_division=0)
        print(f"\n{name}")
        print(f"  acc {rows[-1]['accuracy']:.3f} | macro-F1 "
              f"{rows[-1]['macro_f1']:.3f} | kappa {rows[-1]['kappa']:.3f}")
        print("  per-class F1: " + "  ".join(
            f"{c} {v:.3f}" for c, v in zip(classes, per_class)))

    if not rows:
        return
    R, F = pd.DataFrame(rows), pd.DataFrame(per_fold)
    print("\n" + R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + F.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.out_prefix:
        R.to_csv(OUT_DIR / f"{args.out_prefix}_loso_results.csv", index=False)
        F.to_csv(OUT_DIR / f"{args.out_prefix}_loso_per_fold.csv", index=False)
        print(f"\nWrote {args.out_prefix}_loso_results.csv + "
              f"{args.out_prefix}_loso_per_fold.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
