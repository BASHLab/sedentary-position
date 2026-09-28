#!/usr/bin/env python3
"""What are channels 3-5 of the LIMU-BERT `lb_80_240` pretraining array?

It matters: LIMU-BERT has no input-normalisation layer, so fine-tuning it means
handing it channels in the same semantics and scale it was pretrained on. The
script that built `dataset/lb/data_80_240.npy` is not in the tree (it references
a path on a retired machine), and the array itself is ambiguous from summary
statistics alone -- channels 3-5 have vector magnitude ~0.53, which fits either

    (a) magnetometer  (~0.90 in this corpus, diluted by zero-padded windows), or
    (b) gyroscope in rad/s  (~0.5 for all-day recordings)

These two are easy to tell apart by temporal signature rather than by scale:

    magnetometer  slowly varying -- within a 3 s window it is nearly constant,
                  so within-window std << across-window std of the window means,
                  and the window mean has magnitude ~0.9
    gyroscope     zero-mean and fast -- within-window std is most of the total
                  variance and window means sit near 0

We measure that ratio on their array and on a raw LittleBeats `_imu_sync.txt`
(where we know ch 3-5 = magnetometer and ch 6-8 = gyro from docs/METHODS.md),
then see which our reference channels the mystery channels behave like.

Read-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

RSA = Path("/work/hdd/bebr/Projects/Infant_states_and_RSA/motion/limu")
PRE = RSA / "dataset/lb/data_80_240.npy"
LB = Path("/work/hdd/bebr/Data/LB/cleaned/BRP1")

SEG = 240  # their window length


def signature(w: np.ndarray, name: str) -> dict:
    """w: (n_windows, 3, seq) -- one 3-axis sensor's windows."""
    within = w.std(axis=2).mean()                 # mean wobble inside a window
    wmean = w.mean(axis=2)                        # (n, 3) window means
    across = wmean.std(axis=0).mean()             # spread of those means
    mag = np.linalg.norm(wmean, axis=1)
    out = dict(within=within, across=across,
               ratio=within / (across + 1e-12),
               mean_mag=mag.mean(), frac_near_zero=float((mag < 0.05).mean()))
    print(f"  {name:<34} within-window std {out['within']:8.4f} | "
          f"across-window std {out['across']:8.4f} | ratio {out['ratio']:7.3f} | "
          f"|window mean| {out['mean_mag']:7.3f} | {out['frac_near_zero']*100:5.1f}% ~0")
    return out


def main() -> None:
    print("REFERENCE -- a raw LittleBeats recording, known channel semantics")
    print("  (ch 0-2 accel m/s^2, ch 3-5 magnetometer, ch 6-8 gyro deg/s)\n")
    sess = next(s for fid in sorted(LB.iterdir()) if fid.is_dir() and fid.name == "25003"
                for ctx in sorted(fid.iterdir()) if ctx.is_dir()
                for s in sorted(ctx.iterdir()) if s.is_dir()
                and (s / f"{s.name}_imu_sync.txt").exists())
    print(f"  {sess.name}")
    imu = np.loadtxt(str(sess / sess.name) + "_imu_sync.txt")     # (9, n) @ 70 Hz
    n = imu.shape[1] // SEG
    w = imu[:, :n * SEG].reshape(9, n, SEG).transpose(1, 0, 2)    # (n, 9, seq)
    rng = np.random.default_rng(0)
    w = w[rng.choice(len(w), size=min(3000, len(w)), replace=False)]
    ref = {
        "accel m/s^2 (ch 0-2)": signature(w[:, 0:3], "accel m/s^2 (ch 0-2)"),
        "MAGNETOMETER (ch 3-5)": signature(w[:, 3:6], "MAGNETOMETER (ch 3-5)"),
        "gyro deg/s (ch 6-8)": signature(w[:, 6:9], "gyro deg/s (ch 6-8)"),
    }
    g = w[:, 6:9] * np.pi / 180.0
    ref["gyro rad/s (derived)"] = signature(g, "gyro rad/s (derived)")

    print(f"\n\nTHE MYSTERY ARRAY -- {PRE.name}")
    a = np.load(PRE, mmap_mode="r")
    print(f"  shape {a.shape}\n")
    idx = np.sort(rng.choice(a.shape[0], size=3000, replace=False))
    s = np.asarray(a[idx], dtype=np.float64).transpose(0, 2, 1)   # (n, 6, seq)
    got03 = signature(s[:, 0:3], "their ch 0-2")
    got35 = signature(s[:, 3:6], "their ch 3-5   <-- the question")

    print("\n\nVERDICT")
    cands = {"magnetometer": ref["MAGNETOMETER (ch 3-5)"],
             "gyro deg/s": ref["gyro deg/s (ch 6-8)"],
             "gyro rad/s": ref["gyro rad/s (derived)"]}
    print(f"  their ch 3-5 ratio = {got35['ratio']:.3f}, "
          f"|window mean| = {got35['mean_mag']:.3f}")
    for k, v in cands.items():
        print(f"    vs {k:<14} ratio {v['ratio']:7.3f}  |window mean| {v['mean_mag']:7.3f}"
              f"   ratio off by {abs(np.log10((got35['ratio']+1e-9)/(v['ratio']+1e-9))):.2f} decades")
    best = min(cands, key=lambda k: abs(np.log10((got35["ratio"] + 1e-9)
                                                 / (cands[k]["ratio"] + 1e-9))))
    print(f"\n  closest by temporal signature: {best}")
    print(f"  their ch 0-2 |window mean| {got03['mean_mag']:.3f} vs our accel "
          f"{ref['accel m/s^2 (ch 0-2)']['mean_mag']:.3f} m/s^2 "
          f"-> scale factor {got03['mean_mag'] / ref['accel m/s^2 (ch 0-2)']['mean_mag']:.3f}")


if __name__ == "__main__":
    main()
