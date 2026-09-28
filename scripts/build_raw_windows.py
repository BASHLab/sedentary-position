#!/usr/bin/env python3
"""Raw 10 s IMU windows at a pretrained encoder's native rate.

`windows_10s.parquet` holds only the 19 summary features, which is all the GBDT
baseline needed. A pretrained encoder needs the waveform, so this re-runs the
identical windowing and keeps the signal.

    python build_raw_windows.py --fs-out 30   # harnet10   -> (N, 6, 300)
    python build_raw_windows.py --fs-out 80   # LIMU-BERT  -> (N, 6, 800)

Every encoder is fed at the rate it was pretrained at rather than being asked to
generalise across a sample-rate shift, and 70 Hz resamples to both exactly
(x3/7 and x8/7), so no interpolation artefacts are introduced. Channel layout
and units are the same either way:
  - 0-2 accelerometer in **g**, not m/s^2
  - 3-5 gyro in deg/s
  - magnetometer deliberately dropped (see docs/METHODS.md)

The window selection is asserted identical to windows_10s.parquet, so every
model in this project is scored on exactly the same rows in the same order.
"""

from __future__ import annotations

import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

PROJ = Path("/work/hdd/bebr/Projects/IMU_pretraining")
LB_DIR = Path("/work/hdd/bebr/Data/LB/cleaned/BRP1")
OUT_DIR = PROJ / "analysis" / "posture"

FS_IN = 70.0
# Output rates the encoders want, each an exact rational resample of 70 Hz:
#   30 -> harnet10 (Oxford ssl-wearables pretraining rate)
#   80 -> LIMU-BERT lb_80_240 (Infant_states_and_RSA pretraining rate)
RATES = {30: (3, 7), 80: (8, 7)}
WIN_S, HOP_S, MIN_PURITY = 10.0, 5.0, 0.80
DROP_CLASSES = {"recline forward"}
G = 9.80665

WIN_IN = int(WIN_S * FS_IN)    # 700


def session_index() -> dict[str, Path]:
    return {
        s.name: s
        for fid in LB_DIR.iterdir() if fid.is_dir()
        for ctx in fid.iterdir() if ctx.is_dir()
        for s in ctx.iterdir() if s.is_dir()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs-out", type=int, default=30, choices=sorted(RATES),
                    help="output sample rate; names the output files")
    ap.add_argument("--all-channels", action="store_true",
                    help="keep all 9 channels in raw order (accel, mag, gyro) "
                         "instead of the 6-channel accel+gyro default. The "
                         "magnetometer is excluded from every model in this "
                         "project (docs/METHODS.md); it is carried here only so "
                         "an encoder pretrained on accel+mag can be fed the "
                         "channels it actually saw.")
    args = ap.parse_args()
    fs_out = args.fs_out
    up, down = RATES[fs_out]
    win_out = int(WIN_S * fs_out)
    n_ch = 9 if args.all_channels else 6
    tag = f"windows_raw_{fs_out}hz" + ("_9ch" if args.all_channels else "")
    print(f"{FS_IN:.0f} Hz -> {fs_out} Hz (x{up}/{down}), "
          f"{WIN_S:.0f} s windows = {win_out} samples -> {tag}.npy")

    seg = pd.read_csv(OUT_DIR / "posture_segments.csv")
    seg = seg[seg["imu_alignable"] & ~seg["posture"].isin(DROP_CLASSES)]
    sessions = session_index()
    classes = sorted(seg["posture"].unique())
    cls_idx = {c: i for i, c in enumerate(classes)}
    hop_n = int(HOP_S * FS_IN)

    X, meta = [], []
    groups = list(seg.groupby(["family_id", "obs", "lb_session"], sort=True))
    t0 = time.time()

    for k, ((fid, obs, lb_session), g) in enumerate(groups, 1):
        sess = sessions.get(lb_session)
        if sess is None:
            print(f"  ! no session dir for {lb_session}", flush=True)
            continue
        imu = np.loadtxt(str(sess / sess.name) + "_imu_sync.txt")
        n = imu.shape[1]

        labels = np.full(n, -1, dtype=np.int8)
        for _, r in g.iterrows():
            i0 = max(int(round(r["lb_begin_s"] * FS_IN)), 0)
            i1 = min(int(round(r["lb_end_s"] * FS_IN)), n)
            if i1 > i0:
                labels[i0:i1] = cls_idx[r["posture"]]

        # accel -> g always; magnetometer (raw 3:6) only when asked for
        sig = (np.vstack([imu[0:3] / G, imu[3:6], imu[6:9]]) if n_ch == 9
               else np.vstack([imu[0:3] / G, imu[6:9]])).astype(np.float64)

        for start in range(0, n - WIN_IN + 1, hop_n):
            lab = labels[start:start + WIN_IN]
            coded = lab[lab >= 0]
            if coded.size < WIN_IN * MIN_PURITY:
                continue
            vals, counts = np.unique(coded, return_counts=True)
            if counts.max() < WIN_IN * MIN_PURITY:
                continue
            w = resample_poly(sig[:, start:start + WIN_IN], up, down, axis=1)
            X.append(w.astype(np.float32))
            meta.append((fid, obs, lb_session, start / FS_IN,
                         classes[vals[counts.argmax()]]))

        if k % 10 == 0 or k == len(groups):
            print(f"  [{k:3d}/{len(groups)}] {len(X):6d} windows "
                  f"({time.time() - t0:.0f}s)", flush=True)

    X = np.stack(X)
    md = pd.DataFrame(meta, columns=["family_id", "obs", "lb_session",
                                     "t_start_s", "posture"])
    assert X.shape[1:] == (n_ch, win_out), X.shape

    # Same rows, same order as the feature table the GBDT baseline used.
    ref = pd.read_parquet(OUT_DIR / "windows_10s.parquet")
    cols = ["family_id", "obs", "lb_session", "t_start_s", "posture"]
    pd.testing.assert_frame_equal(md[cols].reset_index(drop=True),
                                  ref[cols].reset_index(drop=True))
    print(f"\nwindow selection identical to windows_10s.parquet ({len(md)} rows)")

    np.save(OUT_DIR / f"{tag}.npy", X)
    md.to_parquet(OUT_DIR / f"{tag}_meta.parquet", index=False)
    print(f"X {X.shape} float32 = {X.nbytes / 1e6:.0f} MB "
          f"-> {tag}.npy")
    print("channels: " + ("0-2 accel (g), 3-5 magnetometer, 6-8 gyro (deg/s)"
                          if n_ch == 9 else "0-2 accel (g), 3-5 gyro (deg/s)"))
    acc = X[:, :3]
    print(f"accel g:  mean |a| = {np.linalg.norm(acc, axis=1).mean():.3f} "
          f"(expect ~1.0)   range [{acc.min():.2f}, {acc.max():.2f}]")


if __name__ == "__main__":
    main()
