#!/usr/bin/env python3
"""Raw 10 s IMU windows, resampled to 30 Hz, for deep models.

`windows_10s.parquet` holds only the 19 summary features, which is all the GBDT
baseline needed. A pretrained encoder needs the waveform, so this re-runs the
identical windowing and keeps the signal.

Output is shaped for Oxford ssl-wearables / harnet10:
  - 30 Hz (harnet's pretraining rate), 10 s -> 300 samples
  - accelerometer in **g** (harnet's unit), not m/s^2
  - gyro carried along at the same rate for models that can use it

The window selection is asserted identical to windows_10s.parquet, so the deep
model and the GBDT baseline are scored on exactly the same rows.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

PROJ = Path("/work/hdd/bebr/Projects/IMU_pretraining")
LB_DIR = Path("/work/hdd/bebr/Data/LB/cleaned/BRP1")
OUT_DIR = PROJ / "analysis" / "posture"

FS_IN, FS_OUT = 70.0, 30.0
UP, DOWN = 3, 7  # 70 * 3/7 = 30 exactly
WIN_S, HOP_S, MIN_PURITY = 10.0, 5.0, 0.80
DROP_CLASSES = {"recline forward"}
G = 9.80665

WIN_IN = int(WIN_S * FS_IN)    # 700
WIN_OUT = int(WIN_S * FS_OUT)  # 300


def session_index() -> dict[str, Path]:
    return {
        s.name: s
        for fid in LB_DIR.iterdir() if fid.is_dir()
        for ctx in fid.iterdir() if ctx.is_dir()
        for s in ctx.iterdir() if s.is_dir()
    }


def main() -> None:
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

        # accel -> g, gyro left in deg/s; magnetometer (3:6) deliberately unused
        sig = np.vstack([imu[0:3] / G, imu[6:9]]).astype(np.float64)

        for start in range(0, n - WIN_IN + 1, hop_n):
            lab = labels[start:start + WIN_IN]
            coded = lab[lab >= 0]
            if coded.size < WIN_IN * MIN_PURITY:
                continue
            vals, counts = np.unique(coded, return_counts=True)
            if counts.max() < WIN_IN * MIN_PURITY:
                continue
            w = resample_poly(sig[:, start:start + WIN_IN], UP, DOWN, axis=1)
            X.append(w.astype(np.float32))
            meta.append((fid, obs, lb_session, start / FS_IN,
                         classes[vals[counts.argmax()]]))

        if k % 10 == 0 or k == len(groups):
            print(f"  [{k:3d}/{len(groups)}] {len(X):6d} windows "
                  f"({time.time() - t0:.0f}s)", flush=True)

    X = np.stack(X)
    md = pd.DataFrame(meta, columns=["family_id", "obs", "lb_session",
                                     "t_start_s", "posture"])
    assert X.shape[1:] == (6, WIN_OUT), X.shape

    # Same rows, same order as the feature table the GBDT baseline used.
    ref = pd.read_parquet(OUT_DIR / "windows_10s.parquet")
    cols = ["family_id", "obs", "lb_session", "t_start_s", "posture"]
    pd.testing.assert_frame_equal(md[cols].reset_index(drop=True),
                                  ref[cols].reset_index(drop=True))
    print(f"\nwindow selection identical to windows_10s.parquet ({len(md)} rows)")

    np.save(OUT_DIR / "windows_raw_30hz.npy", X)
    md.to_parquet(OUT_DIR / "windows_raw_30hz_meta.parquet", index=False)
    print(f"X {X.shape} float32 = {X.nbytes / 1e6:.0f} MB "
          f"-> windows_raw_30hz.npy")
    print("channels: 0-2 accel (g), 3-5 gyro (deg/s)")
    acc = X[:, :3]
    print(f"accel g:  mean |a| = {np.linalg.norm(acc, axis=1).mean():.3f} "
          f"(expect ~1.0)   range [{acc.min():.2f}, {acc.max():.2f}]")


if __name__ == "__main__":
    main()
