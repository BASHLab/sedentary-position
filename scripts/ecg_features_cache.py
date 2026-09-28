#!/usr/bin/env python3
"""Cache the ECG per-window model inputs once, so folds are cheap to train.

Building the heartbeat-graph tensors for one window means reading 30 s of ECG,
band-passing it, slicing beats around the precomputed R-peaks and deriving the
IBI features. That is the whole cost of `ecg_sleep_infer.py` (~27 min for the
corpus) and it does not depend on any model weights -- so doing it once here
makes a leave-one-subject-out fine-tune, which needs the same windows five or
six times over, essentially free.

Layout, matching `single_stream_downstream`'s collator:

    beats      (N, 80, L_MAX)  float16   L_MAX = max_pre_ms + max_post_ms = 1000
    beat_len   (N,)            int16     the true beat length for that window
    rr         (N, 80, 2)      float32
    ibi_feats  (N, 80, 10)     float32
    valid_mask (N, 80)         bool
    meta       parquet         family_id, obs, lb_session, t_start_s, posture

`beat_len` matters: their collator pads each batch to the batch's own maximum,
not to a global one, and beat length tracks median RR, so padding everything to
1000 would feed the per-beat CNN far more zero-padding than it ever saw in
training. Storing the true length lets the loader reproduce the batch-local
padding exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/work/hdd/bebr/Projects/IMU_pretraining")
OUT_DIR = PROJ / "analysis" / "posture"
ECG_REPO = Path("/work/nvme/bebr/mkhan14/ecg_foundation_model")

sys.path.insert(0, str(PROJ / "scripts"))
from ecg_sleep_infer import (WIN_S, _imports, build_window,  # noqa: E402
                             ecg_session_map)


def main() -> None:
    import soundfile as sf
    (bandpass_filter, extract_beats, clean_ibi, build_ibi_features,
     TASK_CONFIGS, _) = _imports()
    fns = (bandpass_filter, extract_beats, clean_ibi, build_ibi_features)
    cfg = TASK_CONFIGS["sleep"]
    L_MAX = cfg.max_pre_ms + cfg.max_post_ms          # 1000 at 1 kHz
    M = cfg.max_len_beats

    md = pd.read_parquet(OUT_DIR / "windows_raw_30hz_meta.parquet").reset_index(drop=True)
    smap = ecg_session_map()
    sessions = [s for s in md["lb_session"].unique() if s in smap]
    sel = md[md["lb_session"].isin(smap)]
    print(f"{len(md):,} windows; {len(sel):,} in {len(sessions)} ECG recordings")
    print(f"beats padded to L_MAX={L_MAX}; true length kept per window")

    n = len(sel)
    beats = np.lib.format.open_memmap(
        OUT_DIR / "ecg_cache_beats.npy", mode="w+", dtype=np.float16,
        shape=(n, M, L_MAX))
    rr = np.zeros((n, M, 2), np.float32)
    feats = np.zeros((n, M, cfg.ibi_feature_dim), np.float32)
    vmask = np.zeros((n, M), bool)
    blen = np.zeros(n, np.int16)
    rows, k = [], 0

    for si, sess in enumerate(sessions, 1):
        info = smap[sess]
        peaks = np.load(info["peaks"])
        ecg, _ = sf.read(str(info["wav"]), dtype="float32", always_2d=False)
        if ecg.ndim > 1:
            ecg = ecg[:, 0]
        for _, r in md[md["lb_session"] == sess].iterrows():
            centre = r["t_start_s"] + WIN_S / 2.0
            t_wav = centre - cfg.window_sec / 2.0 - info["offset_s"]
            w = build_window(ecg, fns, cfg, peaks, t_wav)
            if w is None:
                continue
            b, rr_, f_, m_ = w
            L = min(b.shape[1], L_MAX)
            beats[k, :, :L] = b[:, :L].astype(np.float16)
            blen[k] = L
            rr[k], feats[k], vmask[k] = rr_, f_, m_
            rows.append((int(r["family_id"]), r["obs"], sess,
                         float(r["t_start_s"]), r["posture"]))
            k += 1
        if si % 10 == 0 or si == len(sessions):
            print(f"  [{si:3d}/{len(sessions)}] {k:6d} windows cached", flush=True)

    beats.flush()
    meta = pd.DataFrame(rows, columns=["family_id", "obs", "lb_session",
                                       "t_start_s", "posture"])
    # trim the over-allocated tail
    np.save(OUT_DIR / "ecg_cache_rr.npy", rr[:k])
    np.save(OUT_DIR / "ecg_cache_ibifeats.npy", feats[:k])
    np.save(OUT_DIR / "ecg_cache_validmask.npy", vmask[:k])
    np.save(OUT_DIR / "ecg_cache_beatlen.npy", blen[:k])
    meta.to_parquet(OUT_DIR / "ecg_cache_meta.parquet", index=False)
    print(f"\ncached {k:,} windows")
    print(f"  beats memmap holds {n:,} rows; first {k:,} are valid "
          f"(loaders must slice to len(meta))")
    print(f"  beat_len: min {blen[:k].min()} max {blen[:k].max()} "
          f"median {int(np.median(blen[:k]))}")
    print(f"  per infant: "
          + ", ".join(f"{f}:{c:,}" for f, c in
                      meta['family_id'].value_counts().sort_index().items()))


if __name__ == "__main__":
    main()
