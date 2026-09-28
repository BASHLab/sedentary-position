#!/usr/bin/env python3
"""Sleep vs wake per posture window, from the infant ECG foundation model.

The model lives in /work/nvme/bebr/mkhan14/ecg_foundation_model: each 30 s ECG
window is a graph of heartbeats (per-beat CNN -> +RR features -> transformer ->
GNN -> attention pool), self-supervised on the BCP cohort and fine-tuned on BRP
for sleep vs wake. We import its code rather than reimplement the beat/IBI
feature assembly, so the tensors handed to the classifier are built by the same
functions that built its training data.

Two things about that repo have to be worked around, both recorded here because
they are not obvious:

1. `single_stream_downstream/dataset.py` imports `ibi_graph_model.signal_utils`
   and `dual_stream_pretraining.signal_utils`, neither of which exists in the
   checkout any more, and it reads per-recording IBI `.npz` files that are also
   gone. All four functions it wants survive in
   `single_stream_pretraining/signal_utils.py`, and the IBI it needs is just the
   R-peak differences, so this script rebuilds the window tensors directly from
   the `.wav` + `_peaks.npy` pair instead of importing that dataset class.

2. **A time-base offset their pipeline does not correct.** The `ecg_v2` wavs do
   not start at the recording start: for BRP1_25008_2025-10-09-09-39-05 the wav
   is 4180.6 s against the cleaned recording's 4214.0 s and its first sample is
   ~4.35 s late. Their dataset indexes the wav at `t_start * sr` with `t_start`
   on the tone/annotation clock, so every training window was shifted by that
   per-recording amount. Here the offset is measured per recording, from the
   ECG parquet's wall-clock `datetime` against the LittleBeats recording start,
   and subtracted -- so these predictions land on the same LB clock as the IMU
   windows they will be joined to.

Caveat worth carrying into any result: the sleep model was trained on BRP1
25003-25006 and validated on 25008, so for five of our six infants these
predictions are in-sample. 25007 has no `ecg_v2` recording at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ECG_REPO = Path("/work/nvme/bebr/mkhan14/ecg_foundation_model")
PROJ = Path("/work/hdd/bebr/Projects/IMU_pretraining")
OUT_DIR = PROJ / "analysis" / "posture"
ECG_V2 = Path("/work/hdd/bebr/Data/LB/ecg_v2/BRP1")
LB_CLEAN = Path("/work/hdd/bebr/Data/LB/cleaned/BRP1")

WIN_S = 10.0        # our posture window
SLEEP_CLASSES = ["sleep", "wake"]   # label2idx order, see note in main()


def _imports():
    sys.path.insert(0, str(ECG_REPO))
    sys.path.insert(0, str(ECG_REPO / "single_stream_downstream"))
    from single_stream_pretraining.signal_utils import (  # noqa: E402
        bandpass_filter, build_ibi_features, clean_ibi, extract_beats)
    from config import TASK_CONFIGS                        # noqa: E402
    from model import SingleStreamDownstreamClassifier     # noqa: E402
    return (bandpass_filter, extract_beats, clean_ibi, build_ibi_features,
            TASK_CONFIGS, SingleStreamDownstreamClassifier)


# --------------------------------------------------------------- session map
def ecg_session_map() -> dict[str, dict]:
    """our lb_session -> {wav, peaks, offset_s}, offset on the LB clock.

    `ecg_v2` drops the REC token from the directory name, so the join key is
    (family_id, start-datetime), which both namings carry verbatim.
    """
    clean_start = {}
    for fid in sorted(p for p in LB_CLEAN.iterdir() if p.is_dir()):
        for ctx in sorted(p for p in fid.iterdir() if p.is_dir()):
            for s in sorted(p for p in ctx.iterdir() if p.is_dir()):
                ts = s / f"{s.name}_audio_sync_timestamp.txt"
                if not ts.exists():
                    continue
                with open(ts) as f:
                    t0 = float(f.readline().split()[0])
                # BRP1_<fid>_<REC...>_<YYYY-MM-DD-HH-MM-SS>
                dt = "-".join(s.name.split("_")[-1:])
                clean_start[(fid.name, dt)] = (s.name, t0)

    out = {}
    for fid_dir in sorted(p for p in ECG_V2.iterdir() if p.is_dir()):
        for sd in sorted(p for p in fid_dir.iterdir() if p.is_dir()):
            wav = sd / f"{sd.name}_ecg.wav"
            pk = sd / f"{sd.name}_ecg_peaks.npy"
            pq = sd / f"{sd.name}_ecg.parquet"
            if not (wav.exists() and pk.exists()):
                continue
            dt = sd.name.split("_")[-1]
            key = (fid_dir.name, dt)
            if key not in clean_start:
                continue
            lb_session, t0 = clean_start[key]
            offset = 0.0
            if pq.exists():
                try:
                    import pyarrow.parquet as papq
                    # first row group only -- the full column is 4.2M rows and
                    # we need exactly one value
                    f = papq.ParquetFile(pq)
                    d0 = (f.read_row_group(0, columns=["datetime"])
                           .column("datetime")[0].as_py())
                    offset = float(d0.timestamp()) - t0
                except Exception as e:                       # pragma: no cover
                    print(f"  ! {sd.name}: parquet datetime unreadable ({e}); "
                          f"offset assumed 0")
            out[lb_session] = {"wav": wav, "peaks": pk, "offset_s": offset}
    return out


# ------------------------------------------------------------ window tensors
def build_window(full_ecg, fns, cfg, all_peaks, t_wav):
    """One 30 s ECG window -> (beats, rr, ibi_feats, valid_mask) or None.

    `full_ecg` is the whole recording held in memory (~17 MB at 1 kHz float32);
    slicing it beats a seeking read per window by a wide margin on spinning disk.
    """
    bandpass_filter, extract_beats, clean_ibi, build_ibi_features = fns
    sr = cfg.sampling_rate
    start = int(round(t_wav * sr))
    n = int(cfg.window_sec * sr) + 1
    if start < 0 or start >= len(full_ecg):
        return None
    ecg = np.asarray(full_ecg[start:start + n], dtype=np.float32)
    if len(ecg) < int(0.9 * cfg.window_sec * sr) or float(ecg.std()) < 1e-6:
        return None
    try:
        ecg = bandpass_filter(ecg, sr)
    except Exception:
        return None

    lo, hi = start, start + len(ecg)
    rp = all_peaks[(all_peaks >= lo) & (all_peaks < hi)] - start
    if len(rp) < 3:
        return None

    beats, rr = extract_beats(
        ecg, rp, sr, max_beats=cfg.max_len_beats,
        pre_ratio=cfg.pre_ratio, post_ratio=cfg.post_ratio,
        min_pre_ms=cfg.min_pre_ms, min_post_ms=cfg.min_post_ms,
        max_pre_ms=cfg.max_pre_ms, max_post_ms=cfg.max_post_ms)
    if len(beats) == 0:
        return None

    # The IBI stream their dataset loaded from .npz is just the R-peak diffs.
    ibi = np.diff(rp).astype(np.float32) / sr
    if len(ibi) < 2:
        ibi = np.zeros(1, dtype=np.float32)
    ibi_clean, quality = clean_ibi(ibi, cfg.min_ibi_ms, cfg.max_ibi_ms)
    ibi_feats = build_ibi_features(ibi_clean, quality, cfg.max_len_beats)

    N = max(1, min(len(ibi_feats), len(beats)))
    M, L = cfg.max_len_beats, beats.shape[1]
    ob = np.zeros((M, L), np.float32)
    orr = np.zeros((M, 2), np.float32)
    of = np.zeros((M, cfg.ibi_feature_dim), np.float32)
    vm = np.zeros(M, bool)
    ob[:min(N, len(beats))] = beats[:min(N, len(beats))]
    orr[:min(N, len(rr))] = rr[:min(N, len(rr))]
    of[:min(N, len(ibi_feats))] = ibi_feats[:min(N, len(ibi_feats))]
    vm[:N] = True
    return ob, orr, of, vm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results_sleep_v5_linear",
                    help="directory under single_stream_downstream/")
    ap.add_argument("--ckpt", default="best_model.pt",
                    choices=["best_model.pt", "last_model.pt"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit-sessions", type=int, default=0,
                    help="smoke mode: only this many recordings")
    args = ap.parse_args()

    import soundfile as sf
    (bandpass_filter, extract_beats, clean_ibi, build_ibi_features,
     TASK_CONFIGS, Classifier) = _imports()
    fns = (bandpass_filter, extract_beats, clean_ibi, build_ibi_features)

    cfg = TASK_CONFIGS["sleep"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {dev} | window {cfg.window_sec}s @ {cfg.sampling_rate}Hz | "
          f"max {cfg.max_len_beats} beats | {cfg.num_classes} classes")

    ck = ECG_REPO / "single_stream_downstream" / args.run / args.ckpt
    model = Classifier(cfg).to(dev)
    state = torch.load(ck, map_location=dev, weights_only=False)
    sd = state.get("model", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"checkpoint {args.run}/{args.ckpt}: "
          f"{len(sd)} tensors, missing {len(missing)}, unexpected {len(unexpected)}")
    if missing:
        print("  missing:", list(missing)[:6])
    model.eval()

    md = pd.read_parquet(OUT_DIR / "windows_raw_30hz_meta.parquet").reset_index(drop=True)
    smap = ecg_session_map()
    have = md["lb_session"].isin(smap)
    print(f"\n{len(md):,} windows; ECG available for {int(have.sum()):,} "
          f"({have.mean():.1%}) across {md.loc[have, 'lb_session'].nunique()} "
          f"of {md['lb_session'].nunique()} recordings")
    miss = sorted(set(md.loc[~have, "family_id"]))
    if miss:
        for fid in miss:
            n = int((md["family_id"] == fid).sum() - have[md["family_id"] == fid].sum())
            print(f"  infant {fid}: {n:,} windows without ECG")

    sessions = [s for s in md["lb_session"].unique() if s in smap]
    if args.limit_sessions:
        sessions = sessions[:args.limit_sessions]
        print(f"  SMOKE: first {len(sessions)} recordings only")

    rows = []
    for si, sess in enumerate(sessions, 1):
        info = smap[sess]
        peaks = np.load(info["peaks"])
        full_ecg, _sr = sf.read(str(info["wav"]), dtype="float32", always_2d=False)
        if full_ecg.ndim > 1:
            full_ecg = full_ecg[:, 0]
        sub = md[md["lb_session"] == sess]
        buf, keys = [], []

        def flush():
            if not buf:
                return
            L = max(b[0].shape[1] for b in buf)
            B = len(buf)
            beats = torch.zeros(B, cfg.max_len_beats, L)
            rr = torch.zeros(B, cfg.max_len_beats, 2)
            feats = torch.zeros(B, cfg.max_len_beats, cfg.ibi_feature_dim)
            vm = torch.zeros(B, cfg.max_len_beats, dtype=torch.bool)
            for i, (b, r, f, m) in enumerate(buf):
                beats[i, :, :b.shape[1]] = torch.from_numpy(b)
                rr[i] = torch.from_numpy(r)
                feats[i] = torch.from_numpy(f)
                vm[i] = torch.from_numpy(m)
            with torch.no_grad():
                out = model(beats.to(dev), rr.to(dev), feats.to(dev), vm.to(dev))
                p = torch.softmax(out["logits"], 1).cpu().numpy()
            for k, pr in zip(keys, p):
                rows.append({**k, "p_sleep": float(pr[0]), "p_wake": float(pr[1])})
            buf.clear()
            keys.clear()

        for _, r in sub.iterrows():
            centre = r["t_start_s"] + WIN_S / 2.0
            t_wav = centre - cfg.window_sec / 2.0 - info["offset_s"]
            w = build_window(full_ecg, fns, cfg, peaks, t_wav)
            if w is None:
                continue
            buf.append(w)
            keys.append({"family_id": int(r["family_id"]), "obs": r["obs"],
                         "lb_session": sess, "t_start_s": float(r["t_start_s"])})
            if len(buf) >= args.batch:
                flush()
        flush()
        if si % 10 == 0 or si == len(sessions):
            print(f"  [{si:3d}/{len(sessions)}] {len(rows):6d} windows scored",
                  flush=True)

    if not rows:
        print("no windows scored"); return
    df = pd.DataFrame(rows)
    df["ecg_state"] = np.where(df["p_wake"] >= 0.5, "wake", "sleep")
    out = OUT_DIR / ("ecg_sleep_predictions_smoke.parquet" if args.limit_sessions
                     else "ecg_sleep_predictions.parquet")
    df.to_parquet(out, index=False)
    print(f"\n{len(df):,} windows scored -> "
          f"wake {(df.ecg_state == 'wake').mean():.1%}, "
          f"sleep {(df.ecg_state == 'sleep').mean():.1%}")
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
