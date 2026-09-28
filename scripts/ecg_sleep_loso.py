#!/usr/bin/env python3
"""Leave-one-subject-out sleep/wake from the ECG foundation model.

Why this exists: the ready-made sleep checkpoint in
`ecg_foundation_model/single_stream_downstream/results_sleep_v5_linear/` was
trained on BRP1 25003-25006 and validated on 25008, which is four of our six
infants in-sample and the fifth used for checkpoint selection. Tummy time built
on it therefore has no infant that is clean of the ECG model.

This starts again from the **self-supervised** checkpoint
(`single_stream_pretraining/experiments/single_stream_v1/last_checkpoint.pt`,
which never saw a sleep label) and fine-tunes it once per infant with that
infant held out, so every window gets an out-of-fold prediction. That is the
same protocol harnet10 already uses for posture, so the two halves of tummy time
become comparable: neither model has seen the infant it is predicting.

Labels are the human `state` tier mapped by the ECG repo's own vocabulary
(wake: quiet alert / active alert / active / crying; sleep: drowsy /
drowsy unsure / light sleep / deep sleep), materialised per window by
`tummy_time.py`.

Training recipe is copied from `single_stream_downstream/main.py` so the
comparison is against their method, not a different one: AdamW, head lr 1e-3,
encoder lr 1e-5, weight decay 1e-4, label smoothing 0.05, inverse-frequency
class weights, batch 64. Epoch selection uses a 10% split of the *training*
infants, never the held-out one.

25007 has no `ecg_v2` recording and cannot be predicted at all.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, roc_auc_score)

PROJ = Path("/work/hdd/bebr/Projects/IMU_pretraining")
OUT_DIR = PROJ / "analysis" / "posture"
ECG_REPO = Path("/work/nvme/bebr/mkhan14/ecg_foundation_model")
SSL_CKPT = (ECG_REPO / "single_stream_pretraining/experiments/single_stream_v1"
            / "last_checkpoint.pt")

EPOCHS, PATIENCE, BATCH = 30, 10, 64
LR_HEAD, LR_ENC, WD, LABEL_SMOOTH = 1e-3, 1e-5, 1e-4, 0.05
VAL_FRAC, SEED = 0.10, 0


def _imports():
    sys.path.insert(0, str(ECG_REPO))
    sys.path.insert(0, str(ECG_REPO / "single_stream_downstream"))
    from config import TASK_CONFIGS                      # noqa: E402
    from model import SingleStreamDownstreamClassifier   # noqa: E402
    return TASK_CONFIGS, SingleStreamDownstreamClassifier


class Cache:
    """The precomputed window tensors, sliced per batch like their collator."""

    def __init__(self, out_dir: Path):
        self.meta = pd.read_parquet(out_dir / "ecg_cache_meta.parquet")
        n = len(self.meta)
        self.beats = np.load(out_dir / "ecg_cache_beats.npy", mmap_mode="r")[:n]
        self.rr = np.load(out_dir / "ecg_cache_rr.npy")
        self.feats = np.load(out_dir / "ecg_cache_ibifeats.npy")
        self.vmask = np.load(out_dir / "ecg_cache_validmask.npy")
        self.blen = np.load(out_dir / "ecg_cache_beatlen.npy")

    def batch(self, idx: np.ndarray, dev) -> tuple[torch.Tensor, ...]:
        L = int(self.blen[idx].max())
        b = torch.from_numpy(np.asarray(self.beats[idx][:, :, :L],
                                        dtype=np.float32))
        return (b.to(dev),
                torch.from_numpy(self.rr[idx]).to(dev),
                torch.from_numpy(self.feats[idx]).to(dev),
                torch.from_numpy(self.vmask[idx]).to(dev))


@torch.no_grad()
def predict(model, cache, idx, dev, bs=256):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        sl = idx[i:i + bs]
        logits = model(*cache.batch(sl, dev))["logits"]
        out.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def run_fold(cache, y, tr_idx, te_idx, dev, Classifier, cfg, rng):
    model = Classifier(cfg).to(dev)
    model.load_pretrained(str(SSL_CKPT), dev, strict=False)

    rng.shuffle(tr_idx)
    n_val = int(len(tr_idx) * VAL_FRAC)
    va, tr = tr_idx[:n_val], tr_idx[n_val:]

    cnt = np.bincount(y[tr], minlength=2)
    w = torch.tensor(len(tr) / (2 * np.maximum(cnt, 1)), dtype=torch.float32,
                     device=dev)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)
    opt = torch.optim.AdamW(
        [{"params": list(model.cls_head.parameters()), "lr": LR_HEAD},
         {"params": [p for p in list(model.encoder.parameters())
                     + list(model.pool_fused.parameters()) if p.requires_grad],
          "lr": LR_ENC}], weight_decay=WD)
    yt = torch.from_numpy(y).to(dev)

    best, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = tr[np.random.default_rng(SEED + ep).permutation(len(tr))]
        for i in range(0, len(perm), BATCH):
            sl = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            lossf(model(*cache.batch(sl, dev))["logits"], yt[sl]).backward()
            opt.step()
        vf1 = f1_score(y[va], (predict(model, cache, va, dev) >= 0.5).astype(int),
                       average="macro", zero_division=0)
        if vf1 > best:
            best, best_ep, bad = vf1, ep, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return predict(model, cache, te_idx, dev), best_ep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    globals()["EPOCHS"] = args.epochs

    TASK_CONFIGS, Classifier = _imports()
    cfg = TASK_CONFIGS["sleep"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {dev} | SSL checkpoint {SSL_CKPT.name}")

    cache = Cache(OUT_DIR)
    md = cache.meta
    print(f"{len(md):,} cached ECG windows")

    # labels: the coded state, already aligned per window by tummy_time.py
    w = pd.read_parquet(OUT_DIR / "tummy_time_windows.parquet")
    key = ["family_id", "obs", "t_start_s"]
    md = md.merge(w[key + ["coded_state"]], on=key, how="left")
    ok = md["coded_state"].isin(["wake", "sleep"]).to_numpy()
    y = (md["coded_state"] == "wake").to_numpy().astype(np.int64)
    gr = md["family_id"].to_numpy()
    print(f"  {ok.sum():,} have a coded state -> "
          f"wake {int(y[ok].sum()):,} sleep {int((~y[ok].astype(bool)).sum()):,}")

    subjects = sorted(np.unique(gr))
    p_wake = np.full(len(md), np.nan)
    rows = []
    for s in subjects:
        te = (gr == s) & ok
        tr = (gr != s) & ok
        if te.sum() == 0 or tr.sum() == 0:
            continue
        t0 = time.time()
        pr, ep = run_fold(cache, y, np.where(tr)[0], np.where(te)[0], dev,
                          Classifier, cfg, np.random.default_rng(SEED))
        p_wake[np.where(te)[0]] = pr
        yhat = (pr >= 0.5).astype(int)
        yt = y[te]
        r = {"held_out": int(s), "n": int(te.sum()),
             "accuracy": accuracy_score(yt, yhat),
             "balanced_acc": balanced_accuracy_score(yt, yhat),
             "macro_f1": f1_score(yt, yhat, average="macro", zero_division=0),
             "kappa": cohen_kappa_score(yt, yhat, labels=[0, 1]),
             "auroc": roc_auc_score(yt, pr) if len(np.unique(yt)) == 2 else np.nan,
             "best_epoch": ep}
        rows.append(r)
        print(f"  [fold {s}] n={r['n']:5d} acc={r['accuracy']:.3f} "
              f"bal={r['balanced_acc']:.3f} macroF1={r['macro_f1']:.3f} "
              f"kappa={r['kappa']:.3f} auroc={r['auroc']:.3f} "
              f"(epoch {ep}, {time.time()-t0:.0f}s)", flush=True)

    have = ~np.isnan(p_wake)
    pooled_hat = (p_wake[have & ok] >= 0.5).astype(int)
    yt = y[have & ok]
    print(f"\nPOOLED out-of-fold  n={len(yt):,} "
          f"acc={accuracy_score(yt, pooled_hat):.3f} "
          f"bal={balanced_accuracy_score(yt, pooled_hat):.3f} "
          f"macroF1={f1_score(yt, pooled_hat, average='macro', zero_division=0):.3f} "
          f"kappa={cohen_kappa_score(yt, pooled_hat, labels=[0,1]):.3f} "
          f"auroc={roc_auc_score(yt, p_wake[have & ok]):.3f}")

    out = cache.meta[["family_id", "obs", "lb_session", "t_start_s"]].copy()
    out["p_wake"] = p_wake
    out["p_sleep"] = 1.0 - p_wake
    out["ecg_state"] = np.where(np.isnan(p_wake), None,
                                np.where(p_wake >= 0.5, "wake", "sleep"))
    out = out[out["ecg_state"].notna()]
    out.to_parquet(OUT_DIR / "ecg_sleep_predictions_loso.parquet", index=False)
    pd.DataFrame(rows).to_csv(OUT_DIR / "ecg_sleep_loso_per_fold.csv", index=False)
    print(f"\n{len(out):,} out-of-fold predictions -> "
          f"ecg_sleep_predictions_loso.parquet")
    print("Wrote ecg_sleep_loso_per_fold.csv")


if __name__ == "__main__":
    main()
