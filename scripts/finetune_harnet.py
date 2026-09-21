#!/usr/bin/env python3
"""Fine-tune Oxford ssl-wearables harnet10 on BRP1 infant posture.

harnet10 is a 1D ResNet-V2 (11.0M params, 10.5M of them in the feature
extractor) self-supervised on ~700,000 person-days of UK Biobank wrist
accelerometry, then released for downstream transfer. Its native input is
exactly our window: 3-channel accelerometer in g, 30 Hz, 10 s -> (B, 3, 300).

Two transfer regimes, because they answer different questions:

  probe  frozen encoder, linear head only -- is the pretrained representation
         already linearly separable for infant trunk posture?
  full   everything trainable at a low LR -- best achievable transfer.

Scored identically to the GBDT baseline: leave-one-subject-out on family_id,
the same 48,135 windows in the same order, accuracy / macro-F1 / Cohen's kappa.
Loss is unweighted cross-entropy to match the baseline's unweighted GBDT -- class
re-weighting is a separate axis and would make the comparison unfair.

Epoch selection uses a 10% random split of the TRAINING windows. That split
shares subjects with training, so it is mildly optimistic for choosing an epoch,
but it never touches the held-out infant.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

PROJ = Path("/work/hdd/bebr/Projects/IMU_pretraining")
OUT_DIR = PROJ / "analysis" / "posture"

MAX_EPOCHS = 40
PATIENCE = 6
BATCH = 256
LR = {"probe": 1e-3, "full": 1e-4}
VAL_FRAC = 0.10
SEED = 0


def load_model(n_classes: int) -> nn.Module:
    """Load from the local torch.hub cache -- compute nodes may lack internet."""
    cached = glob.glob(os.path.join(os.environ.get("TORCH_HOME", ""), "hub",
                                    "*ssl-wearables*"))
    if cached:
        return torch.hub.load(cached[0], "harnet10", class_num=n_classes,
                              pretrained=True, source="local")
    return torch.hub.load("OxWearables/ssl-wearables", "harnet10",
                          class_num=n_classes, pretrained=True, trust_repo=True)


@torch.no_grad()
def evaluate(model, X: torch.Tensor, idx: torch.Tensor, bs: int = 1024) -> np.ndarray:
    """X already lives on the GPU; slice it directly."""
    model.eval()
    out = [model(X[idx[i:i + bs]]).argmax(1) for i in range(0, len(idx), bs)]
    return torch.cat(out).cpu().numpy()


def run_fold(Xg, yg, y_np, tr, te, mode, n_classes, dev, rng) -> tuple[np.ndarray, int]:
    """Xg/yg are already resident on the GPU -- the whole set is only ~173 MB,
    so batching by index beats a DataLoader by a wide margin on this model."""
    model = load_model(n_classes).to(dev)
    if mode == "probe":
        for p in model.feature_extractor.parameters():
            p.requires_grad = False

    # inner split for epoch selection -- training subjects only
    idx = np.where(tr)[0]
    rng.shuffle(idx)
    n_val = int(len(idx) * VAL_FRAC)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    va_t = torch.as_tensor(va_i, device=dev)
    tr_t = torch.as_tensor(tr_i, device=dev)
    te_t = torch.as_tensor(np.where(te)[0], device=dev)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR[mode])
    lossf = nn.CrossEntropyLoss()
    gen = torch.Generator(device=dev).manual_seed(SEED)

    best_f1, best_state, best_ep, bad = -1.0, None, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=dev, generator=gen)]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            lossf(model(Xg[b]), yg[b]).backward()
            opt.step()

        vf1 = f1_score(y_np[va_i], evaluate(model, Xg, va_t), average="macro",
                       zero_division=0)
        if vf1 > best_f1:
            best_f1, best_ep, bad = vf1, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return evaluate(model, Xg, te_t), best_ep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["probe", "full"])
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})"
                              if dev.type == "cuda" else ""))

    X6 = np.load(OUT_DIR / "windows_raw_30hz.npy")
    md = pd.read_parquet(OUT_DIR / "windows_raw_30hz_meta.parquet")
    X = np.ascontiguousarray(X6[:, :3])  # harnet is accelerometer-only
    classes = sorted(md["posture"].unique())
    y = md["posture"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int64)
    gr = md["family_id"].to_numpy()
    subjects = np.unique(gr)
    print(f"X {X.shape} | {len(classes)} classes | {len(subjects)} subjects")

    # Whole dataset resident on the GPU (~173 MB) -- no host/device churn per batch.
    Xg = torch.from_numpy(X).to(dev)
    yg = torch.from_numpy(y).to(dev)
    print(f"resident on device: {Xg.element_size() * Xg.nelement() / 1e6:.0f} MB")

    rows, per_fold = [], []
    for mode in args.modes:
        torch.manual_seed(SEED)
        pred = np.empty_like(y)
        eps = {}
        t0 = time.time()
        for s in subjects:
            te = gr == s
            p, ep = run_fold(Xg, yg, y, ~te, te, mode, len(classes), dev,
                             np.random.default_rng(SEED))
            pred[te] = p
            eps[int(s)] = ep
            k = cohen_kappa_score(y[te], p, labels=np.arange(len(classes)))
            f1 = f1_score(y[te], p, average="macro", zero_division=0)
            print(f"  [{mode}] hold out {s}: n={te.sum():5d} "
                  f"macroF1={f1:.3f} kappa={k:.3f} (best epoch {ep})", flush=True)
            per_fold.append({"model": f"harnet10 ({mode})", "held_out": int(s),
                             "n_test": int(te.sum()), "macro_f1": f1, "kappa": k,
                             "best_epoch": ep})

        labels = np.arange(len(classes))
        rows.append({
            "model": f"harnet10 ({mode})",
            "n_feats": int(np.prod(X.shape[1:])),
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(y, pred, labels=labels),
            "mean_fold_kappa": float(np.mean([r["kappa"] for r in per_fold
                                              if r["model"].endswith(f"({mode})")])),
            "worst_subject_f1": min(r["macro_f1"] for r in per_fold
                                    if r["model"].endswith(f"({mode})")),
            "worst_subject_kappa": min(r["kappa"] for r in per_fold
                                       if r["model"].endswith(f"({mode})")),
        })
        print(f"  [{mode}] POOLED acc={rows[-1]['accuracy']:.3f} "
              f"macroF1={rows[-1]['macro_f1']:.3f} kappa={rows[-1]['kappa']:.3f} "
              f"({time.time() - t0:.0f}s)")
        np.save(OUT_DIR / f"harnet10_{mode}_predictions.npy", pred)
        print("  per-class F1:", json.dumps(
            {c: round(float(v), 3) for c, v in zip(
                classes, f1_score(y, pred, average=None, labels=labels,
                                  zero_division=0))}))

    pd.DataFrame(rows).to_csv(OUT_DIR / "harnet10_loso_results.csv", index=False)
    pd.DataFrame(per_fold).to_csv(OUT_DIR / "harnet10_loso_per_fold.csv", index=False)
    print(f"\nWrote harnet10_loso_results.csv + harnet10_loso_per_fold.csv")


if __name__ == "__main__":
    main()
