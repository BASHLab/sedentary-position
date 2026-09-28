#!/usr/bin/env python3
"""Fine-tune the LIMU-BERT IMU encoder from Infant_states_and_RSA on BRP1 posture.

The encoder is the one that project pretrains and reuses:

    /work/hdd/bebr/Projects/Infant_states_and_RSA/motion/limu/
        saved/pretrain_base_lb_80_240/model.pt

a small LIMU-BERT (Transformer, hidden 72, hidden_ff 144, 4 heads, **one block
applied 12 times** -- the architecture shares parameters across depth, so the
checkpoint holds a single block and `n_layers` lives in the config, not the
weights). 71,286 parameters. It was masked-reconstruction pretrained on
1,565,722 three-second windows of LittleBeats IMU at 80 Hz -- about 1,300 hours
of the *same sensor on the same population* as this project's data.

That is the interesting contrast with harnet10. harnet10 brings 700,000
person-days but from adult wrist accelerometry; LIMU-BERT brings four orders of
magnitude less data from exactly the right domain. Both are scored here on the
identical 48,135 windows under the identical leave-one-subject-out split, so the
comparison is like-for-like.

Matching the encoder's input contract
-------------------------------------
* **Rate.** Pretrained at 80 Hz, so we fine-tune on `windows_raw_80hz_9ch.npy`
  (70 -> 80 Hz is an exact x8/7 resample). Feeding it the 30 Hz array built for
  harnet would stretch every motion by 2.7x.
* **Channels -- and this one is a trap.** `Preprocess4Normalization` treats a
  6-channel input as accel+gyro, the LIMU-BERT convention. But the array
  `lb_80_240` was actually pretrained on, `dataset/lb/data_80_240.npy`, is the
  first six *raw LittleBeats columns*, and in that file columns 3-5 are the
  **magnetometer**, not the gyroscope. `scripts/limu_channels.py` establishes
  this from temporal signature rather than scale: those channels barely move
  inside a 3 s window (within-window std 0.015, within/across ratio 0.030),
  which is the magnetometer's signature (ratio 0.068) and is two decades away
  from gyro's (2.705), in any unit. So the encoder's channels 3-5 were
  pretrained on a slow near-constant field, and handing them gyro puts them far
  outside the distribution those weights were fitted to. `--channels` runs it
  either way so the cost of the mismatch is measured rather than assumed.
* **Units.** Pretraining divided accelerometer by 9.8 and left the rest alone,
  so channels 0-2 are effectively g. `build_raw_windows.py` already emits accel
  in g, so no further scaling is applied. (They divide by 9.8, we divided by
  9.80665 -- a 0.07% difference, far below the signal.) Note that ~30% of their
  pretraining windows are all-zero padding, which is why that corpus' mean
  accel magnitude reads 6.9 rather than 9.8.
* **Length.** The position-embedding table is 240 rows, a hard cap: feeding 800
  timesteps would index past the end of it. So each 10 s window is cut into
  three 240-sample (3 s) views at stride 280, which together tile all 800
  samples, encoded separately and pooled. The window, its label and its fold are
  untouched -- only the encoder sees a subdivided view.

Two transfer regimes, as for harnet10:
    probe  frozen encoder, linear head only
    full   everything trainable at a low LR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

PROJ = Path("/work/hdd/bebr/Projects/IMU_pretraining")
OUT_DIR = PROJ / "analysis" / "posture"
RSA = Path("/work/hdd/bebr/Projects/Infant_states_and_RSA/motion")
CKPT = RSA / "limu/saved/pretrain_base_lb_80_240/model.pt"

# base_v3 from limu/config/limu_bert.json -- the config lb_80_240 was trained with.
LIMU_CFG = dict(hidden=72, hidden_ff=144, feature_num=6, n_layers=12, n_heads=4,
                seq_len=240, emb_norm=True)

SUB_LEN = 240      # position-embedding table size; a hard cap
N_SUB = 3          # 3 x 240 at stride 280 tiles all 800 samples of a 10 s window

# Which raw channels (of windows_raw_80hz_9ch.npy: 0-2 accel g, 3-5 mag,
# 6-8 gyro deg/s) to feed the encoder's 6 input dims.
CHANNELS = {
    # faithful to pretraining: exactly the columns lb_80_240 actually saw
    "acc_mag": [0, 1, 2, 3, 4, 5],
    # consistent with every other model in this project; magnetometer excluded
    # as a room/subject-identity feature (docs/METHODS.md)
    "acc_gyr": [0, 1, 2, 6, 7, 8],
    # only the channels that genuinely match; 3-5 zeroed, which is itself
    # in-distribution since ~30% of the pretraining windows are all-zero
    "acc_only": [0, 1, 2, -1, -1, -1],
}

MAX_EPOCHS = 40
PATIENCE = 6
BATCH = 256
LR = {"probe": 1e-3, "full": 1e-4}
VAL_FRAC = 0.10
SEED = 0


# ------------------------------------------------------------------ the model
def limu_transformer() -> nn.Module:
    """Their Transformer class, imported rather than re-implemented.

    LIMU-BERT shares one block across depth and applies its own LayerNorm
    variant; re-typing that here would be an invitation to a silent mismatch
    with the checkpoint. Importing keeps one source of truth.
    """
    sys.path.insert(0, str(RSA))
    from limu.config import PretrainModelConfig
    from limu.models import Transformer

    cfg = PretrainModelConfig(**LIMU_CFG)
    return Transformer(cfg)


def load_pretrained(enc: nn.Module) -> tuple[int, int]:
    """Copy transformer.* weights out of the pretrain checkpoint into `enc`."""
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in sd.items()}
    enc_sd = {k[len("transformer."):]: v for k, v in sd.items()
              if k.startswith("transformer.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} "
                           f"unexpected={unexpected}")
    return len(enc_sd), sum(v.numel() for v in enc_sd.values())


class LimuPosture(nn.Module):
    """Encode each 3 s view, mean-pool over time, mean-pool over views, classify."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.encoder = limu_transformer()
        self.classifier = nn.Linear(LIMU_CFG["hidden"], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_SUB, SUB_LEN, 6) -> fold views into the batch for one pass
        b, s, t, c = x.shape
        h = self.encoder(x.reshape(b * s, t, c))   # (B*S, T, hidden)
        h = h.mean(dim=1).reshape(b, s, -1).mean(dim=1)
        return self.classifier(h)


# --------------------------------------------------------------------- pieces
def to_views(X: np.ndarray) -> np.ndarray:
    """(N, 6, 800) -> (N, N_SUB, 240, 6), views at stride 280 covering all 800."""
    n, c, t = X.shape
    stride = (t - SUB_LEN) // (N_SUB - 1)
    starts = [i * stride for i in range(N_SUB)]
    assert starts[-1] + SUB_LEN == t, (starts, t)
    v = np.stack([X[:, :, s:s + SUB_LEN] for s in starts], axis=1)  # (N,S,6,240)
    return np.ascontiguousarray(v.transpose(0, 1, 3, 2))            # (N,S,240,6)


@torch.no_grad()
def evaluate(model, X: torch.Tensor, idx: torch.Tensor, bs: int = 512) -> np.ndarray:
    model.eval()
    out = [model(X[idx[i:i + bs]]).argmax(1) for i in range(0, len(idx), bs)]
    return torch.cat(out).cpu().numpy()


def run_fold(Xg, yg, y_np, tr, te, mode, n_classes, dev, rng) -> tuple[np.ndarray, int]:
    model = LimuPosture(n_classes).to(dev)
    load_pretrained(model.encoder)
    if mode == "probe":
        for p in model.encoder.parameters():
            p.requires_grad = False

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


# ----------------------------------------------------------------------- main
def smoke() -> None:
    """Build the model, load the checkpoint strictly, push a batch through it.

    Everything that can silently go wrong -- a key-name mismatch, a seq_len the
    position table cannot index, a channel-order slip -- fails here in seconds
    rather than an hour into a GPU job.
    """
    print(f"checkpoint: {CKPT}  exists={CKPT.exists()}")
    m = LimuPosture(7)
    n_k, n_p = load_pretrained(m.encoder)
    print(f"loaded {n_k} tensors, {n_p:,} params, strict key match OK")
    print(f"encoder params {sum(p.numel() for p in m.encoder.parameters()):,} "
          f"| head {sum(p.numel() for p in m.classifier.parameters()):,}")

    # the view split must tile a 10 s window at 80 Hz exactly
    fake = np.random.randn(5, 6, 800).astype(np.float32)
    v = to_views(fake)
    print(f"to_views: {fake.shape} -> {v.shape}")
    out = m(torch.from_numpy(v))
    print(f"forward:  {tuple(v.shape)} -> {tuple(out.shape)}  (expect (5, 7))")
    assert out.shape == (5, 7), out.shape

    # a frozen encoder must leave only the head trainable
    for p in m.encoder.parameters():
        p.requires_grad = False
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"probe mode: {n_train:,} trainable params (expect head only)")
    print("\nSMOKE TEST PASSED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["probe", "full"])
    ap.add_argument("--channels", default="acc_gyr", choices=sorted(CHANNELS),
                    help="acc_gyr: as every other model in this project. "
                         "acc_mag: the channels lb_80_240 was really pretrained "
                         "on. acc_only: accel plus zeros.")
    ap.add_argument("--smoke", action="store_true",
                    help="check model wiring and exit; no data, no training")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})"
                              if dev.type == "cuda" else ""))
    print(f"checkpoint: {CKPT}")

    X9 = np.load(OUT_DIR / "windows_raw_80hz_9ch.npy")
    md = pd.read_parquet(OUT_DIR / "windows_raw_80hz_9ch_meta.parquet")
    classes = sorted(md["posture"].unique())
    y = md["posture"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int64)
    gr = md["family_id"].to_numpy()
    subjects = np.unique(gr)
    print(f"X {X9.shape} | {len(classes)} classes | {len(subjects)} subjects")

    sel = CHANNELS[args.channels]
    X6 = np.stack([np.zeros_like(X9[:, 0]) if c < 0 else X9[:, c] for c in sel],
                  axis=1)
    print(f"--channels {args.channels} -> raw columns {sel} "
          f"(-1 = zero-filled)")

    # Units must already match what the encoder was pretrained on.
    print(f"  ch 0-2 |v| mean {np.linalg.norm(X6[:, :3], axis=1).mean():.3f} "
          f"(expect ~1.0 g; pretraining divided accel by 9.8)")
    print(f"  ch 3-5 |v| mean {np.linalg.norm(X6[:, 3:6], axis=1).mean():.3f} "
          f"(pretraining saw ~0.53 here -- magnetometer)")

    X = to_views(X6)
    print(f"views {X.shape}  ({N_SUB} x {SUB_LEN} samples = "
          f"{N_SUB * SUB_LEN / 80:.1f} s of the 10 s window)")

    Xg = torch.from_numpy(X).to(dev)
    yg = torch.from_numpy(y).to(dev)
    print(f"resident on device: {Xg.element_size() * Xg.nelement() / 1e6:.0f} MB")

    probe = LimuPosture(len(classes))
    n_k, n_p = load_pretrained(probe.encoder)
    print(f"loaded {n_k} pretrained tensors ({n_p:,} params) into the encoder; "
          f"head is {sum(p.numel() for p in probe.classifier.parameters()):,}")

    rows, per_fold = [], []
    for mode in args.modes:
        torch.manual_seed(SEED)
        pred = np.empty_like(y)
        t0 = time.time()
        for s in subjects:
            te = gr == s
            p, ep = run_fold(Xg, yg, y, ~te, te, mode, len(classes), dev,
                             np.random.default_rng(SEED))
            pred[te] = p
            k = cohen_kappa_score(y[te], p, labels=np.arange(len(classes)))
            f1 = f1_score(y[te], p, average="macro", zero_division=0)
            print(f"  [{mode}] hold out {s}: n={te.sum():5d} "
                  f"macroF1={f1:.3f} kappa={k:.3f} (best epoch {ep})", flush=True)
            per_fold.append({"model": f"limu-bert ({mode}, {args.channels})", "held_out": int(s),
                             "n_test": int(te.sum()), "macro_f1": f1, "kappa": k,
                             "best_epoch": ep})

        labels = np.arange(len(classes))
        mine = [r for r in per_fold
                if r["model"].endswith(f"({mode}, {args.channels})")]
        rows.append({
            "model": f"limu-bert ({mode}, {args.channels})",
            "n_feats": int(np.prod(X.shape[1:])),
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(y, pred, labels=labels),
            "mean_fold_kappa": float(np.mean([r["kappa"] for r in mine])),
            "worst_subject_f1": min(r["macro_f1"] for r in mine),
            "worst_subject_kappa": min(r["kappa"] for r in mine),
        })
        print(f"  [{mode}] POOLED acc={rows[-1]['accuracy']:.3f} "
              f"macroF1={rows[-1]['macro_f1']:.3f} kappa={rows[-1]['kappa']:.3f} "
              f"({time.time() - t0:.0f}s)")
        np.save(OUT_DIR / f"limu_{args.channels}_{mode}_predictions.npy", pred)
        # Flush after each mode: a 4 h wall-clock kill during `full` previously
        # discarded a completed `probe` that had taken 2.5 h.
        pd.DataFrame(rows).to_csv(
            OUT_DIR / f"limu_{args.channels}_loso_results.csv", index=False)
        pd.DataFrame(per_fold).to_csv(
            OUT_DIR / f"limu_{args.channels}_loso_per_fold.csv", index=False)
        print("  per-class F1:", json.dumps(
            {c: round(float(v), 3) for c, v in zip(
                classes, f1_score(y, pred, average=None, labels=labels,
                                  zero_division=0))}))

    pd.DataFrame(rows).to_csv(
        OUT_DIR / f"limu_{args.channels}_loso_results.csv", index=False)
    pd.DataFrame(per_fold).to_csv(
        OUT_DIR / f"limu_{args.channels}_loso_per_fold.csv", index=False)
    print(f"\nWrote limu_{args.channels}_loso_results.csv + "
          f"limu_{args.channels}_loso_per_fold.csv")


if __name__ == "__main__":
    main()
