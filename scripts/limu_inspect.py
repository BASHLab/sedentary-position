#!/usr/bin/env python3
"""What is in the LIMU-BERT checkpoints, and do their units match ours?

Before fine-tuning the Infant_states_and_RSA IMU encoder on posture we need
three facts it is not safe to guess at:

  1. the exact architecture each checkpoint was trained with (seq_len comes
     straight off the position-embedding table, feature_num off the input
     projection, n_layers off the block count);
  2. the scale of the data it was pretrained on -- LIMU-BERT has no input
     normalisation layer, so feeding it accelerometer in m/s^2 when it was
     pretrained on g would be a silent 9.8x mismatch;
  3. how that compares to `windows_raw_30hz.npy`, the array we would fine-tune
     on.

Read-only. Prints; writes nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

RSA = Path("/work/hdd/bebr/Projects/Infant_states_and_RSA/motion/limu")
OURS = Path("/work/hdd/bebr/Projects/IMU_pretraining/analysis/posture")

CKPTS = {
    "lb_20_120": RSA / "saved/pretrain_base_lb_20_120/model.pt",
    "lb_70_210_old": RSA / "saved/pretrain_base_lb_70_210_old/model.pt",
    "lb_80_240": RSA / "saved/pretrain_base_lb_80_240/model.pt",
    "lb_80_240_v2": RSA / "saved/pretrain_base_lb_80_240_v2/model.pt",
    "lb_80_240_v4": RSA / "saved/pretrain_base_lb_80_240_v4/model.pt",
}

# Pretraining corpora, as (path, channel-axis). These are huge; mmap + sample.
PRETRAIN = {
    "lb_80_240": RSA / "dataset/lb/data_80_240.npy",
    "lb_20_120": RSA / "dataset/lb/data_20_120.npy",
    "lb_70_210_train": RSA / "dataset/lb/train_data_70_210.npy",
}


def describe_ckpt(name: str, path: Path) -> None:
    print("=" * 84)
    print(f"{name}   {path}")
    if not path.exists():
        print("  MISSING")
        return
    sd = torch.load(path, map_location="cpu", weights_only=True)
    # model.py wraps in DataParallel for some checkpoints -> "module." prefix
    prefix = "module." if any(k.startswith("module.") for k in sd) else ""
    if prefix:
        print(f"  (keys carry the '{prefix}' DataParallel prefix)")
    g = lambda k: sd.get(prefix + k)

    pos = g("transformer.embed.pos_embed.weight")
    lin = g("transformer.embed.lin.weight")
    blocks = sorted({int(k.split(".")[2 if not prefix else 3])
                     for k in sd if ".blocks." in k
                     for _ in [0]} | {int(k.split("blocks.")[1].split(".")[0])
                                      for k in sd if "blocks." in k})
    n_layers = (max(blocks) + 1) if blocks else None
    if n_layers is None:  # LIMU-BERT shares one block, looping n_layers times
        n_layers = "shared block (n_layers is a config value, not in the weights)"

    print(f"  seq_len      {tuple(pos.shape) if pos is not None else '?'}"
          f"   -> {pos.shape[0] if pos is not None else '?'} timesteps")
    print(f"  feature_num  {tuple(lin.shape) if lin is not None else '?'}"
          f"   -> {lin.shape[1] if lin is not None else '?'} input channels, "
          f"hidden {lin.shape[0] if lin is not None else '?'}")
    print(f"  n_layers     {n_layers}")
    print(f"  total params {sum(v.numel() for v in sd.values()):,}")
    print("  keys:")
    for k, v in sd.items():
        print(f"    {k:<52} {tuple(v.shape)}")


def describe_array(name: str, path: Path, n: int = 4000) -> None:
    print("=" * 84)
    print(f"{name}   {path}")
    if not path.exists():
        print("  MISSING")
        return
    a = np.load(path, mmap_mode="r")
    print(f"  shape {a.shape}  dtype {a.dtype}  "
          f"({a.size * a.dtype.itemsize / 1e9:.1f} GB on disk)")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(a.shape[0], size=min(n, a.shape[0]), replace=False))
    s = np.asarray(a[idx], dtype=np.float64)          # (n, seq, ch) expected
    ch_axis = -1 if s.shape[-1] <= 9 else 1
    s = np.moveaxis(s, ch_axis, 1)                    # (n, ch, seq)
    print(f"  sampled {len(idx)} windows -> per-channel stats (channel axis {ch_axis}):")
    print(f"    {'ch':>3} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}")
    for c in range(s.shape[1]):
        v = s[:, c]
        print(f"    {c:>3} {v.mean():>10.3f} {v.std():>10.3f} "
              f"{v.min():>10.3f} {v.max():>10.3f}")
    if s.shape[1] >= 3:
        mag = np.linalg.norm(s[:, :3], axis=1)
        print(f"  ||ch 0-2|| per sample: mean {mag.mean():.3f}  std {mag.std():.3f}"
              f"   <- 1.0 means accel in g, 9.81 means m/s^2")
    if s.shape[1] >= 6:
        mag = np.linalg.norm(s[:, 3:6], axis=1)
        print(f"  ||ch 3-5|| per sample: mean {mag.mean():.3f}  std {mag.std():.3f}")


def main() -> None:
    print(f"torch {torch.__version__}   cuda available: {torch.cuda.is_available()}")
    print("\n\n#################### CHECKPOINTS ####################")
    for name, p in CKPTS.items():
        describe_ckpt(name, p)

    print("\n\n#################### PRETRAINING DATA ####################")
    for name, p in PRETRAIN.items():
        describe_array(name, p)

    print("\n\n#################### OUR POSTURE WINDOWS ####################")
    describe_array("windows_raw_30hz (ch 0-2 accel g, 3-5 gyro deg/s)",
                   OURS / "windows_raw_30hz.npy")


if __name__ == "__main__":
    main()
