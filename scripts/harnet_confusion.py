#!/usr/bin/env python3
"""Per-fold confusion matrices for harnet10, and why macro-F1 sits below kappa.

`finetune_harnet.py` already saved the pooled out-of-fold predictions
(`harnet10_{probe,full}_predictions.npy`), one row per window in the order of
`windows_raw_30hz_meta.parquet`. Every number here is derived from those, so
nothing is re-trained and the GPU is not needed.

Two things come out:

1. **Confusion, per held-out infant.** The pooled matrix in
   `baseline_confusion.csv` averages over six very different label
   distributions -- 25004 is 29% `stand` and 0.3% `on back`, everyone else is
   back/stomach-dominant -- so the per-fold matrices are the ones that show
   where transfer actually breaks.

2. **The macro-F1 / kappa gap.** Two separate causes, and it is worth keeping
   them apart:

   *Pooled*, they disagree because they weight classes differently --

       macro-F1 = unweighted mean of per-class F1 -> `crawl` (1.1% of windows)
                  counts as much as `on back` (26.2%)
       kappa    = (p_o - p_e) / (1 - p_e), i.e. chance-corrected accuracy, which
                  the three big lying classes (72% of windows) dominate

   *Per fold*, the gap is also inflated -- sometimes reversed -- by a denominator
   mismatch. `f1_score(..., average="macro")` with no `labels=` averages over the
   classes present in `y_true | y_pred`, so a fold whose infant never crawls
   still gets a 0 for `crawl` as soon as the model predicts it once, dividing by
   7 instead of 6. Kappa is computed with `labels=` pinned and is unaffected.
   Three macro-F1 variants are reported side by side so the artefact is visible:
   `macro_f1_union` (what the results sheets currently report), `macro_f1_pinned`
   (all 7, absent classes scored 0) and `macro_f1_present` (only the classes the
   held-out infant was actually coded for -- the comparable one).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

PROJ = Path("/work/hdd/bebr/Projects/IMU_pretraining")
OUT_DIR = PROJ / "analysis" / "posture"
MODES = ("probe", "full")

# ---------------------------------------------------------------- design tokens
# Same tokens as posture_distribution.py -- one house style across the figures.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95"]
# Sequential: magnitude gets one hue, light -> dark, anchored on the surface.
SEQ = LinearSegmentedColormap.from_list("blue_seq", [SURFACE, *BLUE_RAMP])
# Diverging: two poles + a neutral midpoint, for a signed quantity.
POS, NEG, MID = "#2a78d6", "#eb6834", "#c9c8c4"


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------- the numbers
def chance_agreement(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """p_e -- the agreement two raters with these marginals get by luck."""
    ty = np.bincount(y, minlength=k) / len(y)
    tp = np.bincount(p, minlength=k) / len(p)
    return float((ty * tp).sum())


def fold_metrics(y: np.ndarray, p: np.ndarray, labels: np.ndarray) -> dict:
    pe = chance_agreement(y, p, len(labels))
    prev = np.bincount(y, minlength=len(labels)) / len(y)
    present = labels[prev > 0]                      # classes this infant was coded for
    union = np.union1d(np.unique(y), np.unique(p))  # sklearn's default divisor
    return {
        "n": len(y),
        "n_classes_present": int(len(present)),
        "n_classes_union": int(len(union)),
        "accuracy": accuracy_score(y, p),
        # the divisor is the whole story -- three ways to take the same mean
        "macro_f1_union": f1_score(y, p, average="macro", zero_division=0),
        "macro_f1_pinned": f1_score(y, p, average="macro", labels=labels,
                                    zero_division=0),
        "macro_f1_present": f1_score(y, p, average="macro", labels=present,
                                     zero_division=0),
        "weighted_f1": f1_score(y, p, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y, p, labels=labels),
        "p_e": pe,
        # inverse Simpson: how many classes the labels are *effectively* spread over
        "eff_n_classes": float(1.0 / (prev ** 2).sum()),
        "max_class_share": float(prev.max()),
    }


# ----------------------------------------------------------------------- figures
def fig_confusion_by_fold(cm_pct: dict[int, np.ndarray], support: dict[int, np.ndarray],
                          classes: list[str], mode: str, metrics: pd.DataFrame) -> None:
    """Six small multiples, row-normalised: each row reads 'of the windows the
    coder called X, where did the model put them?'"""
    folds = sorted(cm_pct)
    ncol = 3
    nrow = int(np.ceil(len(folds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.55 * ncol, 4.35 * nrow),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()

    for ax, fid in zip(axes, folds):
        M = cm_pct[fid]
        ax.imshow(M, cmap=SEQ, vmin=0, vmax=100, aspect="equal")
        m = metrics.loc[metrics["held_out"] == fid].iloc[0]
        ax.set_title(f"held out {fid}",
                     color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=20)
        ax.text(0, 1.045, f"macro-F1 {m.macro_f1_union:.2f}   κ {m.kappa:.2f}   "
                          f"acc {m.accuracy:.2f}   n={int(m.n):,}",
                transform=ax.transAxes, color=INK_2, fontsize=8.5, va="bottom")

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticklabels([f"{c}  ({int(s)})" if s else f"{c}  (—)"
                            for c, s in zip(classes, support[fid])], fontsize=8.5)
        for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            lbl.set_color(INK_2)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        # 2px surface gap between cells
        ax.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2.0)
        ax.tick_params(which="minor", length=0)

        for i in range(len(classes)):
            if support[fid][i] == 0:
                ax.text(len(classes) / 2 - 0.5, i, "class absent in this infant",
                        ha="center", va="center", fontsize=7.5, color=INK_MUTED,
                        style="italic")
                continue
            for j in range(len(classes)):
                v = M[i, j]
                if v < 0.5:
                    continue
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if v >= 55 else INK_2,
                        fontweight="bold" if i == j else "normal")

        ax.set_xlabel("predicted", color=INK_2, fontsize=9.5, labelpad=6)
        ax.set_ylabel("coded (n)", color=INK_2, fontsize=9.5, labelpad=6)

    for ax in axes[len(folds):]:
        ax.axis("off")

    fig.suptitle(f"harnet10 ({mode}) — confusion per held-out infant, "
                 f"row-normalised (% of that infant's coded windows)",
                 color=INK, fontsize=13.5, fontweight="bold", x=0.012, ha="left",
                 y=0.995)
    fig.text(0.012, 0.005,
             "Leave-one-subject-out; each window predicted once while its infant "
             "was held out. Rows sum to 100%; blank cells are <0.5%.",
             color=INK_MUTED, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0.018, 1, 0.972))
    fig.savefig(OUT_DIR / f"harnet10_confusion_{mode}_by_fold.png", dpi=200,
                facecolor=SURFACE)
    plt.close(fig)


def fig_f1_vs_kappa(per_class: pd.DataFrame, pooled: dict, folds: pd.DataFrame,
                    mode: str) -> None:
    """Left: where macro-F1 loses its points. Right: the per-fold divisor effect."""
    fig = plt.figure(figsize=(15.0, 6.9), facecolor=SURFACE)
    # Explicit margins, not tight_layout -- the titles and the legend rows need
    # reserved space and tight_layout cannot see annotation text.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.06],
                          left=0.098, right=0.988, top=0.735, bottom=0.175,
                          wspace=0.30)
    axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # --- left: per-class F1, ordered by support ------------------------------
    d = per_class.sort_values("support")
    tot = d["support"].sum()
    colors = [BLUE_RAMP[round(i * (len(BLUE_RAMP) - 1) / max(len(d) - 1, 1))]
              for i in range(len(d))]
    axL.barh(d["posture"], d["f1"], color=colors, height=0.6)
    style_axes(axL)
    axL.xaxis.grid(True, color=GRID, linewidth=1.0)
    axL.set_xlim(0, 1.62)                      # headroom for the direct labels
    axL.set_xticks(np.arange(0, 1.01, 0.2))
    axL.set_xlabel("per-class F1 (pooled out-of-fold)", color=INK_2, fontsize=10,
                   labelpad=8)
    for lbl in axL.get_yticklabels():
        lbl.set_color(INK)
        lbl.set_fontsize(10.5)
    # Labels in a fixed column past every bar, so they never cross the reference
    # lines at 0.75-0.86 the way a label hugging a mid-length bar would.
    for y, (v, s) in enumerate(zip(d["f1"], d["support"])):
        axL.text(1.03, y, f"{v:.2f}", va="center", ha="left", fontsize=9.5,
                 color=INK, fontweight="bold")
        axL.text(1.16, y, f"n={int(s):,}  ({s / tot * 100:.1f}%)", va="center",
                 ha="left", fontsize=9.5, color=INK_2)

    refs = ((pooled["macro_f1_union"], f"macro-F1  {pooled['macro_f1_union']:.3f}", "-"),
            (pooled["kappa"], f"κ  {pooled['kappa']:.3f}", ":"),
            (pooled["weighted_f1"], f"weighted-F1  {pooled['weighted_f1']:.3f}", "--"))
    for x, _, style in refs:
        axL.axvline(x, color=INK, linewidth=1.4, linestyle=style, alpha=0.8)
    axL.legend([plt.Line2D([0], [0], color=INK, lw=1.4, ls=s) for _, _, s in refs],
               [lab for _, lab, _ in refs],
               loc="upper left", bbox_to_anchor=(0.0, -0.135), ncol=3, frameon=False,
               fontsize=9.5, labelcolor=INK_2, handlelength=2.6, columnspacing=2.4,
               borderpad=0.0, handletextpad=0.7)
    axL.set_title("macro-F1 averages these seven bars equally;\n"
                  "κ and weighted-F1 weight them by support",
                  color=INK, fontsize=12, fontweight="bold", loc="left", pad=14)

    # --- right: per fold, macro-F1 (three divisors) against κ ----------------
    f = folds.sort_values("kappa").reset_index(drop=True)
    ypos = np.arange(len(f))
    style_axes(axR)
    axR.xaxis.grid(True, color=GRID, linewidth=1.0)
    for y, r in f.iterrows():
        lo = min(r.macro_f1_union, r.macro_f1_present, r.kappa)
        hi = max(r.macro_f1_union, r.macro_f1_present, r.kappa)
        axR.plot([lo, hi], [y, y], color=GRID, linewidth=2.4, zorder=1,
                 solid_capstyle="round")
    axR.scatter(f["macro_f1_union"], ypos, s=95, marker="o", color="#5598e7",
                edgecolor=SURFACE, linewidth=2.0, zorder=3,
                label="macro-F1, divisor = classes in y ∪ ŷ  (as reported)")
    axR.scatter(f["macro_f1_present"], ypos, s=95, marker="o", facecolor=SURFACE,
                edgecolor="#5598e7", linewidth=2.0, zorder=3,
                label="macro-F1, divisor = classes the infant was coded for")
    axR.scatter(f["kappa"], ypos, s=105, marker="D", color="#184f95",
                edgecolor=SURFACE, linewidth=2.0, zorder=3, label="Cohen's κ")

    axR.set_yticks(ypos)
    axR.set_yticklabels([f"{int(s)}" + ("" if n == 7 else f"   ({n} of 7 classes)")
                         for s, n in zip(f["held_out"], f["n_classes_present"])])
    for lbl in axR.get_yticklabels():
        lbl.set_color(INK)
        lbl.set_fontsize(10.5)
    axR.set_ylim(-0.7, len(f) - 0.3)
    axR.set_xlim(0.30, 1.0)
    axR.set_xlabel("score on the held-out infant", color=INK_2, fontsize=10, labelpad=8)
    axR.set_ylabel("held-out infant", color=INK_2, fontsize=10, labelpad=8)
    axR.legend(loc="upper left", bbox_to_anchor=(0.0, -0.135), ncol=1, frameon=False,
               fontsize=9.5, labelcolor=INK_2, borderpad=0.0, handletextpad=0.6,
               labelspacing=0.5)
    axR.set_title("per fold the gap is unstable and can invert —\n"
                  "part of it is the macro-F1 divisor, not the model",
                  color=INK, fontsize=12, fontweight="bold", loc="left", pad=14)

    fig.text(0.008, 0.955,
             f"harnet10 ({mode}) — why macro-F1 ({pooled['macro_f1_union']:.3f}) "
             f"sits below κ ({pooled['kappa']:.3f})",
             color=INK, fontsize=14, fontweight="bold", ha="left", va="top")
    fig.text(0.008, 0.906,
             f"κ = (p_o − pₑ)/(1 − pₑ) with p_o = accuracy {pooled['accuracy']:.3f} and "
             f"pₑ = {pooled['p_e']:.3f}, so κ is essentially chance-corrected accuracy — "
             f"and accuracy is carried by the three large lying\nclasses (72.4% of "
             f"windows). macro-F1 spends a seventh of its budget on each of crawl (1.1% "
             f"of windows) and stand (8.2%), which are the two the model cannot do.",
             color=INK_2, fontsize=10, ha="left", va="top", linespacing=1.5)
    fig.savefig(OUT_DIR / f"harnet10_f1_vs_kappa_{mode}.png", dpi=200,
                facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main() -> None:
    md = pd.read_parquet(OUT_DIR / "windows_raw_30hz_meta.parquet")
    classes = sorted(md["posture"].unique())
    labels = np.arange(len(classes))
    y = md["posture"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int64)
    gr = md["family_id"].to_numpy()
    subjects = np.unique(gr)
    print(f"{len(md):,} windows | {len(classes)} classes | {len(subjects)} subjects")
    print("classes:", ", ".join(f"{c} ({(y == i).sum():,})"
                                for i, c in enumerate(classes)))

    cm_rows, pc_rows, fold_rows, pooled_rows = [], [], [], []

    for mode in MODES:
        pred_path = OUT_DIR / f"harnet10_{mode}_predictions.npy"
        if not pred_path.exists():
            print(f"  ! {pred_path.name} missing, skipping {mode}")
            continue
        pred = np.load(pred_path)
        assert len(pred) == len(y), (len(pred), len(y))

        cm_pct, support = {}, {}
        for s in subjects:
            te = gr == s
            ys, ps = y[te], pred[te]
            C = confusion_matrix(ys, ps, labels=labels)
            sup = C.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                P = np.where(sup[:, None] > 0, C / np.maximum(sup[:, None], 1) * 100, 0.0)
            cm_pct[int(s)], support[int(s)] = P, sup

            for i, ci in enumerate(classes):
                for j, cj in enumerate(classes):
                    cm_rows.append({"model": f"harnet10 ({mode})", "held_out": int(s),
                                    "coded": ci, "predicted": cj,
                                    "n": int(C[i, j]), "row_pct": float(P[i, j])})

            pr, rc, f1c, _ = precision_recall_fscore_support(
                ys, ps, labels=labels, zero_division=0)
            for i, c in enumerate(classes):
                pc_rows.append({"model": f"harnet10 ({mode})", "held_out": int(s),
                                "posture": c, "support": int(sup[i]),
                                "precision": pr[i], "recall": rc[i], "f1": f1c[i]})

            m = fold_metrics(ys, ps, labels)
            m.update(model=f"harnet10 ({mode})", held_out=int(s))
            m["gap"] = m["kappa"] - m["macro_f1_union"]
            m["divisor_penalty"] = m["macro_f1_present"] - m["macro_f1_union"]
            fold_rows.append(m)

        # pooled
        pm = fold_metrics(y, pred, labels)
        pm.update(model=f"harnet10 ({mode})", held_out="pooled")
        pm["gap"] = pm["kappa"] - pm["macro_f1_union"]
        pm["divisor_penalty"] = pm["macro_f1_present"] - pm["macro_f1_union"]
        pooled_rows.append(pm)

        pr, rc, f1c, sup = precision_recall_fscore_support(
            y, pred, labels=labels, zero_division=0)
        per_class = pd.DataFrame({"posture": classes, "support": sup,
                                  "precision": pr, "recall": rc, "f1": f1c})
        tot_sup = int(sup.sum())
        for i, c in enumerate(classes):
            pc_rows.append({"model": f"harnet10 ({mode})", "held_out": "pooled",
                            "posture": c, "support": int(sup[i]),
                            "precision": pr[i], "recall": rc[i], "f1": f1c[i]})

        fdf = pd.DataFrame([r for r in fold_rows if r["model"].endswith(f"({mode})")])
        fig_confusion_by_fold(cm_pct, support, classes, mode, fdf)
        fig_f1_vs_kappa(per_class, pm, fdf, mode)

        # ------------------------------------------------------------ report
        print("\n" + "=" * 100)
        print(f"harnet10 ({mode})  —  pooled  acc {pm['accuracy']:.3f} | "
              f"macro-F1 {pm['macro_f1_union']:.3f} | "
              f"weighted-F1 {pm['weighted_f1']:.3f} | "
              f"κ {pm['kappa']:.3f} | pₑ {pm['p_e']:.3f}")
        print("=" * 100)
        print(per_class.sort_values("support")
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\n  [1] mean of the 7 F1 values         = {f1c.mean():.3f}   (macro-F1)")
        print(f"  [2] support-weighted mean of those  = {pm['weighted_f1']:.3f}   (weighted-F1)")
        print(f"  [3] accuracy (p_o)                  = {pm['accuracy']:.3f}")
        print(f"  [4] chance agreement  p_e           = {pm['p_e']:.3f}")
        print(f"  [5] κ = (p_o − p_e)/(1 − p_e)       = "
              f"({pm['accuracy']:.3f} − {pm['p_e']:.3f})/(1 − {pm['p_e']:.3f}) "
              f"= {pm['kappa']:.3f}")
        big = per_class.nlargest(3, "support")
        print(f"\n  the three largest classes ({', '.join(big['posture'])}) are "
              f"{big['support'].sum() / tot_sup * 100:.1f}% of windows at mean F1 "
              f"{big['f1'].mean():.3f};")
        small = per_class.nsmallest(3, "support")
        print(f"  the three smallest ({', '.join(small['posture'])}) are "
              f"{small['support'].sum() / tot_sup * 100:.1f}% at mean F1 "
              f"{small['f1'].mean():.3f}. [3]/[5] see the first group, [1] splits "
              f"its budget evenly.")
        print("\n  per fold — three macro-F1 divisors against κ:")
        print(fdf[["held_out", "n", "n_classes_present", "n_classes_union",
                   "accuracy", "macro_f1_union", "macro_f1_pinned",
                   "macro_f1_present", "kappa", "gap", "divisor_penalty", "p_e"]]
              .sort_values("kappa")
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        art = fdf[fdf["n_classes_present"] < len(classes)]
        if len(art):
            print(f"\n  {len(art)} of {len(fdf)} folds hold out an infant that was never "
                  f"coded for all {len(classes)} classes. For those, the reported "
                  f"macro-F1 divides by\n  the classes the model *predicted* rather "
                  f"than the classes that exist, costing "
                  f"{art['divisor_penalty'].mean():.3f} macro-F1 on average "
                  f"(max {art['divisor_penalty'].max():.3f}).")

    # ------------------------------------------------------------------- write
    pd.DataFrame(cm_rows).to_csv(OUT_DIR / "harnet10_confusion_by_fold.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(OUT_DIR / "harnet10_per_class_by_fold.csv", index=False)
    pd.concat([pd.DataFrame(fold_rows), pd.DataFrame(pooled_rows)], ignore_index=True) \
        .to_csv(OUT_DIR / "harnet10_f1_vs_kappa.csv", index=False)
    print(f"\nWrote harnet10_confusion_by_fold.csv, harnet10_per_class_by_fold.csv, "
          f"harnet10_f1_vs_kappa.csv and 4 figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
