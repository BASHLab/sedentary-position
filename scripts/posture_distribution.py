#!/usr/bin/env python3
"""Distribution of the `posture` tier for the BRP1 EMA annotations.

Annotations live in Behavioral_Coding/BRP1_EMAannotations as ELAN tab-delimited
exports (one row per annotation, 9 columns):

    tier | <blank> | begin hh:mm:ss.mmm | begin_s | end hh:mm:ss.mmm | end_s |
    dur hh:mm:ss.mmm | dur_s | label

The annotation clock starts at the GoPro tone, not at the start of the LB
recording. BRP1_EMAfiles_ToneTimes_tracking.xlsx carries that offset
(LBoffset_tonetime_secs) per (family_id, EMA_obs_number), so a posture segment
is only mappable onto the IMU stream when the offset is known.

Outputs CSV tables under analysis/posture/ and figures under figures/.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

ANNOT_DIR = Path("/work/hdd/bebr/Data/Behavioral_Coding/BRP1_EMAannotations")
LB_DIR = Path("/work/hdd/bebr/Data/LB/cleaned/BRP1")
TRACKING_XLSX = Path(
    "/work/hdd/bebr/Projects/IMU_pretraining/BRP1_EMAfiles_ToneTimes_tracking.xlsx"
)
OUT_DIR = Path("/work/hdd/bebr/Projects/IMU_pretraining/analysis/posture")
# Tables live with the analysis; every figure in the project goes to figures/.
FIG_DIR = Path("/work/hdd/bebr/Projects/IMU_pretraining/figures")

COLUMNS = [
    "tier",
    "blank",
    "begin_hms",
    "begin_s",
    "end_hms",
    "end_s",
    "dur_hms",
    "dur_s",
    "label",
]

# `position` is a one-off tier name in BRP1_25008_Obs1A; same label vocabulary.
POSTURE_TIERS = {"posture", "position"}

FNAME_RE = re.compile(r"^BRP1_(?P<fid>\d+)_Obs(?P<obs>\d+[A-Za-z]?)_EDITED\.txt$")

# ---------------------------------------------------------------- design tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

# Blue sequential ramp, steps 250-600 (light-mode ordinal floor is step 250).
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95"]
# Categorical theme, fixed slot order (light mode).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#4a3aa7", "#e34948"]


def hours(seconds: float) -> float:
    return seconds / 3600.0


def hms(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# ------------------------------------------------------------------- load data
def load_annotations() -> pd.DataFrame:
    """Every annotation row across every EDITED file, with file provenance."""
    frames = []
    for path in sorted(ANNOT_DIR.glob("*_EDITED.txt")):
        m = FNAME_RE.match(path.name)
        if not m:
            print(f"  ! unexpected filename, skipped: {path.name}")
            continue
        df = pd.read_csv(
            path, sep="\t", header=None, names=COLUMNS, dtype={"label": "string"},
            keep_default_na=False, na_values=[],
        )
        df["family_id"] = int(m.group("fid"))
        df["obs"] = m.group("obs").upper()
        df["source_file"] = path.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["tier_norm"] = out["tier"].str.strip().str.lower()
    out["label_norm"] = out["label"].str.strip().str.lower()
    for c in ("begin_s", "end_s", "dur_s"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_tracking() -> pd.DataFrame:
    trk = pd.read_excel(TRACKING_XLSX)
    trk["family_id"] = trk["family_id"].astype(int)
    trk["obs"] = trk["EMA_obs_number"].astype(str).str.strip().str.upper()
    trk["tone_offset_s"] = pd.to_numeric(trk["LBoffset_tonetime_secs"], errors="coerce")
    trk["lb_session"] = (
        trk["LB_audiofile"].astype(str).str.replace("_audio_sync$", "", regex=True)
    )
    return trk[["family_id", "obs", "lb_session", "tone_offset_s", "Notes"]]


def scan_lb_recordings() -> pd.DataFrame:
    """Every cleaned LB session on disk, and whether it carries an IMU file."""
    rows = []
    for fid_dir in sorted(p for p in LB_DIR.iterdir() if p.is_dir()):
        for ctx_dir in sorted(p for p in fid_dir.iterdir() if p.is_dir()):
            for sess in sorted(p for p in ctx_dir.iterdir() if p.is_dir()):
                imu = sess / f"{sess.name}_imu_sync.txt"
                rows.append(
                    {
                        "family_id": int(fid_dir.name),
                        "context": ctx_dir.name,
                        "lb_session": sess.name,
                        "has_imu": imu.exists(),
                        "imu_bytes": imu.stat().st_size if imu.exists() else 0,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- figures
def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_axisbelow(True)


def fig_by_class(by_class: pd.DataFrame, total_h: float) -> None:
    """Magnitude, low->high: horizontal bar, one hue, direct-labeled."""
    d = by_class.sort_values("hours")
    n = len(d)
    # Darkest step to the largest bar; stay inside the validated ordinal range.
    colors = [BLUE_RAMP[round(i * (len(BLUE_RAMP) - 1) / max(n - 1, 1))] for i in range(n)]

    fig, ax = plt.subplots(figsize=(9.0, 0.62 * n + 2.1), facecolor=SURFACE)
    ax.barh(d["posture"], d["hours"], color=colors, height=0.62)
    style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_xlim(0, d["hours"].max() * 1.22)
    ax.set_xlabel("Annotated duration (hours)", color=INK_2, fontsize=10, labelpad=8)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=11)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)

    for y, (h, pct, cnt) in enumerate(zip(d["hours"], d["pct_time"], d["n_segments"])):
        ax.text(h + d["hours"].max() * 0.018, y, f"{h:.2f} h   {pct:.1f}%   n={cnt}",
                va="center", ha="left", fontsize=9.5, color=INK_2)

    ax.set_title("Posture tier — annotated time per class", color=INK, fontsize=13,
                 fontweight="bold", loc="left", pad=14)
    fig.text(0.008, 0.008,
             f"BRP1 EMA annotations · {total_h:.1f} h of coded posture across "
             f"{int(by_class['n_segments'].sum())} segments",
             color=INK_MUTED, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG_DIR / "posture_by_class.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_by_subject(matrix: pd.DataFrame, order: list[str]) -> None:
    """Part-to-whole per subject: 100% stacked horizontal bar, categorical."""
    share = matrix.div(matrix.sum(axis=1), axis=0) * 100.0
    share = share.loc[sorted(share.index, reverse=True)]
    colors = dict(zip(order, CATEGORICAL))

    fig, ax = plt.subplots(figsize=(11.0, 0.66 * len(share) + 2.9), facecolor=SURFACE)
    left = pd.Series(0.0, index=share.index)
    ylab = [str(i) for i in share.index]
    for posture in order:
        vals = share[posture]
        ax.barh(ylab, vals, left=left, color=colors[posture], height=0.62,
                edgecolor=SURFACE, linewidth=2.0)  # 2px surface gap between fills
        for y, (v, l0) in enumerate(zip(vals, left)):
            if v >= 7.0:  # direct-label the segments that have room
                ax.text(l0 + v / 2, y, f"{v:.0f}", ha="center", va="center",
                        fontsize=9, color="#ffffff", fontweight="bold")
        left += vals

    style_axes(ax)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("Share of that subject's coded posture time (%)", color=INK_2,
                  fontsize=10, labelpad=8)
    ax.set_ylabel("Family ID", color=INK_2, fontsize=10)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)
        lbl.set_fontsize(11)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in order]
    ax.legend(handles, order, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=min(len(order), 4), frameon=False, fontsize=10, labelcolor=INK_2,
              handlelength=1.1, handleheight=1.1, columnspacing=1.6)
    ax.set_title("Posture composition by subject", color=INK, fontsize=13,
                 fontweight="bold", loc="left", pad=14)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG_DIR / "posture_by_subject.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_segment_durations(seg: pd.DataFrame, order: list[str]) -> None:
    """Segment-length spread per class -- what a window sampler actually sees."""
    order_lo_hi = list(reversed(order))
    data = [seg.loc[seg["posture"] == p, "dur_s"].to_numpy() for p in order_lo_hi]

    fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(order) + 2.3), facecolor=SURFACE)
    bp = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=SURFACE, linewidth=2.0),
                    whiskerprops=dict(color=INK_MUTED, linewidth=1.4),
                    capprops=dict(color=INK_MUTED, linewidth=1.4),
                    boxprops=dict(linewidth=0))
    for patch in bp["boxes"]:
        patch.set_facecolor("#2a78d6")
    ax.set_yticklabels(order_lo_hi)
    ax.set_xscale("log")
    style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_xlabel("Segment duration (seconds, log scale)", color=INK_2, fontsize=10,
                  labelpad=8)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)
        lbl.set_fontsize(11)

    for y, p in enumerate(order_lo_hi, start=1):
        v = seg.loc[seg["posture"] == p, "dur_s"]
        ax.text(1.02, y, f"median {v.median():.0f}s   max {v.max():.0f}s",
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=9, color=INK_2)

    ax.set_title("Posture segment length (boxes: IQR, whiskers: 1.5×IQR, outliers hidden)",
                 color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=14)
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    fig.savefig(FIG_DIR / "posture_segment_durations.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading annotations …")
    ann = load_annotations()
    trk = load_tracking()
    lb = scan_lb_recordings()

    print(f"  {ann['source_file'].nunique()} EDITED files, {len(ann)} annotation rows")
    print("  tiers present:",
          ", ".join(f"{t}={c}" for t, c in ann["tier_norm"].value_counts().items()))

    seg = ann[ann["tier_norm"].isin(POSTURE_TIERS)].copy()
    seg = seg.rename(columns={"label_norm": "posture"})
    seg = seg[["family_id", "obs", "source_file", "tier_norm", "begin_s", "end_s",
               "dur_s", "posture"]]

    # dur_s is the exporter's own column; confirm it matches end-begin.
    drift = (seg["dur_s"] - (seg["end_s"] - seg["begin_s"])).abs().max()
    print(f"  max |dur_s - (end-begin)| = {drift:.6f} s")

    # Attach tone offsets -> is this segment placeable on the LB/IMU clock?
    seg = seg.merge(trk, on=["family_id", "obs"], how="left")
    seg["in_tracking"] = seg["lb_session"].notna()
    seg["imu_alignable"] = seg["tone_offset_s"].notna()
    seg["lb_begin_s"] = seg["begin_s"] + seg["tone_offset_s"]
    seg["lb_end_s"] = seg["end_s"] + seg["tone_offset_s"]

    total_h = hours(seg["dur_s"].sum())

    # ---------------------------------------------------------------- tables
    by_class = (
        seg.groupby("posture")
        .agg(n_segments=("dur_s", "size"), total_s=("dur_s", "sum"),
             median_s=("dur_s", "median"), mean_s=("dur_s", "mean"),
             max_s=("dur_s", "max"), n_subjects=("family_id", "nunique"),
             n_observations=("source_file", "nunique"))
        .reset_index()
    )
    by_class["hours"] = by_class["total_s"].map(hours)
    by_class["pct_time"] = by_class["total_s"] / by_class["total_s"].sum() * 100
    by_class["pct_segments"] = by_class["n_segments"] / by_class["n_segments"].sum() * 100
    by_class = by_class.sort_values("total_s", ascending=False).reset_index(drop=True)

    align = (
        seg.groupby(["posture", "imu_alignable"])["dur_s"].sum().unstack(fill_value=0.0)
    )
    for col, name in ((True, "alignable_h"), (False, "unalignable_h")):
        by_class[name] = by_class["posture"].map(
            align[col].map(hours) if col in align.columns else {}
        ).fillna(0.0)

    order = by_class["posture"].tolist()  # rank order, reused for every figure

    matrix = (
        seg.pivot_table(index="family_id", columns="posture", values="dur_s",
                        aggfunc="sum", fill_value=0.0)
        .reindex(columns=order, fill_value=0.0)
    )

    by_obs = (
        seg.groupby(["family_id", "obs", "source_file", "lb_session", "tone_offset_s",
                     "imu_alignable"], dropna=False)
        .agg(n_segments=("dur_s", "size"), coded_s=("dur_s", "sum"),
             span_end_s=("end_s", "max"), n_classes=("posture", "nunique"))
        .reset_index()
        .sort_values(["family_id", "obs"])
    )
    by_obs["coded_h"] = by_obs["coded_s"].map(hours)

    # ---------------------------------------------------------------- coverage
    ids_on_disk = sorted(int(p.name) for p in LB_DIR.iterdir() if p.is_dir())
    ids_annotated = sorted(seg["family_id"].unique())
    ids_missing = [i for i in ids_on_disk if i not in ids_annotated]

    cov = (
        lb.groupby("family_id")
        .agg(lb_sessions=("lb_session", "size"), sessions_with_imu=("has_imu", "sum"))
        .reset_index()
    )
    a = seg.groupby("family_id").agg(
        posture_segments=("dur_s", "size"), posture_h=("dur_s", lambda s: hours(s.sum())),
        observations=("source_file", "nunique"))
    cov = cov.merge(a, on="family_id", how="left").fillna(
        {"posture_segments": 0, "posture_h": 0.0, "observations": 0})
    cov["has_annotations"] = cov["family_id"].isin(ids_annotated)

    # matched LB sessions actually carrying IMU
    sess_ok = set(lb.loc[lb["has_imu"], "lb_session"])
    by_obs["lb_session_on_disk"] = by_obs["lb_session"].isin(set(lb["lb_session"]))
    by_obs["lb_session_has_imu"] = by_obs["lb_session"].isin(sess_ok)

    # ------------------------------------------------------------------ write
    seg.to_csv(OUT_DIR / "posture_segments.csv", index=False)
    by_class.to_csv(OUT_DIR / "posture_by_class.csv", index=False)
    matrix.to_csv(OUT_DIR / "posture_by_subject_seconds.csv")
    by_obs.to_csv(OUT_DIR / "posture_by_observation.csv", index=False)
    cov.to_csv(OUT_DIR / "annotation_coverage.csv", index=False)

    fig_by_class(by_class, total_h)
    fig_by_subject(matrix, order)
    fig_segment_durations(seg, order)

    # ---------------------------------------------------------------- summary
    w = by_class[["posture", "n_segments", "hours", "pct_time", "median_s", "max_s",
                  "n_subjects", "alignable_h"]].copy()
    w.columns = ["posture", "n_seg", "hours", "% time", "med_s", "max_s", "n_subj",
                 "align_h"]
    print("\n" + "=" * 78)
    print(f"POSTURE TIER — {total_h:.2f} h ({hms(seg['dur_s'].sum())}) over "
          f"{len(seg)} segments, {seg['source_file'].nunique()} observations, "
          f"{seg['family_id'].nunique()} subjects")
    print("=" * 78)
    print(w.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    al = hours(seg.loc[seg["imu_alignable"], "dur_s"].sum())
    print(f"\nIMU-alignable (tone offset known): {al:.2f} h of {total_h:.2f} h "
          f"({al / total_h * 100:.1f}%)")
    ua = by_obs.loc[~by_obs["imu_alignable"]]
    if len(ua):
        print(f"  {len(ua)} observations have no LB tone and cannot be placed on the "
              f"IMU clock ({ua['coded_h'].sum():.2f} h):")
        print(ua[["family_id", "obs", "lb_session", "n_segments", "coded_h"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    nt = seg.loc[~seg["in_tracking"], "source_file"].unique()
    if len(nt):
        print(f"Not in tracking sheet: {', '.join(sorted(nt))}")
    bad = by_obs.loc[by_obs["lb_session_on_disk"] & ~by_obs["lb_session_has_imu"],
                     "lb_session"].unique()
    if len(bad):
        print(f"LB sessions present but without _imu_sync.txt: {len(bad)}")
    off = by_obs.loc[~by_obs["lb_session_on_disk"] & by_obs["lb_session"].notna(),
                     "lb_session"].unique()
    if len(off):
        print(f"Tracked LB sessions not found under {LB_DIR}: {len(off)}")

    print(f"\nSubjects on disk: {ids_on_disk}")
    print(f"Subjects with posture annotations: {ids_annotated}")
    print(f"No annotations (no video to correct against): {ids_missing}")
    print("\nPer-subject coverage:")
    print(cov.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nWrote tables to {OUT_DIR}, figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
