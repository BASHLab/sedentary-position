#!/usr/bin/env python3
"""Tummy time = infant **awake** AND **on stomach**.

Two signals have to agree for a window to count:

    posture   on stomach      <- IMU, harnet10 fine-tuned (LOSO out-of-fold)
    state     awake           <- ECG sleep/wake model, or the human coding

Both are evaluated on the same 48,135 ten-second windows the rest of this
project uses, so the tummy-time number inherits that dataset's provenance
exactly.

Wake/sleep vocabulary
---------------------
Taken verbatim from the ECG model's own parser
(`ecg_foundation_model/single_stream_downstream/dataset.py::_parse_sleep_segments`)
so that the annotation-derived reference and the ECG model's predictions mean
the same thing:

    wake  : quiet alert, active alert, active, crying
    sleep : drowsy, drowsy unsure, light sleep, deep sleep

Note `drowsy` counts as **sleep**, so a drowsy infant on its stomach is not
tummy time here. That is the ECG model's convention, not a clinical one; it is
the single most debatable line in this script and `--drowsy-awake` flips it.

How time is counted
-------------------
Windows are 10 s long on a 5 s hop, so they overlap by half and simply summing
their lengths would double-count. Each window is credited with one hop (5 s),
which makes the totals a true partition of covered time, and the same rule is
applied to the reference and to every prediction so the columns are comparable.

Sources, in increasing order of "how much is modelled":

    annotated       coded posture + coded state        the reference
    imu_only        harnet10 posture + coded state     isolates the IMU model
    predicted       harnet10 posture + ECG state       fully model-derived
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/work/hdd/bebr/Projects/IMU_pretraining")
OUT_DIR = PROJ / "analysis" / "posture"
ANNOT_DIR = Path("/work/hdd/bebr/Data/Behavioral_Coding/BRP1_EMAannotations")

COLUMNS = ["tier", "blank", "begin_hms", "begin_s", "end_hms", "end_s",
           "dur_hms", "dur_s", "label"]
FNAME_RE = re.compile(r"^BRP1_(?P<fid>\d+)_Obs(?P<obs>\d+[A-Za-z]?)_EDITED\.txt$")

# verbatim from the ECG model's _parse_sleep_segments
STATE_MAP = {
    "drowsy": "sleep", "drowsy unsure": "sleep",
    "light sleep": "sleep", "deep sleep": "sleep",
    "crying": "wake", "active alert": "wake",
    "active": "wake", "quiet alert": "wake",
}

WIN_S, HOP_S, MIN_PURITY = 10.0, 5.0, 0.80
STOMACH = "on stomach"


def load_state_segments() -> pd.DataFrame:
    """Every `state` annotation, on the LB clock, via the same tone offsets."""
    seg = pd.read_csv(OUT_DIR / "posture_segments.csv")
    tone = (seg[["family_id", "obs", "lb_session", "tone_offset_s"]]
            .drop_duplicates(subset=["family_id", "obs"]))

    rows = []
    for path in sorted(ANNOT_DIR.glob("*_EDITED.txt")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        df = pd.read_csv(path, sep="\t", header=None, names=COLUMNS,
                         dtype={"label": "string"}, keep_default_na=False,
                         na_values=[])
        df = df[df["tier"].str.strip().str.lower() == "state"].copy()
        if df.empty:
            continue
        df["family_id"] = int(m.group("fid"))
        df["obs"] = m.group("obs").upper()
        df["state"] = df["label"].str.strip().str.lower().map(STATE_MAP)
        for c in ("begin_s", "end_s"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        rows.append(df[["family_id", "obs", "begin_s", "end_s", "state", "label"]])

    st = pd.concat(rows, ignore_index=True)
    unmapped = st.loc[st["state"].isna(), "label"].str.strip().str.lower()
    if len(unmapped):
        print("  state labels not in the ECG model's vocabulary (dropped): "
              + ", ".join(f"{k!r}x{v}" for k, v in unmapped.value_counts().items()))
    st = st[st["state"].notna()]
    st = st.merge(tone, on=["family_id", "obs"], how="left")
    st = st[st["tone_offset_s"].notna()]
    st["lb_begin_s"] = st["begin_s"] + st["tone_offset_s"]
    st["lb_end_s"] = st["end_s"] + st["tone_offset_s"]
    return st


def window_state(md: pd.DataFrame, st: pd.DataFrame) -> np.ndarray:
    """Per-window coded state, by the same >=80%-coverage rule as posture."""
    out = np.full(len(md), "", dtype=object)
    idx_of = {k: i for i, k in enumerate(zip(md["family_id"], md["obs"],
                                             md["t_start_s"]))}
    for (fid, obs), g in st.groupby(["family_id", "obs"]):
        sel = md.index[(md["family_id"] == fid) & (md["obs"] == obs)]
        if len(sel) == 0:
            continue
        starts = md.loc[sel, "t_start_s"].to_numpy()
        for lab in ("wake", "sleep"):
            sub = g[g["state"] == lab]
            cov = np.zeros(len(starts))
            for b, e in zip(sub["lb_begin_s"], sub["lb_end_s"]):
                cov += np.clip(np.minimum(starts + WIN_S, e)
                               - np.maximum(starts, b), 0, None)
            hit = cov >= WIN_S * MIN_PURITY
            for i, h in zip(sel, hit):
                if h:
                    out[md.index.get_loc(i)] = lab
    return out


def summarise(md: pd.DataFrame, cols: dict[str, np.ndarray]) -> pd.DataFrame:
    """Hours of tummy time per infant for each source, plus coverage."""
    rows = []
    for fid, g in md.groupby("family_id"):
        r = {"family_id": int(fid), "windows": len(g),
             "covered_h": len(g) * HOP_S / 3600.0}
        for name, v in cols.items():
            r[f"{name}_h"] = float(v[g.index].sum() * HOP_S / 3600.0)
        rows.append(r)
    tot = {"family_id": "ALL", "windows": len(md),
           "covered_h": len(md) * HOP_S / 3600.0}
    for name, v in cols.items():
        tot[f"{name}_h"] = float(v.sum() * HOP_S / 3600.0)
    return pd.DataFrame(rows + [tot])


def agreement(ref: np.ndarray, pred: np.ndarray, name: str) -> dict:
    tp = int((ref & pred).sum())
    fp = int((~ref & pred).sum())
    fn = int((ref & ~pred).sum())
    tn = int((~ref & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    return {"source": name, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "pred_h": (tp + fp) * HOP_S / 3600.0,
            "ref_h": (tp + fn) * HOP_S / 3600.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg-pred", type=Path,
                    default=OUT_DIR / "ecg_sleep_predictions.parquet",
                    help="per-window ECG sleep/wake predictions; optional")
    ap.add_argument("--drowsy-awake", action="store_true",
                    help="count drowsy / drowsy unsure as awake")
    args = ap.parse_args()

    tag = ""
    if args.drowsy_awake:
        STATE_MAP["drowsy"] = STATE_MAP["drowsy unsure"] = "wake"
        tag = "_drowsy_awake"
        print("drowsy / drowsy unsure counted as AWAKE (non-default)")

    md = pd.read_parquet(OUT_DIR / "windows_raw_30hz_meta.parquet").reset_index(drop=True)
    classes = sorted(md["posture"].unique())
    print(f"{len(md):,} windows | {md['family_id'].nunique()} infants")

    print("\nLoading `state` annotations ...")
    st = load_state_segments()
    print(f"  {len(st):,} state segments over "
          f"{st.groupby(['family_id', 'obs']).ngroups} observations")

    coded_state = window_state(md, st)
    n_wake = int((coded_state == "wake").sum())
    n_sleep = int((coded_state == "sleep").sum())
    print(f"  windows with a coded state at >={MIN_PURITY:.0%} coverage: "
          f"{n_wake + n_sleep:,} ({(n_wake + n_sleep) / len(md):.1%})"
          f"  -> wake {n_wake:,}  sleep {n_sleep:,}")

    # ---------------------------------------------------------------- posture
    pred_path = OUT_DIR / "harnet10_full_predictions.npy"
    pred_idx = np.load(pred_path)
    pred_posture = np.array(classes, dtype=object)[pred_idx]
    print(f"\nharnet10 posture (LOSO out-of-fold): "
          f"on-stomach predicted for {(pred_posture == STOMACH).sum():,} windows, "
          f"coded for {(md['posture'] == STOMACH).sum():,}")

    annot_stomach = (md["posture"] == STOMACH).to_numpy()
    pred_stomach = pred_posture == STOMACH
    coded_wake = coded_state == "wake"

    cols = {
        "annotated": annot_stomach & coded_wake,
        "imu_only": pred_stomach & coded_wake,
    }

    # ------------------------------------------------------------------- ECG
    ecg_covered = None
    ecg_ok = args.ecg_pred.exists()
    if ecg_ok:
        e = pd.read_parquet(args.ecg_pred)
        key = ["family_id", "obs", "t_start_s"]
        e = md[key].merge(e, on=key, how="left")
        ecg_wake = (e["ecg_state"] == "wake").to_numpy()
        ecg_covered = e["ecg_state"].notna().to_numpy()
        cols["predicted"] = pred_stomach & ecg_wake
        n_have = int(ecg_covered.sum())
        print(f"ECG sleep/wake: {n_have:,}/{len(md):,} windows covered "
              f"({n_have / len(md):.1%})")
    else:
        print(f"\n! {args.ecg_pred.name} not found -- reporting the annotated "
              f"reference and the IMU-only column only.\n"
              f"  Run scripts/ecg_sleep_infer.py to add the fully-predicted column.")

    # --------------------------------------------------------------- reports
    summ = summarise(md, cols)
    if ecg_covered is not None:
        frac = (pd.Series(ecg_covered).groupby(md["family_id"]).mean()
                .rename("ecg_coverage"))
        summ = summ.merge(frac.reset_index().rename(
            columns={"family_id": "family_id"}), on="family_id", how="left")
        summ.loc[summ["family_id"] == "ALL", "ecg_coverage"] = ecg_covered.mean()
    summ.to_csv(OUT_DIR / f"tummy_time_by_subject{tag}.csv", index=False)
    print("\n" + "=" * 78)
    print("TUMMY TIME (hours) -- awake AND on stomach")
    print("=" * 78)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    ref = cols["annotated"]
    ag = [agreement(ref, v, k) for k, v in cols.items() if k != "annotated"]
    if ecg_covered is not None:
        # Windows with no ECG can only ever be false negatives for `predicted`,
        # so scoring it over the whole corpus charges it for 25007's missing
        # recordings. Restrict to where the ECG model could actually speak.
        for k, v in cols.items():
            if k == "annotated":
                continue
            r = agreement(ref[ecg_covered], v[ecg_covered],
                          f"{k} (ECG-covered only)")
            ag.append(r)
    if ag:
        A = pd.DataFrame(ag)
        A.to_csv(OUT_DIR / f"tummy_time_agreement{tag}.csv", index=False)
        print("\nAgreement against the annotated reference (per window):")
        print(A.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    per_window = md[["family_id", "obs", "lb_session", "t_start_s", "posture"]].copy()
    per_window["coded_state"] = coded_state
    per_window["pred_posture"] = pred_posture
    for k, v in cols.items():
        per_window[f"tummy_{k}"] = v
    per_window.to_parquet(OUT_DIR / f"tummy_time_windows{tag}.parquet", index=False)
    print(f"\nWrote tummy_time_by_subject{tag}.csv, tummy_time_agreement{tag}.csv, "
          f"tummy_time_windows{tag}.parquet to {OUT_DIR}")


if __name__ == "__main__":
    main()
