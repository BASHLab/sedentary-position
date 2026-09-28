#!/usr/bin/env python3
"""How much does counting `drowsy` as sleep vs awake change tummy time?

The ECG model maps drowsy / drowsy unsure to **sleep**, and tummy_time.py
follows it so the model and the human reference mean the same thing. That is
debatable -- a drowsy infant on its stomach arguably is tummy time -- so this
measures the size of the choice rather than arguing about it.

Compares the two already-written window tables, which differ only in that map.
"""
import pandas as pd, numpy as np, re
from pathlib import Path
OUT=Path("/work/hdd/bebr/Projects/IMU_pretraining/analysis/posture")
ANN=Path("/work/hdd/bebr/Data/Behavioral_Coding/BRP1_EMAannotations")
COLS=["tier","blank","begin_hms","begin_s","end_hms","end_s","dur_hms","dur_s","label"]

# raw segment-level counts straight from the state tier
rows=[]
for p in sorted(ANN.glob("*_EDITED.txt")):
    d=pd.read_csv(p,sep="\t",header=None,names=COLS,dtype={"label":"string"},
                  keep_default_na=False,na_values=[])
    d=d[d["tier"].str.strip().str.lower()=="state"]
    if len(d): rows.append(d)
st=pd.concat(rows,ignore_index=True)
st["lab"]=st["label"].str.strip().str.lower()
st["dur"]=pd.to_numeric(st["dur_s"],errors="coerce")
print("state tier, segment counts and coded duration:")
g=st.groupby("lab").agg(n=("dur","size"),hours=("dur",lambda s:s.sum()/3600)).sort_values("hours",ascending=False)
print(g.to_string(float_format=lambda v:f"{v:.2f}"))
dro=g.loc[[i for i in g.index if "drowsy" in i]]
print(f"\n  drowsy total: {int(dro['n'].sum())} segments, {dro['hours'].sum():.2f} h "
      f"({dro['hours'].sum()/g['hours'].sum()*100:.1f}% of coded state time)")

# window-level: the two runs differ ONLY in how drowsy is mapped
a=pd.read_parquet(OUT/"tummy_time_windows.parquet")
b=pd.read_parquet(OUT/"tummy_time_windows_drowsy_awake.parquet")
key=["family_id","obs","t_start_s"]
m=a[key+["coded_state","posture","tummy_annotated"]].merge(
    b[key+["coded_state","tummy_annotated"]],on=key,suffixes=("_sleepmap","_wakemap"))
flip=m["coded_state_sleepmap"]!=m["coded_state_wakemap"]
print(f"\nwindows whose coded state flips when drowsy->awake: {int(flip.sum()):,} "
      f"({flip.mean()*100:.2f}% of {len(m):,})")
print("  their posture breakdown:")
print(m.loc[flip,"posture"].value_counts().to_string())
add=m["tummy_annotated_wakemap"]&~m["tummy_annotated_sleepmap"]
print(f"\ndrowsy AND on stomach -> windows added to tummy time: {int(add.sum()):,} "
      f"= {add.sum()*5/3600:.3f} h")
print(f"  tummy time  drowsy=sleep {m['tummy_annotated_sleepmap'].sum()*5/3600:.2f} h"
      f"  ->  drowsy=awake {m['tummy_annotated_wakemap'].sum()*5/3600:.2f} h"
      f"  (+{add.sum()*5/3600/ (m['tummy_annotated_sleepmap'].sum()*5/3600)*100:.1f}%)")
