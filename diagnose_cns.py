#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_cns.py - follow-up to diagnose_sofa.py.  Pre-specified question: the inverse association of SOFA/qSOFA with shock in MIMIC
is carried by the neurological component (s_cns AUROC 0.42, gcs<15 AUROC 0.41), not by the cardiovascular component.
Tests: (1) event rate by GCS category incl. missing and by s_cns; (2) SOFA with s_cns removed / s_cns+s_cv removed;
(3) qSOFA without the GCS item; (4) event rate by missing pf_min (ventilation proxy) and by first_careunit if present in
cohort_final_mimic; (5) the same on the temporal TEST period only (2017-2022) to match the reported comparator numbers.
Outcomes: shock at any time, and within 24 h.  Aggregated outputs only (--out/diagnostics).
"""
import argparse, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/content/drive/MyDrive/sepsis_project"); ap.add_argument("--nboot", type=int, default=500); a = ap.parse_args()
D = os.path.join(a.out, "diagnostics"); os.makedirs(D, exist_ok=True)
d0 = pd.read_parquet(f"{a.out}/features_mimic.parquet"); num = d0.select_dtypes("number").columns; d0[num] = d0[num].replace([np.inf, -np.inf], np.nan)
try:
    co = pd.read_parquet(f"{a.out}/cohort_final_mimic.parquet"); cu = [c for c in co.columns if "careunit" in c.lower()]
    if cu: d0 = d0.merge(co[["stay_id", cu[0]]].rename(columns={cu[0]: "careunit"}), on="stay_id", how="left")
except Exception as e: print("careunit not available:", e)
TRAIN = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
def auc(y, p):
    m = ~np.isnan(p); return roc_auc_score(y[m], p[m]) if 0 < y[m].sum() < m.sum() else np.nan
def ci(y, p, cl, seed=1):
    rng = np.random.default_rng(seed); codes, u = pd.factorize(cl); g = [np.flatnonzero(codes == i) for i in range(len(u))]; v = []
    for _ in range(a.nboot):
        i = np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]); v.append(auc(y[i], p[i]))
    v = np.array(v); v = v[~np.isnan(v)]; return np.percentile(v, [2.5, 97.5])
rows, rates = [], []
for H in (0, 24):
    for part, mask in (("all", np.ones(len(d0), bool)), ("temporal test 2017-2022", ~d0.anchor_year_group.isin(TRAIN).values)):
        d = d0[mask].reset_index(drop=True)
        y = (d.shock == 1).astype(int).values if H == 0 else ((d.shock == 1) & (d.hours_to_shock <= H)).astype(int).values
        tag = f"{'any time' if H == 0 else f'within {H}h'} | {part}"
        def add(label, p):
            p = np.asarray(p, float); lo, hi = ci(y, p, d.subject_id.values); rows.append(dict(outcome=tag, label=label, n=len(y), events=int(y.sum()), auroc=auc(y, p), lo=lo, hi=hi))
        s = d.sofa_total
        add("SOFA total", s.fillna(0)); add("SOFA without s_cns", s - d.s_cns); add("SOFA without s_cns and s_cv", s - d.s_cns - d.s_cv)
        add("s_cns", d.s_cns); add("gcs_min (lower=worse; AUROC<0.5 expected)", d.gcs_min)
        q = (d.rr_max >= 22).astype(float) + (d.sbp_min <= 100); add("qSOFA approx", q + (d.gcs_min < 15)); add("qSOFA approx without GCS item", q)
        g = pd.cut(d.gcs_min, [0, 8, 13, 14, 15], labels=["3-8", "9-13", "14", "15"]).astype(object).where(d.gcs_min.notna(), "missing")
        for k, v in pd.DataFrame({"y": y, "g": g}).groupby("g").y.agg(["size", "mean"]).iterrows(): rates.append(dict(outcome=tag, variable="gcs_min category", level=k, n=int(v["size"]), event_rate=v["mean"]))
        for k, v in pd.DataFrame({"y": y, "g": d.s_cns}).groupby("g").y.agg(["size", "mean"]).iterrows(): rates.append(dict(outcome=tag, variable="s_cns", level=k, n=int(v["size"]), event_rate=v["mean"]))
        pm = d.pf_min.isna().map({True: "pf_min missing", False: "pf_min present"})
        for k, v in pd.DataFrame({"y": y, "g": pm}).groupby("g").y.agg(["size", "mean"]).iterrows(): rates.append(dict(outcome=tag, variable="ventilation proxy", level=k, n=int(v["size"]), event_rate=v["mean"]))
        if "careunit" in d and part == "all":
            for k, v in pd.DataFrame({"y": y, "g": d.careunit.fillna("missing")}).groupby("g").y.agg(["size", "mean"]).iterrows():
                if v["size"] >= 100: rates.append(dict(outcome=tag, variable="first_careunit", level=k, n=int(v["size"]), event_rate=v["mean"]))
r = pd.DataFrame(rows); r["formatted"] = r.apply(lambda x: f"{x.auroc:.3f} ({x.lo:.3f}-{x.hi:.3f})", axis=1); r.to_csv(f"{D}/cns_diagnostic_auroc.csv", index=False)
t = pd.DataFrame(rates); t["event_rate"] = t.event_rate.round(3); t.to_csv(f"{D}/cns_diagnostic_event_rates.csv", index=False)
pd.set_option("display.width", 200, "display.max_rows", 500)
print(r[["outcome", "label", "n", "events", "formatted"]].to_string(index=False)); print(); print(t.to_string(index=False)); print(f"\nSaved to {D}")
