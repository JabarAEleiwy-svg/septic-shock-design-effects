#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_gcs_missing.py - the previous run showed: 7,332 of 29,566 MIMIC rows (25%) have gcs_min missing and are scored s_cns = 0
(normal) by the SOFA code; their event rate is 24.7% vs 5.6-10.0% for observed GCS.  Among OBSERVED GCS the cns score is positive.
So the inversion looks like a missing-data artifact of the hourly GCS construction (eye+verbal+motor must all be charted in the
same hour), not neurology.  This script tests that directly (pre-specified):
 (a) SOFA / qSOFA / s_cns AUROC restricted to rows with OBSERVED GCS, all rows and temporal-test rows, any-time and 24 h outcomes;
 (b) who has missing GCS: hours_icu_to_pred, on_pressor, pf_min present, lactate present, age, by calendar group;
 (c) event rate by hours_icu_to_pred bin x GCS missing.
Aggregated outputs only (--out/diagnostics).
"""
import argparse, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/content/drive/MyDrive/sepsis_project"); ap.add_argument("--nboot", type=int, default=500); a = ap.parse_args()
D = os.path.join(a.out, "diagnostics"); os.makedirs(D, exist_ok=True)
d0 = pd.read_parquet(f"{a.out}/features_mimic.parquet"); num = d0.select_dtypes("number").columns; d0[num] = d0[num].replace([np.inf, -np.inf], np.nan)
TRAIN = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
def auc(y, p):
    m = ~np.isnan(p); return roc_auc_score(y[m], p[m]) if 0 < y[m].sum() < m.sum() else np.nan
def ci(y, p, cl, seed=1):
    rng = np.random.default_rng(seed); codes, u = pd.factorize(cl); g = [np.flatnonzero(codes == i) for i in range(len(u))]; v = []
    for _ in range(a.nboot):
        i = np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]); v.append(auc(y[i], p[i]))
    v = np.array(v); v = v[~np.isnan(v)]; return np.percentile(v, [2.5, 97.5])
rows = []
for H in (0, 24):
    for part, pm in (("all", np.ones(len(d0), bool)), ("temporal test 2017-2022", ~d0.anchor_year_group.isin(TRAIN).values)):
        for sub, sm in (("GCS observed", d0.gcs_min.notna().values), ("GCS missing", d0.gcs_min.isna().values)):
            d = d0[pm & sm].reset_index(drop=True)
            y = (d.shock == 1).astype(int).values if H == 0 else ((d.shock == 1) & (d.hours_to_shock <= H)).astype(int).values
            tag = f"{'any time' if H == 0 else f'within {H}h'} | {part} | {sub}"
            q = (d.rr_max >= 22).astype(float) + (d.sbp_min <= 100)
            for lab, p in [("SOFA total", d.sofa_total), ("SOFA without s_cns", d.sofa_total - d.s_cns), ("s_cns", d.s_cns), ("qSOFA approx", q + (d.gcs_min < 15)), ("lactate_last", d.lactate_last)]:
                p = np.asarray(p, float)
                if y.sum() < 20: continue
                lo, hi = ci(y, p, d.subject_id.values); rows.append(dict(subset=tag, label=lab, n=len(y), events=int(y.sum()), auroc=auc(y, p), lo=lo, hi=hi))
r = pd.DataFrame(rows); r["formatted"] = r.apply(lambda x: f"{x.auroc:.3f} ({x.lo:.3f}-{x.hi:.3f})", axis=1); r.to_csv(f"{D}/gcs_missing_auroc.csv", index=False)
m = d0.assign(gcs_missing=d0.gcs_min.isna(), y=(d0.shock == 1).astype(int))
prof = m.groupby("gcs_missing").agg(n=("y", "size"), event_rate=("y", "mean"), hours_icu_to_pred_median=("hours_icu_to_pred", "median"), hours_icu_to_pred_q1=("hours_icu_to_pred", lambda x: x.quantile(.25)),
        hours_icu_to_pred_q3=("hours_icu_to_pred", lambda x: x.quantile(.75)), on_pressor_pct=("on_pressor", lambda x: 100 * x.mean()), pf_min_present_pct=("pf_min", lambda x: 100 * x.notna().mean()),
        lactate_present_pct=("lactate_last", lambda x: 100 * x.notna().mean()), age_median=("age", "median"), sofa_median=("sofa_total", "median")).round(3).reset_index()
prof.to_csv(f"{D}/gcs_missing_profile.csv", index=False)
byg = m.groupby("anchor_year_group").gcs_missing.agg(["size", "mean"]).round(3).reset_index().rename(columns={"mean": "gcs_missing_share"}); byg.to_csv(f"{D}/gcs_missing_by_period.csv", index=False)
m["h_bin"] = pd.cut(m.hours_icu_to_pred, [-1, 0, 1, 3, 6, 12, 24, 1e6]).astype(str)
byh = m.groupby(["h_bin", "gcs_missing"]).y.agg(["size", "mean"]).round(3).reset_index().rename(columns={"mean": "event_rate"}); byh.to_csv(f"{D}/gcs_missing_by_hours.csv", index=False)
pd.set_option("display.width", 220, "display.max_rows", 500)
print(r[["subset", "label", "n", "events", "formatted"]].to_string(index=False)); print("\n", prof.T.to_string(), "\n\n", byg.to_string(index=False), "\n\n", byh.to_string(index=False)); print(f"\nSaved to {D}")
