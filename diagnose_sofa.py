#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_sofa.py - why is SOFA-only / qSOFA AUROC below 0.5 in MIMIC?
Pre-specified hypotheses (each has a test; nothing is tuned):
 H1  Cohort construction removes patients already in shock at the index (shock_time must be > index), so the remaining
     high-SOFA patients are those on vasopressors WITHOUT the lactate criterion -> they are non-events with high s_cv.
     Test: AUROC of SOFA in strata on_pressor 0/1; SOFA excluding the cardiovascular component; event rate by on_pressor.
 H2  Direction/encoding: SOFA missing filled with 0, qSOFA approximated from rr/sbp/gcs.  Test: missing share, AUROC with the
     missing rows dropped, qSOFA components separately.
 H3  Calendar/era effect.  Test: AUROC by anchor_year_group.
 H4  Time-since-admission effect (index defined as the latest of 3 times).  Test: AUROC by hours_icu_to_pred tercile.
Reports every component AUROC so the sign of each association is visible.  Aggregated outputs only (--out/diagnostics).
"""
import argparse, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/content/drive/MyDrive/sepsis_project"); ap.add_argument("--horizon", type=int, default=0, help="0 = shock at any time (primary outcome)")
ap.add_argument("--nboot", type=int, default=500); a = ap.parse_args()
D = os.path.join(a.out, "diagnostics"); os.makedirs(D, exist_ok=True)
def auc(y, p):
    m = ~np.isnan(p)
    return roc_auc_score(y[m], p[m]) if 0 < y[m].sum() < m.sum() else np.nan
def ci(y, p, cl, nb=a.nboot, seed=1):
    rng = np.random.default_rng(seed); codes, u = pd.factorize(cl); g = [np.flatnonzero(codes == i) for i in range(len(u))]; v = []
    for _ in range(nb):
        i = np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]); v.append(auc(y[i], p[i]))
    v = np.array(v); v = v[~np.isnan(v)]; return np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
rows = []
for name, f, cl in [("MIMIC", "features_mimic", "subject_id"), ("eICU", "features_eicu", "hospitalid")]:
    d = pd.read_parquet(f"{a.out}/{f}.parquet"); num = d.select_dtypes("number").columns; d[num] = d[num].replace([np.inf, -np.inf], np.nan)
    y = (d.shock == 1).astype(int).values if a.horizon == 0 else ((d.shock == 1) & (d.hours_to_shock <= a.horizon)).astype(int).values
    cls = d[cl].values
    def add(test, label, p, mask=None):
        m = np.ones(len(d), bool) if mask is None else np.asarray(mask)
        yy, pp = y[m], np.asarray(p, float)[m]
        if yy.sum() < 10 or (len(yy) - yy.sum()) < 10: rows.append(dict(dataset=name, test=test, label=label, n=int(m.sum()), events=int(yy.sum()), auroc=np.nan, lo=np.nan, hi=np.nan)); return
        lo, hi = ci(yy, pp, cls[m]); rows.append(dict(dataset=name, test=test, label=label, n=int(m.sum()), events=int(yy.sum()), auroc=auc(yy, pp), lo=lo, hi=hi))
    s = d.sofa_total
    add("overall", "SOFA total (NaN->0)", s.fillna(0)); add("overall", "SOFA total (rows with SOFA only)", s)
    for c in ["s_resp", "s_coag", "s_liver", "s_cv", "s_cns", "s_renal", "gcs_min", "pf_min", "on_pressor", "age"]:
        if c in d: add("component", c, d[c])
    for c in ["lactate_last", "map_min", "sbp_min", "rr_max", "hr_max", "temp_max", "creat_last", "plt_last", "bili_last"]:
        if c in d: add("variable", c, d[c])
    nocv = d[["s_resp", "s_coag", "s_liver", "s_cns", "s_renal"]].sum(axis=1, min_count=1)
    add("H1", "SOFA without cardiovascular component", nocv)
    if "on_pressor" in d:
        for v in (0, 1): add("H1", f"SOFA | on_pressor={v}", s, d.on_pressor == v)
        for v in (0, 1): rows.append(dict(dataset=name, test="H1", label=f"event rate | on_pressor={v}", n=int((d.on_pressor == v).sum()), events=int(y[d.on_pressor == v].sum()), auroc=y[d.on_pressor == v].mean(), lo=np.nan, hi=np.nan))
    add("H1", "SOFA | s_cv<3", s, d.s_cv < 3)
    rows.append(dict(dataset=name, test="H2", label="share of SOFA missing", n=len(d), events=int(y.sum()), auroc=s.isna().mean(), lo=np.nan, hi=np.nan))
    add("H2", "SOFA | rows with SOFA", s, s.notna())
    q_rr, q_sbp, q_gcs = (d.rr_max >= 22), (d.sbp_min <= 100), (d.gcs_min < 15)
    for lab, p in [("qSOFA approx (rr>=22+sbp<=100+gcs<15)", q_rr.astype(float) + q_sbp + q_gcs), ("rr_max>=22", q_rr.astype(float)), ("sbp_min<=100", q_sbp.astype(float)), ("gcs_min<15", q_gcs.astype(float))]:
        add("H2", lab, p)
    rows.append(dict(dataset=name, test="H2", label="share gcs_min<15 among non-missing", n=int(d.gcs_min.notna().sum()), events=int(y.sum()), auroc=float((d.gcs_min < 15)[d.gcs_min.notna()].mean()), lo=np.nan, hi=np.nan))
    if "anchor_year_group" in d:
        for g, ix in d.groupby("anchor_year_group").groups.items(): add("H3", f"SOFA | {g}", s.fillna(0), d.index.isin(ix))
    t = pd.qcut(d.hours_icu_to_pred, 3, duplicates="drop", labels=False)
    for g in sorted(t.dropna().unique()): add("H4", f"SOFA | hours_icu_to_pred tercile {int(g)+1}", s.fillna(0), t == g)
    # event rate by SOFA bin (shape of the association)
    b = pd.cut(s, [-1, 1, 3, 5, 7, 9, 11, 30]); r = pd.DataFrame({"y": y, "b": b}).groupby("b", observed=True).y.agg(["mean", "size"]).reset_index()
    r.insert(0, "dataset", name); r.to_csv(f"{D}/sofa_bin_event_rate_{name}.csv", index=False)
out = pd.DataFrame(rows); out["formatted"] = out.apply(lambda r: f"{r.auroc:.3f} ({r.lo:.3f}-{r.hi:.3f})" if pd.notna(r.lo) else (f"{r.auroc:.3f}" if pd.notna(r.auroc) else "n/a"), axis=1)
out.to_csv(f"{D}/sofa_direction_diagnostic.csv", index=False)
pd.set_option("display.width", 200, "display.max_rows", 500); print(out[["dataset", "test", "label", "n", "events", "formatted"]].to_string(index=False))
print("\nNOTE: for gcs_min, pf_min, map_min, sbp_min, plt_last, a LOWER value means worse, so AUROC < 0.5 for them is the expected direction.")
print(f"Saved to {D}")
