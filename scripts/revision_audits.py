#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
revision_audits.py  -  three audits requested in peer review (v5 -> v6).  Runs on the files in --out (sepsis_project/).
Evaluation / counting only: nothing is retrained.  Aggregated outputs (no patient rows) are written to <out>/revision_audits/.

 AUDIT 1  Lactate measurement by outcome (informative presence of the outcome definition)
          The outcome needs a lactate > 2 within +-6 h of a vasopressor hour.  A stay that received a vasopressor after the index
          but had NO lactate measured within +-6 h of ANY vasopressor hour can never be labelled as shock.  The audit counts them
          ("unclassifiable"), compares them with events and other non-events, and reports how much the event count could change.
 AUDIT 2  Calendar-period (COVID-19) sensitivity of the primary model on the MIMIC-IV temporal test set, by anchor-year group.
 AUDIT 3  eICU-CRD hospital audit: per-hospital recording of vasopressors and lactate, and the primary model's AUROC in
          hospital subsets defined by median splits of those recording rates (no arbitrary thresholds).

Inputs read: features_mimic, features_eicu, cohort_final_mimic, sofa_hourly (stay_id, hr, lactate, any_pressor),
             sofa_hourly_eicu, results/private/predictions_temporal_test, results/private/predictions_eicu.
Self-check: the primary outcome re-derived from the hourly tables (outcome_definitions.py of the repository) must reproduce
`shock` / `hours_to_shock` of the feature tables; otherwise the run stops (use --no_outcome_check only for synthetic tests).
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/content/drive/MyDrive/sepsis_project")
ap.add_argument("--proj", default="/content/proj/scripts", help="folder with outcome_definitions.py (cloned repository)")
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--no_outcome_check", action="store_true")
ap.add_argument("--fast", action="store_true")
a = ap.parse_args()
OUT = a.out; NB = 100 if a.fast else a.nboot
WD = os.path.join(OUT, "revision_audits"); os.makedirs(WD, exist_ok=True)
LOG = open(os.path.join(WD, "audit_log.txt"), "w")
def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

TRAIN_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
fm = pd.read_parquet(f"{OUT}/features_mimic.parquet"); fe = pd.read_parquet(f"{OUT}/features_eicu.parquet")
coh = pd.read_parquet(f"{OUT}/cohort_final_mimic.parquet")[["stay_id", "index_time"]]
say(f"MIMIC stays {len(fm):,} (events {int(fm.shock.sum()):,}) | eICU stays {len(fe):,} (events {int(fe.shock.sum()):,}, {fe.hospitalid.nunique()} hospitals)")
fe = fe.assign(index_hr=fe.hours_icu_to_pred - 1)


# ---------------------------------------------------------------- helpers
def make_boot(clusters, nb, seed):
    rng = np.random.default_rng(seed); codes = pd.factorize(pd.Series(clusters))[0]
    order = np.argsort(codes, kind="stable"); groups = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    return [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(nb)]

def auc_ci(y, p, clusters, seed=0):
    y = np.asarray(y); p = np.asarray(p)
    if y.sum() == 0 or y.sum() == len(y): return np.nan, np.nan, np.nan
    v = []
    for i in make_boot(clusters, NB, seed):
        yy = y[i]
        v.append(roc_auc_score(yy, p[i]) if 0 < yy.sum() < len(yy) else np.nan)
    v = np.array(v); v = v[~np.isnan(v)]
    return roc_auc_score(y, p), np.percentile(v, 2.5), np.percentile(v, 97.5)

fmt = lambda t: f"{t[0]:.3f} ({t[1]:.3f}-{t[2]:.3f})"


# ---------------------------------------------------------------- hourly-table scan (shared by audits 1 and 3)
def scan_mimic():
    con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=2")
    con.register("coh", coh)
    src = f"read_parquet('{OUT}/sofa_hourly.parquet')"
    cols = set(con.execute(f"SELECT * FROM {src} LIMIT 0").df().columns)
    if not {"lactate", "any_pressor"} <= cols: sys.exit(f"sofa_hourly lacks lactate/any_pressor (found {sorted(cols)}).")
    dens = con.execute(f"SELECT AVG(CASE WHEN lactate IS NOT NULL THEN 1.0 ELSE 0 END) FROM {src}").fetchone()[0]
    say(f"[MIMIC hourly] share of stay-hours with a lactate value: {100 * dens:.1f}% (near 100% would mean forward-filled lactate: the audit would then be wrong)")
    if not a.no_outcome_check:
        sys.path.insert(0, a.proj); import outcome_definitions as od
        say("[MIMIC hourly] grid gaps:", od.grid_gaps(con, src, True))
        onset = od.onset_frame(con, src, "primary", True)
        rl = od.relabel(fm[["stay_id", "shock", "hours_to_shock"]], coh.set_index("stay_id").index_time, onset, "primary-check", log=say)
        m = fm[["stay_id", "shock", "hours_to_shock"]].merge(rl, on="stay_id", suffixes=("", "_re"), how="left")
        ok = (m.shock == m.shock_re.fillna(-1)).all() and np.allclose(m.hours_to_shock.fillna(-1), m.hours_to_shock_re.fillna(-1), atol=1e-3)
        say("[MIMIC outcome check]", "PASSED" if ok else "FAILED")
        if not ok: sys.exit("primary outcome re-derived from sofa_hourly does not reproduce the feature table; stop.")
    con.execute(f"""CREATE TABLE h AS SELECT s.stay_id, s.hr, s.lactate, s.any_pressor,
        COUNT(s.lactate) OVER (PARTITION BY s.stay_id ORDER BY s.hr ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING) AS n_lac_win
        FROM {src} s JOIN coh USING (stay_id)""")
    return con.execute("""SELECT h.stay_id,
        MAX(CASE WHEN h.any_pressor = 1 AND h.hr > coh.index_time THEN 1 ELSE 0 END) AS press_after,
        MAX(CASE WHEN h.any_pressor = 1 AND h.hr > coh.index_time AND h.n_lac_win > 0 THEN 1 ELSE 0 END) AS press_with_lac,
        MAX(CASE WHEN h.hr > coh.index_time AND h.hr <= coh.index_time + INTERVAL '24 hours' AND h.lactate IS NOT NULL THEN 1 ELSE 0 END) AS lac_24h
        FROM h JOIN coh USING (stay_id) GROUP BY h.stay_id""").df()


def scan_eicu():
    con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=2")
    con.register("ei", fe[["stay_id", "index_hr"]])
    src = f"read_parquet('{OUT}/sofa_hourly_eicu.parquet')"
    cols = set(con.execute(f"SELECT * FROM {src} LIMIT 0").df().columns)
    if not {"lactate", "any_pressor"} <= cols: sys.exit(f"sofa_hourly_eicu lacks lactate/any_pressor (found {sorted(cols)}).")
    dens = con.execute(f"SELECT AVG(CASE WHEN lactate IS NOT NULL THEN 1.0 ELSE 0 END) FROM {src}").fetchone()[0]
    say(f"[eICU hourly] share of stay-hours with a lactate value: {100 * dens:.1f}%")
    if not a.no_outcome_check:
        sys.path.insert(0, a.proj); import outcome_definitions as od
        say("[eICU hourly] grid gaps:", od.grid_gaps(con, src, False))
        onset = od.onset_frame(con, src, "primary", False)
        rl = od.relabel(fe[["stay_id", "shock", "hours_to_shock"]], fe.set_index("stay_id").index_hr, onset, "primary-check", log=say)
        m = fe[["stay_id", "shock", "hours_to_shock"]].merge(rl, on="stay_id", suffixes=("", "_re"), how="left")
        ok = (m.shock == m.shock_re.fillna(-1)).all() and np.allclose(m.hours_to_shock.fillna(-1), m.hours_to_shock_re.fillna(-1), atol=1e-3)
        say("[eICU outcome check]", "PASSED" if ok else "FAILED")
        if not ok: sys.exit("primary outcome re-derived from sofa_hourly_eicu does not reproduce the feature table; stop.")
    con.execute(f"""CREATE TABLE h AS SELECT s.stay_id, s.hr, s.lactate, s.any_pressor,
        COUNT(s.lactate) OVER (PARTITION BY s.stay_id ORDER BY s.hr ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING) AS n_lac_win
        FROM {src} s JOIN ei USING (stay_id)""")
    return con.execute("""SELECT h.stay_id,
        MAX(CASE WHEN h.any_pressor = 1 AND h.hr > ei.index_hr THEN 1 ELSE 0 END) AS press_after,
        MAX(CASE WHEN h.any_pressor = 1 AND h.hr > ei.index_hr AND h.n_lac_win > 0 THEN 1 ELSE 0 END) AS press_with_lac,
        MAX(CASE WHEN h.hr > ei.index_hr AND h.hr <= ei.index_hr + 24 AND h.lactate IS NOT NULL THEN 1 ELSE 0 END) AS lac_24h
        FROM h JOIN ei USING (stay_id) GROUP BY h.stay_id""").df()


sm = fm.merge(scan_mimic(), on="stay_id", how="left")
se = fe.merge(scan_eicu(), on="stay_id", how="left")
for d in (sm, se):
    for c in ("press_after", "press_with_lac", "lac_24h"): d[c] = d[c].fillna(0).astype(int)
    d["lac_pred"] = d["lactate_last"].notna().astype(int)
    d["unclass"] = ((d.press_after == 1) & (d.press_with_lac == 0) & (d.shock == 0)).astype(int)
say(f"[sanity] events without a lactate within +-6 h of a vasopressor hour (must be 0): MIMIC {int(((sm.shock == 1) & (sm.press_with_lac == 0)).sum())}, eICU {int(((se.shock == 1) & (se.press_with_lac == 0)).sum())}")


# ---------------------------------------------------------------- AUDIT 1
rows = []
for nm, d in (("MIMIC-IV, all cohort stays", sm), ("MIMIC-IV, temporal test", sm[~sm.anchor_year_group.isin(TRAIN_GROUPS)]), ("eICU-CRD", se)):
    ev, ne = d[d.shock == 1], d[d.shock == 0]
    un, ot = ne[ne.unclass == 1], ne[ne.unclass == 0]
    pc = lambda x, c: 100 * float(x[c].mean()) if len(x) else np.nan
    med = lambda x: float(x.sofa_total.median()) if len(x) and "sofa_total" in x else np.nan
    for lab, x in (("events (shock)", ev), ("non-events", ne), ("  non-events, vasopressor after index but no lactate within +-6 h (unclassifiable)", un),
                   ("  non-events, all others", ot)):
        rows.append({"dataset": nm, "group": lab, "n": len(x), "pct_lactate_before_prediction_time": pc(x, "lac_pred"),
                     "pct_lactate_in_first_24h_after_index": pc(x, "lac_24h"), "pct_vasopressor_after_index": pc(x, "press_after"),
                     "median_SOFA_at_prediction_time": med(x)})
    rows.append({"dataset": nm, "group": "SUMMARY: unclassifiable / all non-events (%)", "n": len(un), "pct_lactate_before_prediction_time": 100 * len(un) / max(len(ne), 1)})
    rows.append({"dataset": nm, "group": "SUMMARY: events if unclassifiable stays were events (relative increase, %)", "n": len(ev) + len(un),
                 "pct_lactate_before_prediction_time": 100 * len(un) / max(len(ev), 1)})
a1 = pd.DataFrame(rows); a1.to_csv(f"{WD}/audit1_lactate_by_outcome.csv", index=False)
say("\n=== AUDIT 1: lactate measurement by outcome ===")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90); pd.set_option("display.max_columns", 20)
say(a1.round(1).to_string(index=False))

# ---------------------------------------------------------------- AUDIT 2
say("\n=== AUDIT 2: calendar period of the MIMIC-IV temporal test set (evaluation only) ===")
pt = pd.read_parquet(f"{OUT}/results/private/predictions_temporal_test.parquet")
t = pt.merge(fm[["stay_id", "subject_id", "anchor_year_group", "sofa_total", "lactate_last", "on_pressor"]], on="stay_id", how="left")
pc_ = [c for c in t.columns if c.startswith("p_")]
rows = []
for g in ["all test"] + sorted(t.anchor_year_group.dropna().unique()):
    d = t if g == "all test" else t[t.anchor_year_group == g]
    r = {"period": g, "n": len(d), "events": int(d.shock.sum()), "prevalence_pct": 100 * d.shock.mean(),
         "mean_SOFA_at_prediction_time": float(d.sofa_total.mean()), "pct_on_vasopressor_at_prediction_time": 100 * float(d.on_pressor.mean())}
    for c in pc_: r[f"AUROC_{c[2:]}"] = fmt(auc_ci(d.shock.values, d[c].values, d.subject_id.values, seed=11))
    rows.append(r)
a2 = pd.DataFrame(rows); a2.to_csv(f"{WD}/audit2_calendar_period.csv", index=False)
say(a2[["period", "n", "events", "prevalence_pct", "mean_SOFA_at_prediction_time"] + [c for c in a2.columns if c in ("AUROC_RF", "AUROC_LightGBM", "AUROC_Core LR")]].round(2).to_string(index=False))

# ---------------------------------------------------------------- AUDIT 3
say("\n=== AUDIT 3: eICU-CRD hospitals ===")
hh = se.groupby("hospitalid").agg(stays=("stay_id", "size"), events=("shock", "sum"), pct_vasopressor_after_index=("press_after", lambda x: 100 * x.mean()),
                                  pct_lactate_first24h=("lac_24h", lambda x: 100 * x.mean()), pct_lactate_before_prediction=("lac_pred", lambda x: 100 * x.mean()),
                                  stays_with_vasopressor=("press_after", "sum"), vasopressor_stays_with_lactate_in_window=("press_with_lac", "sum")).reset_index()
hh["pct_vasopressor_stays_with_lactate_in_window"] = 100 * hh.vasopressor_stays_with_lactate_in_window / hh.stays_with_vasopressor.replace(0, np.nan)
hh.to_csv(f"{WD}/audit3_eicu_hospitals.csv", index=False)
hh["group"] = np.where(hh.events == 0, "hospitals with 0 events", "hospitals with >=1 event")
sm_ = hh.groupby("group").agg(hospitals=("hospitalid", "size"), median_stays=("stays", "median"), stays_total=("stays", "sum"),
                              median_pct_vasopressor=("pct_vasopressor_after_index", "median"), median_pct_lactate_first24h=("pct_lactate_first24h", "median"),
                              median_pct_vasopressor_stays_with_lactate=("pct_vasopressor_stays_with_lactate_in_window", "median")).reset_index()
sm_.to_csv(f"{WD}/audit3_eicu_hospital_summary.csv", index=False); say(sm_.round(1).to_string(index=False))
pe = pd.read_parquet(f"{OUT}/results/private/predictions_eicu.parquet")
e = pe.drop(columns=[c for c in ["hospitalid"] if c in pe.columns]).merge(se[["stay_id", "hospitalid"]], on="stay_id").merge(hh[["hospitalid", "events", "pct_vasopressor_after_index", "pct_lactate_first24h"]], on="hospitalid")
med_v, med_l = hh.pct_vasopressor_after_index.median(), hh.pct_lactate_first24h.median()
subsets = {"all hospitals": np.ones(len(e), bool), "hospitals with >=1 event": (e.events >= 1).values,
           "vasopressor recording >= median hospital": (e.pct_vasopressor_after_index >= med_v).values,
           "lactate recording (first 24 h) >= median hospital": (e.pct_lactate_first24h >= med_l).values,
           "both >= median": ((e.pct_vasopressor_after_index >= med_v) & (e.pct_lactate_first24h >= med_l)).values}
pcol = "p_RF" if "p_RF" in e.columns else [c for c in e.columns if c.startswith("p_")][0]
rows = []
for k, m in subsets.items():
    d = e[m]; r = auc_ci(d.shock.values, d[pcol].values, d.hospitalid.values, seed=21)
    rows.append({"subset": k, "hospitals": int(d.hospitalid.nunique()), "stays": len(d), "events": int(d.shock.sum()), "prevalence_pct": 100 * d.shock.mean(), "AUROC_RF": fmt(r)})
a3 = pd.DataFrame(rows); a3.to_csv(f"{WD}/audit3_eicu_subset_auroc.csv", index=False)
say(f"(medians used: vasopressor recording {med_v:.1f}%, lactate recording {med_l:.1f}%)"); say(a3.round(2).to_string(index=False))
say(f"\nDone. Aggregated outputs in {WD}"); LOG.close()
