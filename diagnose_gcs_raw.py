#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_gcs_raw.py - decisive check on WHY gcs_min is missing for 25% of the MIMIC cohort.
The hourly GCS in the notebook is eye+verbal+motor and exists only if all three were charted in the same clock hour.
For every stay we count, inside the same window the SOFA code uses (hour of index minus 23 h ... end of index hour),
how many charted values exist for eye (220739), verbal (223900), motor (223901) and whether ANY hour has all three.
Patterns of the GCS-missing stays:  nothing charted | eye+motor but no verbal | components never in the same hour | other.
If most missing stays have eye+motor but no verbal (typically intubated patients), the missingness is a CONSTRUCTION artefact
that can be fixed; if nothing is charted, it is a documentation gap.  Aggregated outputs only (--out/diagnostics).
"""
import argparse, os, numpy as np, pandas as pd, duckdb
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/content/drive/MyDrive/sepsis_project"); a = ap.parse_args()
D = os.path.join(a.out, "diagnostics"); os.makedirs(D, exist_ok=True)
con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=2"); R = lambda f: f"read_parquet('{a.out}/{f}.parquet')"
con.execute(f"""CREATE TABLE cf AS SELECT f.stay_id, f.gcs_min IS NULL AS gcs_missing, f.shock, f.hours_to_shock, f.pf_min IS NOT NULL AS pf_present, c.index_time
                FROM {R('features_mimic')} f JOIN {R('cohort_final_mimic')} c USING (stay_id)""")
con.execute(f"""CREATE TABLE g AS
  SELECT cf.stay_id, date_trunc('hour', ch.charttime) AS hr, ch.itemid, ch.valuenum
  FROM {R('chart_sub')} ch JOIN cf USING (stay_id)
  WHERE ch.itemid IN (220739, 223900, 223901) AND ch.valuenum IS NOT NULL AND date_trunc('hour', ch.charttime) BETWEEN cf.index_time - INTERVAL '23 hours' AND cf.index_time""")
con.execute("""CREATE TABLE per_hour AS SELECT stay_id, hr, COUNT(DISTINCT itemid) AS k FROM g GROUP BY 1,2""")
con.execute("""CREATE TABLE p AS
  SELECT cf.stay_id, cf.gcs_missing, cf.shock, cf.pf_present,
    COALESCE(SUM(CASE WHEN g.itemid=220739 THEN 1 END),0) AS n_eye, COALESCE(SUM(CASE WHEN g.itemid=223900 THEN 1 END),0) AS n_verbal,
    COALESCE(SUM(CASE WHEN g.itemid=223901 THEN 1 END),0) AS n_motor,
    COALESCE((SELECT MAX(k) FROM per_hour h WHERE h.stay_id=cf.stay_id),0) AS max_components_in_one_hour
  FROM cf LEFT JOIN g USING (stay_id) GROUP BY ALL""")
d = con.execute("SELECT * FROM p").df()
def pat(r):
    if r.n_eye + r.n_verbal + r.n_motor == 0: return "nothing charted"
    if r.max_components_in_one_hour == 3: return "all 3 in one hour (should not be missing)"
    if r.n_verbal == 0 and r.n_eye > 0 and r.n_motor > 0: return "eye+motor charted, verbal never charted"
    if r.n_verbal > 0 and r.n_eye > 0 and r.n_motor > 0: return "all 3 charted but never in the same hour"
    return "other partial"
d["pattern"] = d.apply(pat, axis=1)
t = d.groupby(["gcs_missing", "pattern"]).agg(n=("stay_id", "size"), event_rate=("shock", "mean"), pf_min_present_pct=("pf_present", lambda x: 100 * x.mean())).round(3).reset_index()
t["share_within_group_%"] = (100 * t.n / t.groupby("gcs_missing").n.transform("sum")).round(1)
t.to_csv(f"{D}/gcs_raw_patterns.csv", index=False)
pd.set_option("display.width", 200); print(t.to_string(index=False)); print(f"\nSaved to {D}")
