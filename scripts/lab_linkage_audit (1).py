#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lab_linkage_audit.py  -  audit of how MIMIC-IV laboratory measurements were linked to ICU stays.

The extraction links labevents to a stay by subject_id and the time window [intime - 24 h, outtime] (hourly SOFA, outcome) and
[prediction time - 24 h, prediction time) (predictors), not by hadm_id.  A laboratory row can therefore belong to
  same   the hadm_id of the stay (intended),
  null   no hadm_id (ED or outpatient laboratory, or not linked in MIMIC-IV), or
  other  a DIFFERENT hospital admission of the same patient (cross-admission transfer; should not be used).
The audit counts the rows and stays in each category for the two windows, and for the two laboratory predictors that matter most
(last lactate and last creatinine) it counts the stays whose predictor value would CHANGE if rows from another admission were removed.
Evaluation / counting only; aggregated outputs, no patient rows.  Reads: lab_sub, cohort_mimic, cohort_final_mimic, features_mimic.
Writes: <out>/revision_audits/lab_linkage_*.csv and lab_linkage_log.txt
"""
import argparse, os, sys
import numpy as np, pandas as pd, duckdb

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/content/drive/MyDrive/sepsis_project")
a = ap.parse_args()
OUT = a.out; WD = os.path.join(OUT, "revision_audits"); os.makedirs(WD, exist_ok=True)
LOG = open(os.path.join(WD, "lab_linkage_log.txt"), "w")
def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

coh = pd.read_parquet(f"{OUT}/cohort_mimic.parquet")
need = {"stay_id", "subject_id", "hadm_id", "intime", "outtime"}
if not need <= set(coh.columns): sys.exit(f"cohort_mimic lacks {sorted(need - set(coh.columns))}; found {sorted(coh.columns)}")
fin = pd.read_parquet(f"{OUT}/cohort_final_mimic.parquet")[["stay_id", "index_time"]]
c = coh[list(need)].merge(fin, on="stay_id")
c["pred_time"] = pd.to_datetime(c["index_time"]) + pd.Timedelta(hours=1)
c["intime"] = pd.to_datetime(c["intime"]); c["outtime"] = pd.to_datetime(c["outtime"])
say(f"analysed MIMIC-IV stays: {len(c):,}")
con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=2")
con.register("c", c)
con.execute(f"""CREATE TABLE j AS
  SELECT c.stay_id, c.hadm_id AS stay_hadm, l.hadm_id AS lab_hadm, l.itemid, l.charttime, l.valuenum,
         CASE WHEN l.hadm_id IS NULL THEN 'null' WHEN l.hadm_id = c.hadm_id THEN 'same' ELSE 'other' END AS cat,
         (l.charttime >= c.intime - INTERVAL '24 hours' AND l.charttime <= c.outtime) AS w_hourly,
         (l.charttime >= c.pred_time - INTERVAL '24 hours' AND l.charttime < c.pred_time) AS w_pred
  FROM read_parquet('{OUT}/lab_sub.parquet') l JOIN c ON l.subject_id = c.subject_id
  WHERE l.charttime >= c.intime - INTERVAL '24 hours' AND l.charttime <= c.outtime""")
tot = con.execute("SELECT COUNT(*) FROM j").fetchone()[0]
say(f"laboratory rows linked to the analysed stays by subject_id and time window: {tot:,}")

rows = []
for wname, col in (("hourly SOFA / outcome window [intime-24h, outtime]", "w_hourly"), ("predictor window [pred_time-24h, pred_time)", "w_pred")):
    d = con.execute(f"""SELECT cat, COUNT(*) AS rows, COUNT(DISTINCT stay_id) AS stays FROM j WHERE {col} GROUP BY cat""").df().set_index("cat")
    n = int(d["rows"].sum())
    for cat in ("same", "null", "other"):
        r = d.loc[cat] if cat in d.index else pd.Series({"rows": 0, "stays": 0})
        rows.append({"window": wname, "category": cat, "rows": int(r["rows"]), "pct_of_rows": 100 * r["rows"] / max(n, 1),
                     "stays_with_at_least_one": int(r["stays"]), "pct_of_stays": 100 * r["stays"] / len(c)})
a1 = pd.DataFrame(rows); a1.to_csv(f"{WD}/lab_linkage_by_category.csv", index=False)
say("\n=== rows by hadm_id category ===\n" + a1.round(3).to_string(index=False))

# per item in the predictor window
items = {50813: "lactate", 50912: "creatinine"}
d = con.execute("""SELECT itemid, cat, COUNT(*) AS rows, COUNT(DISTINCT stay_id) AS stays FROM j WHERE w_pred AND itemid IN (50813, 50912)
                   GROUP BY itemid, cat ORDER BY itemid, cat""").df()
d["item"] = d.itemid.map(items); d.drop(columns="itemid").to_csv(f"{WD}/lab_linkage_predictor_items.csv", index=False)

# does removing rows from another admission change the last-value predictors?
fm = pd.read_parquet(f"{OUT}/features_mimic.parquet")
out = []
for iid, nm in items.items():
    col = f"{nm if nm != 'creatinine' else 'creat'}_last"
    if col not in fm.columns: say(f"column {col} not in features_mimic; skipped"); continue
    q = lambda cond: con.execute(f"""SELECT stay_id, arg_max(valuenum, charttime) AS v FROM j WHERE w_pred AND itemid = {iid} {cond}
                                      AND valuenum IS NOT NULL GROUP BY stay_id""").df().set_index("stay_id").v
    allv, keep = q(""), q("AND cat <> 'other'")
    f = fm.set_index("stay_id")[col]
    rep = pd.concat([f, allv], axis=1, keys=["feat", "all"]).reindex(fm.stay_id)
    same_as_feat = (np.isclose(rep.feat, rep["all"], equal_nan=True)).mean()
    ch = pd.concat([allv, keep], axis=1, keys=["all", "keep"]).reindex(allv.index)
    changed = ~np.isclose(ch["all"], ch["keep"], equal_nan=True)
    lost = ch["keep"].isna() & ch["all"].notna()
    out.append({"predictor": col, "stays_with_value": int(allv.notna().sum()), "recomputed_equals_feature_table_pct": 100 * same_as_feat,
                "stays_whose_last_value_changes_without_other_admission_rows": int(changed.sum()),
                "of_which_value_becomes_missing": int(lost.sum()),
                "pct_of_all_stays": 100 * changed.sum() / len(fm)})
a2 = pd.DataFrame(out); a2.to_csv(f"{WD}/lab_linkage_effect_on_predictors.csv", index=False)
say("\n=== effect of removing rows from another admission on the last-value predictors ===\n" + a2.round(3).to_string(index=False))
say(f"\nDone. Aggregated outputs in {WD}"); LOG.close()
