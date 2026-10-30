#!/usr/bin/env python3
"""
Sensitivity analysis: does the cohort definition use information that was not yet available at the prediction time?

Question
  The suspected-infection criterion is fulfilled by an antibiotic AND a culture (MIMIC-IV: culture up to 72 h after
  the antibiotic, or antibiotic up to 24 h after the culture; eICU-CRD: antibiotic AND (culture OR sepsis diagnosis),
  all within -48 h .. +24 h of ICU admission).  A stay can therefore enter the cohort because of a culture or a
  diagnosis recorded AFTER the prediction time (index time + 1 h).  This script re-evaluates the SAVED primary
  predictions (no retraining, no new model) in the subset of stays whose infection criterion was already fulfilled
  at the prediction time, and in the complement.

Definitions (computed from the raw tables with the same SQL logic as the notebook)
  MIMIC-IV
    pair      = (antibiotic, culture) in the same admission that satisfies the Sepsis-3 window rule
    confirmed = max(antibiotic time, culture time) of a pair
    S1 (primary)   : at least one qualifying pair with confirmed <= prediction time
    S2 (stricter)  : the pair(s) that DEFINE the suspected-infection time (def_time = si_time) have confirmed <= prediction time
  eICU-CRD
    S1 (primary)   : antibiotic start <= prediction time AND (a culture taken in [-48 h, prediction time]
                     OR a sepsis diagnosis entered at or before the prediction time)
    (S2 is not defined: eICU has no antibiotic-culture pairing)

Inputs  (all already in your Drive from the notebook / extended_analysis_v3.py)
  --out        e.g. /content/drive/MyDrive/sepsis_project   (cohort_final_mimic.parquet, features_mimic.parquet,
                                                             features_eicu.parquet)
  --mimic_base e.g. /content/mimic/mimiciv                  (raw MIMIC-IV: Hosp/prescriptions, Hosp/microbiologyevents)
  --eicu_src   e.g. /content/eicu/eicu-collaborative-research-database-2.0/eicu-collaborative-research-database-2.0
  --preds_dir  folder with predictions_temporal_test.parquet and predictions_eicu.parquet
               (written by extended_analysis_v3.py --save_preds; they are PRIVATE, never upload them)
Outputs (aggregated only; safe to share) in --out/sensitivity:
  future_info_flags_summary.csv, future_info_sensitivity.csv, future_info_log.txt
"""
import argparse, os, sys, re
import numpy as np, pandas as pd
import duckdb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

ABX_RX = ("cillin|cef|penem|floxacin|mycin|cycline|vancomycin|azithro|metronidazole|sulfameth|trimethoprim|linezolid|"
          "daptomycin|aztreonam|colistin|polymyxin|gentamicin|tobramycin|amikacin|tazobactam|clavulan|sulbactam|"
          "rifampin|tigecycline")


# ----------------------------------------------------------------------------------------------------------------
# flag builders (pure SQL on tables that the caller creates; also used by the synthetic self-test)
# ----------------------------------------------------------------------------------------------------------------
def mimic_flags(con):
    """needs tables: abx(hadm_id, abx_time), cx(hadm_id, cx_time), cf(stay_id, hadm_id, si_time, pred_time)"""
    return con.execute("""
    WITH pairs AS (
      SELECT a.hadm_id, a.abx_time, c.cx_time,
        CASE WHEN a.abx_time <= c.cx_time AND c.cx_time <= a.abx_time + INTERVAL '72 hours' THEN a.abx_time
             WHEN c.cx_time  <  a.abx_time AND a.abx_time <= c.cx_time + INTERVAL '24 hours' THEN c.cx_time END AS def_time
      FROM abx a JOIN cx c USING (hadm_id)
    ), q AS (
      SELECT *, GREATEST(abx_time, cx_time) AS confirmed FROM pairs WHERE def_time IS NOT NULL
    )
    SELECT f.stay_id,
           COALESCE(bool_or(q.confirmed <= f.pred_time), FALSE)                         AS s1_mimic,
           COALESCE(bool_or(q.def_time = f.si_time AND q.confirmed <= f.pred_time), FALSE) AS s2_mimic,
           MIN(q.confirmed) AS first_confirmed, COUNT(q.hadm_id) AS n_pairs
    FROM cf f LEFT JOIN q ON q.hadm_id = f.hadm_id
    GROUP BY f.stay_id
    """).df()


def eicu_flags(con):
    """needs tables: abx(stay_id, abx_t), cx(stay_id, cx_t), dx(stay_id, dx_t), cf(stay_id, pred_min); all times in minutes"""
    return con.execute("""
    SELECT f.stay_id,
      COALESCE( (SELECT MIN(abx_t) FROM abx a WHERE a.stay_id=f.stay_id) <= f.pred_min, FALSE)
      AND ( EXISTS (SELECT 1 FROM cx c WHERE c.stay_id=f.stay_id AND c.cx_t BETWEEN -48*60 AND f.pred_min)
         OR EXISTS (SELECT 1 FROM dx d WHERE d.stay_id=f.stay_id AND d.dx_t <= f.pred_min) ) AS s1_eicu
    FROM cf f
    """).df()


# ----------------------------------------------------------------------------------------------------------------
# cluster bootstrap
# ----------------------------------------------------------------------------------------------------------------
def fast_auc(y, p):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cluster_boot_auc(y, p, cl, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(cl)
    order = np.argsort(codes, kind="stable")
    y, p, codes = y[order], p[order], codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]; ends = np.r_[starts[1:], len(codes)]
    k = len(starts); out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, k, k)
        idx = np.concatenate([np.arange(starts[j], ends[j]) for j in pick])
        out[b] = fast_auc(y[idx], p[idx])
    return np.nanpercentile(out, [2.5, 97.5])


def evaluate(df, pcols, mask, label, dataset, cluster, flag, rows, B):
    sub = df[mask]
    y = sub.shock.values.astype(int)
    for pc in pcols:
        p = sub[pc].values.astype(float)
        if y.sum() < 10 or (len(y) - y.sum()) < 10:
            rows.append(dict(dataset=dataset, flag=flag, subset=label, model=pc[2:], n=len(sub), events=int(y.sum()),
                             auroc=np.nan, lo=np.nan, hi=np.nan, auprc=np.nan, note="too few events/non-events")); continue
        lo, hi = cluster_boot_auc(y, p, sub[cluster].values, B=B)
        rows.append(dict(dataset=dataset, flag=flag, subset=label, model=pc[2:], n=len(sub), events=int(y.sum()),
                         prevalence=y.mean(), auroc=roc_auc_score(y, p), lo=lo, hi=hi,
                         auprc=average_precision_score(y, p), note=""))


# ----------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--mimic_base", required=True)
    ap.add_argument("--eicu_src", required=True); ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    SD = os.path.join(a.out, "sensitivity"); os.makedirs(SD, exist_ok=True)
    logf = open(os.path.join(SD, "future_info_log.txt"), "w")
    def say(*s):
        m = " ".join(str(x) for x in s); print(m); logf.write(m + "\n"); logf.flush()

    con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=2")
    opts = "header=true, strict_mode=false, ignore_errors=true"

    # ---------------- load saved predictions and check that they reproduce the manuscript ----------------
    pt = pd.read_parquet(os.path.join(a.preds_dir, "predictions_temporal_test.parquet"))
    pe = pd.read_parquet(os.path.join(a.preds_dir, "predictions_eicu.parquet"))
    r_t, r_e = roc_auc_score(pt.shock, pt["p_RF"]), roc_auc_score(pe.shock, pe["p_RF"])
    ok = abs(r_t - 0.778) <= 0.005 and abs(r_e - 0.783) <= 0.005
    say(f"REPRODUCIBILITY: RF AUROC temporal {r_t:.3f} (manuscript 0.778), eICU {r_e:.3f} (0.783) -> {'OK' if ok else 'MISMATCH - STOP'}")
    if not ok: sys.exit("Saved predictions do not reproduce the primary result; do not use this run.")
    pcols_t = [c for c in pt.columns if c.startswith("p_")]; pcols_e = [c for c in pe.columns if c.startswith("p_")]

    # ---------------- MIMIC-IV ----------------
    P = lambda folder, name: f"{a.mimic_base}/{folder}/{name}.csv/{name}.csv.gz"
    con.execute(f"""CREATE TABLE abx AS SELECT hadm_id, starttime AS abx_time FROM read_csv('{P('Hosp','prescriptions')}', header=true,
        types={{'starttime':'TIMESTAMP','stoptime':'TIMESTAMP'}}, ignore_errors=true)
        WHERE hadm_id IS NOT NULL AND starttime IS NOT NULL AND regexp_matches(lower(drug), '{ABX_RX}')
        AND NOT regexp_matches(lower(drug), 'bleomycin|mitomycin|dactinomycin|rapamycin')
        AND coalesce(route,'') NOT IN ('OU','OS','OD','AU','AS','AD','TP','IH')""")
    con.execute(f"""CREATE TABLE cx AS SELECT DISTINCT hadm_id, micro_specimen_id,
        coalesce(charttime, CAST(chartdate AS TIMESTAMP)) AS cx_time
        FROM read_csv('{P('Hosp','microbiologyevents')}', header=true, types={{'charttime':'TIMESTAMP','chartdate':'DATE'}}, ignore_errors=true)
        WHERE hadm_id IS NOT NULL""")
    con.execute(f"""CREATE TABLE cf AS SELECT stay_id, hadm_id, si_time, index_time + INTERVAL '1 hour' AS pred_time
        FROM read_parquet('{a.out}/cohort_final_mimic.parquet')""")
    fm = mimic_flags(con)
    # consistency check: every cohort stay must have >= 1 qualifying pair (otherwise the SQL does not match the cohort)
    n_nopair = int((fm.n_pairs == 0).sum())
    say(f"MIMIC cohort stays {len(fm):,}; stays with no qualifying pair (must be 0): {n_nopair}")
    if n_nopair: sys.exit("MIMIC flag logic does not reproduce the cohort definition; stop.")
    feat_m = pd.read_parquet(f"{a.out}/features_mimic.parquet", columns=["stay_id", "subject_id"])
    dm = pt.merge(fm, on="stay_id", how="left").merge(feat_m, on="stay_id", how="left")
    assert dm.s1_mimic.notna().all(), "test stays without flags"
    dm[["s1_mimic", "s2_mimic"]] = dm[["s1_mimic", "s2_mimic"]].astype(bool)

    # ---------------- eICU-CRD ----------------
    F = lambda n: f"{a.eicu_src}/{n}.csv.gz"
    con.execute(f"""CREATE TABLE e_abx AS SELECT patientunitstayid AS stay_id, drugstartoffset AS abx_t
        FROM read_csv('{F('medication')}', {opts}) WHERE drugstartoffset BETWEEN -48*60 AND 24*60
        AND coalesce(drugordercancelled,'No') <> 'Yes' AND regexp_matches(lower(drugname), '{ABX_RX}')
        AND NOT regexp_matches(lower(drugname), 'bleomycin|mitomycin|dactinomycin|rapamycin')
        AND NOT regexp_matches(lower(coalesce(routeadmin,'')), 'ophth|opht|topical|inhal|otic|ear|eye')""")
    con.execute(f"""CREATE TABLE e_cx AS SELECT DISTINCT patientunitstayid AS stay_id, culturetakenoffset AS cx_t
        FROM read_csv('{F('microLab')}', {opts}) WHERE culturetakenoffset BETWEEN -48*60 AND 24*60""")
    con.execute(f"""CREATE TABLE e_dx AS SELECT DISTINCT patientunitstayid AS stay_id, diagnosisoffset AS dx_t
        FROM read_csv('{F('diagnosis')}', {opts}, types={{'icd9code':'VARCHAR'}})
        WHERE diagnosisoffset <= 24*60 AND (
             (lower(diagnosisstring) LIKE '%sepsis%' AND lower(diagnosisstring) NOT LIKE '%shock%')
          OR regexp_matches(coalesce(icd9code,''), '(^|, )(038|995\\.91)') OR regexp_matches(coalesce(icd9code,''), 'A40|A41'))""")
    fe_raw = pd.read_parquet(f"{a.out}/features_eicu.parquet", columns=["stay_id", "hours_icu_to_pred"])
    con2 = duckdb.connect(); con2.execute("SET memory_limit='8GB'")
    for t in ["e_abx", "e_cx", "e_dx"]:
        con2.register(t + "_df", con.execute(f"SELECT * FROM {t}").df())
    con2.execute("CREATE TABLE abx AS SELECT * FROM e_abx_df"); con2.execute("CREATE TABLE cx AS SELECT * FROM e_cx_df")
    con2.execute("CREATE TABLE dx AS SELECT * FROM e_dx_df")
    fe_raw["pred_min"] = fe_raw.hours_icu_to_pred * 60
    con2.register("cf_df", fe_raw[["stay_id", "pred_min"]]); con2.execute("CREATE TABLE cf AS SELECT * FROM cf_df")
    fe = eicu_flags(con2)
    de = pe.merge(fe, on="stay_id", how="left")
    assert de.s1_eicu.notna().all(), "eICU stays without flags"
    de["s1_eicu"] = de.s1_eicu.astype(bool)

    # ---------------- summaries ----------------
    summ = []
    for ds, d, flags in [("MIMIC-IV temporal test", dm, ["s1_mimic", "s2_mimic"]), ("eICU-CRD", de, ["s1_eicu"])]:
        for f in flags:
            summ.append(dict(dataset=ds, flag=f, n=len(d), n_fulfilled_by_pred_time=int(d[f].sum()),
                             pct_fulfilled=100 * d[f].mean(), events=int(d.shock.sum()),
                             events_fulfilled=int(d.loc[d[f], "shock"].sum()),
                             pct_events_fulfilled=100 * d.loc[d[f], "shock"].sum() / d.shock.sum()))
    pd.DataFrame(summ).round(3).to_csv(f"{SD}/future_info_flags_summary.csv", index=False)
    say(pd.DataFrame(summ).round(3).to_string())

    rows = []
    for ds, d, pcols, cl, flags in [("MIMIC-IV temporal test", dm, pcols_t, "subject_id", ["s1_mimic", "s2_mimic"]),
                                     ("eICU-CRD", de, pcols_e, "hospitalid", ["s1_eicu"])]:
        evaluate(d, pcols, np.ones(len(d), bool), "all stays (primary)", ds, cl, "none", rows, a.boot)
        for f in flags:
            evaluate(d, pcols, d[f].values, "criterion fulfilled by prediction time", ds, cl, f, rows, a.boot)
            evaluate(d, pcols, ~d[f].values, "criterion NOT yet fulfilled at prediction time", ds, cl, f, rows, a.boot)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(f"{SD}/future_info_sensitivity.csv", index=False)
    say(res[res.model.isin(["RF", "Lactate only", "SOFA only"])][["dataset", "flag", "subset", "model", "n", "events", "auroc", "lo", "hi"]]
        .round(3).to_string(index=False))
    say("DONE. Share only the three files in", SD)


if __name__ == "__main__":
    main()
