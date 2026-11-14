#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
zhou_design_analysis.py
=======================
EMULATION (not replication) of the design of Zhou et al., Int J Med Inform 2026;206:106169, inside the cohort, outcome definition
and pipeline of this project, to measure how much of an apparent AUROC comes from the design itself.

What Zhou et al. did (from the paper): adults with sepsis; first ICU stay; ICU stay of 1-30 days; predictors = "the first recorded
measurement within 24 h of ICU admission" (vital signs, laboratory values, severity scores that include SOFA), outcome = septic shock
at ANY time during the ICU stay (no prediction time); training on MIMIC-IV, validation on eICU-CRD; random forest best (external AUC 0.785).

What is emulated here (all of it can be read in the code):
  cohort     Sepsis-3 stays of the project (sepsis3_mimic; eICU-CRD sepsis cohort re-derived with the notebook SQL), age 18-89, first ICU
             stay (MIMIC-IV: first stay in the suspected-infection extract cohort_mimic; eICU-CRD: unitvisitnumber = 1, already applied),
             ICU stay of 24 h to 30 days.  Stays that are already in shock at the index time are KEPT (they are excluded from the primary
             cohort, but Zhou et al. count every shock during the stay).
  window     [t0, t0 + 24 h) with t0 = hour of ICU admission; prediction time = t0 + 24 h.  Same window logic as ladder_analysis.py
             (so the builder can be checked against features_mimic / features_eicu, see "self-checks").
  predictors "first": first recorded value of each vital sign / laboratory variable inside the window (Zhou-literal) + first-24-h SOFA
             columns (the hourly SOFA table is a rolling 24-h worst value, read at hour t0 + 23) + age + sex.
             "agg":  mean / min / max / last of the window (the feature set of the primary analysis) + the same SOFA columns, age, sex.
  outcome    E1  septic shock at any time within --max_hours of t0 (Zhou-like; default 168 h = the horizon of the eICU-CRD hourly table).
                 Events with onset inside the predictor window (hour <= 23) are "early", the others "late".
             E2  the same stays minus those with an early event; outcome = late event (shock after the window = a real prediction).
  models     the project's random forest (500 trees, fixed) or LightGBM (--model), identical in every design; trained on MIMIC-IV and
             applied to eICU-CRD without retraining; MIMIC-IV also by 5-fold subject-grouped cross-validation (out-of-fold predictions).
  ablation   PL = without vasopressor-derived inputs (on_pressor, s_cv, sofa_total) and without lactate (--variants full,PL, the default);
             L  = without lactate only, SOFA kept: the closest to the inputs retained by the LASSO selection of Zhou et al. (their Fig. 2 has SOFA,
                  SAPS II and OASIS but no lactate).  Run  --variants full,L --tag L  to add it without overwriting the first run; the "full" rows
                  must then reproduce the first run exactly (same seeds), which the log reports.
  decomposition  the E1 model evaluated on (non-events + early events) and on (non-events + late events).
  statistics paired cluster bootstrap (subjects in MIMIC-IV, hospitals in eICU-CRD), same resamples in every contrast.

NOT emulated (not in the extract, or not specified in the paper): SAPS II, OASIS, Charlson, comorbidities, AKI status, invasive ventilation;
DNR / palliative / CCU / metastatic-cancer exclusions; the ">40 % missing key variables" exclusion; LASSO selection; their hyper-parameters.
The absolute AUROC is therefore NOT comparable with 0.785; only the contrasts between designs are interpretable.

Self-checks (the run stops if one fails):
  S1  eICU-CRD sepsis cohort: every stay of features_eicu is in the re-derived cohort and its index hour is reproduced exactly.
  S2  shock onset: for every stay of cohort_final_mimic / features_eicu that is also in the emulated cohort, `shock` and `hours_to_shock`
      are reproduced exactly from the hourly table.
  S3  features: for stays whose primary prediction time equals t0 + 24 h (hours_icu_to_pred = 24) the "agg" columns reproduce
      features_mimic / features_eicu.

Inputs (in --out):  sepsis3_mimic, cohort_mimic, cohort_final_mimic, features_mimic, features_eicu, cohort_eicu, chart_sub, lab_sub,
                    sofa_hourly, eicu_vitalp_sub, eicu_vitala_sub, eicu_lab_sub, sofa_hourly_eicu  (.parquet)
Outputs (in --out/zhou_emulation): zhou_{mimic,eicu}_cohort.parquet (PATIENT-LEVEL: keep private), zhou_results.csv,
                    zhou_paired_differences.csv, zhou_design_ns.csv, zhou_overlap_exposure.csv, zhou_log.txt
"""
import argparse, os, sys, warnings, platform
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

VARS = ["hr", "rr", "spo2", "sbp", "dbp", "map", "temp"]
M_LAB = {"creat": 50912, "bili": 50885, "plt": 51265, "lactate": 50813, "wbc": 51301, "hgb": 51222,
         "k": 50971, "na": 50983, "bicarb": 50882, "ph": 50820, "pao2": 50821, "base_excess": 50802}
E_LAB = {"creat": "creatinine", "bili": "total bilirubin", "plt": "platelets x 1000", "lactate": "lactate", "wbc": "wbc x 1000",
         "hgb": "hgb", "k": "potassium", "na": "sodium", "bicarb": "bicarbonate", "ph": "ph", "pao2": "pao2", "base_excess": "base excess"}
SOFA_COLS = ["sofa_total", "s_resp", "s_coag", "s_liver", "s_cv", "s_cns", "s_renal", "gcs_min", "pf_min", "on_pressor"]
SOFA_SEL = ("s.sofa AS sofa_total, s.s_resp, s.s_coag, s.s_liver, s.s_cv, s.s_cns, s.s_renal, s.gcs_min, s.pf_min, "
            "s.any_pressor AS on_pressor")
PRESS = ["on_pressor", "s_cv", "sofa_total"]                      # vasopressor-derived inputs (as in ladder_analysis.py)
LIM = {"hr": (0, 300), "rr": (0, 100), "spo2": (0, 100), "sbp": (0, 350), "dbp": (0, 250), "map": (0, 300), "temp": (25, 45)}
WIN = 24                                                          # predictor window (hours)
AGG_COLS = ([f"{x}_{s}" for x in VARS for s in ("mean", "min", "max", "last")] +
            [f"{x}_{s}" for x in M_LAB for s in ("last", "min", "max")] + SOFA_COLS)
FIRST_COLS = [f"{x}_first" for x in VARS] + [f"{x}_first" for x in M_LAB] + SOFA_COLS
STATIC = ["age", "male"]


def clean(d):                                                     # identical to the primary analysis
    d = d.copy()
    num = d.select_dtypes(include=[np.number]).columns
    d[num] = d[num].replace([np.inf, -np.inf], np.nan)
    for c in d.columns:
        for k, (lo, hi) in LIM.items():
            if c.startswith(k + "_"):
                d.loc[(d[c] < lo) | (d[c] > hi), c] = np.nan
    if "pf_min" in d.columns:
        d.loc[d["pf_min"] > 800, "pf_min"] = np.nan
    return d


def newcon():
    import duckdb
    con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=2"); return con


# ------------------------------------------------------------------ builders
SHOCK_SQL = """CREATE TABLE shock AS
SELECT stay_id, MIN({hr}) AS shock_hr FROM (
  SELECT stay_id, {hr}, any_pressor,
         MAX(lactate) OVER (PARTITION BY stay_id ORDER BY {hr} ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING) AS lac_win
  FROM sofa) WHERE any_pressor = 1 AND lac_win > 2 GROUP BY stay_id"""                       # identical to the notebook


def build_mimic_z(OUT, max_los_days=30):
    """one row per eligible stay: stay_id, subject_id, age, male, t0, hours_icu_to_pred, shock_h (hours from t0, NaN = none), features"""
    con = newcon(); R = lambda f: f"read_parquet('{OUT}/{f}.parquet')"
    con.execute(f"CREATE TABLE s3 AS SELECT * FROM {R('sepsis3_mimic')}")
    con.execute(f"CREATE TABLE sofa AS SELECT stay_id, hr, lactate, any_pressor FROM {R('sofa_hourly')}")
    con.execute(SHOCK_SQL.format(hr="hr"))
    con.execute(f"""CREATE TABLE firsts AS SELECT stay_id FROM (
        SELECT stay_id, row_number() OVER (PARTITION BY subject_id ORDER BY intime, stay_id) AS rn FROM {R('cohort_mimic')}) WHERE rn = 1""")
    flow = {"sepsis-3 stays": con.execute("SELECT COUNT(*) FROM s3").fetchone()[0],
            "of which in cohort_mimic": con.execute("SELECT COUNT(*) FROM s3 JOIN (SELECT DISTINCT stay_id FROM " + R('cohort_mimic') + ") USING (stay_id)").fetchone()[0]}
    if flow["of which in cohort_mimic"] != flow["sepsis-3 stays"]:
        raise RuntimeError("sepsis3_mimic contains stays that are not in cohort_mimic; the first-stay rule cannot be applied.")
    con.execute("CREATE TABLE z1 AS SELECT s.* FROM s3 s JOIN firsts USING (stay_id)"); flow["first ICU stay"] = con.execute("SELECT COUNT(*) FROM z1").fetchone()[0]
    con.execute("CREATE TABLE z2 AS SELECT * FROM z1 WHERE age BETWEEN 18 AND 89"); flow["age 18-89"] = con.execute("SELECT COUNT(*) FROM z2").fetchone()[0]
    con.execute(f"""CREATE TABLE zc AS SELECT z.stay_id, z.subject_id, z.age, (z.gender = 'M')::INT AS male, z.intime,
          date_trunc('hour', z.intime) AS t0, date_trunc('hour', z.intime) + INTERVAL '{WIN} hours' AS pred_time,
          date_diff('hour', date_trunc('hour', z.intime), k.shock_hr) AS shock_h
        FROM z2 z LEFT JOIN shock k USING (stay_id)
        WHERE date_diff('minute', z.intime, z.outtime) >= {WIN * 60} AND date_diff('minute', z.intime, z.outtime) <= {max_los_days * 1440}""")
    flow["ICU stay 24 h to 30 d (analysis set)"] = con.execute("SELECT COUNT(*) FROM zc").fetchone()[0]
    items = [220045, 220210, 220277, 220179, 220050, 220180, 220051, 220181, 220052, 225312, 223762, 223761]
    con.execute(f"""CREATE TABLE chart_long AS
        SELECT zc.stay_id, ch.charttime,
          CASE WHEN ch.itemid=220045 THEN 'hr' WHEN ch.itemid=220210 THEN 'rr' WHEN ch.itemid=220277 THEN 'spo2'
               WHEN ch.itemid IN (220179,220050) THEN 'sbp' WHEN ch.itemid IN (220180,220051) THEN 'dbp'
               WHEN ch.itemid IN (220181,220052,225312) THEN 'map' WHEN ch.itemid IN (223762,223761) THEN 'temp' END AS var,
          CASE WHEN ch.itemid=223761 THEN (ch.valuenum-32)/1.8 ELSE ch.valuenum END AS v
        FROM {R('chart_sub')} ch JOIN zc USING (stay_id)
        WHERE ch.itemid IN ({','.join(map(str, items))}) AND ch.charttime < zc.pred_time AND ch.charttime >= zc.pred_time - INTERVAL '{WIN} hours'""")
    aggs = []
    for x in VARS:
        aggs += [f"AVG(v) FILTER (WHERE var='{x}') AS {x}_mean", f"MIN(v) FILTER (WHERE var='{x}') AS {x}_min",
                 f"MAX(v) FILTER (WHERE var='{x}') AS {x}_max", f"arg_max(v, charttime) FILTER (WHERE var='{x}') AS {x}_last",
                 f"arg_min(v, charttime) FILTER (WHERE var='{x}') AS {x}_first"]
    con.execute(f"CREATE TABLE chart_feat AS SELECT stay_id, {', '.join(aggs)} FROM chart_long GROUP BY stay_id")
    case = "CASE " + " ".join(f"WHEN lb.itemid={i} THEN '{n}'" for n, i in M_LAB.items()) + " END"
    con.execute(f"""CREATE TABLE lab_long AS
        SELECT zc.stay_id, lb.charttime, {case} AS var, lb.valuenum AS v
        FROM {R('lab_sub')} lb JOIN zc ON zc.subject_id = lb.subject_id
        WHERE lb.itemid IN ({','.join(map(str, M_LAB.values()))}) AND lb.charttime < zc.pred_time AND lb.charttime >= zc.pred_time - INTERVAL '{WIN} hours'""")
    aggs = []
    for x in M_LAB:
        aggs += [f"arg_max(v, charttime) FILTER (WHERE var='{x}') AS {x}_last", f"MIN(v) FILTER (WHERE var='{x}') AS {x}_min",
                 f"MAX(v) FILTER (WHERE var='{x}') AS {x}_max", f"arg_min(v, charttime) FILTER (WHERE var='{x}') AS {x}_first"]
    con.execute(f"CREATE TABLE lab_feat AS SELECT stay_id, {', '.join(aggs)} FROM lab_long GROUP BY stay_id")
    con.execute(f"""CREATE TABLE sofa_idx AS SELECT zc.stay_id, {SOFA_SEL}
        FROM zc JOIN {R('sofa_hourly')} s ON s.stay_id = zc.stay_id AND s.hr = zc.pred_time - INTERVAL '1 hour'""")
    df = con.execute(f"""SELECT zc.stay_id, zc.subject_id, zc.age, zc.male, zc.t0, {WIN} AS hours_icu_to_pred_z, zc.shock_h,
                              date_diff('hour', zc.intime, zc.pred_time) AS hours_icu_to_pred,
                              ch.* EXCLUDE (stay_id), lb.* EXCLUDE (stay_id), so.* EXCLUDE (stay_id)
                       FROM zc LEFT JOIN chart_feat ch USING (stay_id) LEFT JOIN lab_feat lb USING (stay_id) LEFT JOIN sofa_idx so USING (stay_id)
                       ORDER BY zc.stay_id""").df()
    shock_all = con.execute("SELECT s.stay_id, date_diff('hour', date_trunc('hour', s.intime), k.shock_hr) AS shock_h, date_trunc('hour', s.intime) AS t0 "
                            "FROM s3 s LEFT JOIN shock k USING (stay_id)").df()
    return df, flow, shock_all


def build_eicu_z(OUT, max_los_days=30):
    con = newcon(); R = lambda f: f"read_parquet('{OUT}/{f}.parquet')"
    H = lambda c: f"CAST(floor({c}/60.0) AS INTEGER)"
    con.execute(f"CREATE TABLE cohort AS SELECT * FROM {R('cohort_eicu')}")
    con.execute(f"CREATE TABLE sofa AS SELECT stay_id, hr, lactate, any_pressor, sofa FROM {R('sofa_hourly_eicu')}")
    con.execute(f"""CREATE TABLE sepsis3 AS SELECT k.*, {H('k.si_t')} AS si_hr, MIN(s.hr) AS sofa2_hr
        FROM cohort k JOIN sofa s ON s.stay_id = k.stay_id AND s.hr BETWEEN {H('k.si_t')} - 48 AND {H('k.si_t')} + 24 AND s.sofa >= 2 GROUP BY ALL""")
    con.execute(SHOCK_SQL.format(hr="hr"))
    flow = {"suspected-infection first adult unit stays": con.execute("SELECT COUNT(*) FROM cohort").fetchone()[0],
            "sepsis-3 stays": con.execute("SELECT COUNT(*) FROM sepsis3").fetchone()[0]}
    con.execute("CREATE TABLE z2 AS SELECT * FROM sepsis3 WHERE age_num BETWEEN 18 AND 89"); flow["age 18-89"] = con.execute("SELECT COUNT(*) FROM z2").fetchone()[0]
    con.execute(f"""CREATE TABLE zc AS SELECT z.stay_id, z.hospitalid, z.age_num AS age, (z.gender = 'Male')::INT AS male,
          GREATEST(z.si_hr, z.sofa2_hr, 0) AS index_hr, k.shock_hr AS shock_h
        FROM z2 z LEFT JOIN shock k USING (stay_id)
        WHERE z.outoffset >= {WIN * 60} AND z.outoffset <= {max_los_days * 1440}""")
    flow["ICU stay 24 h to 30 d (analysis set)"] = con.execute("SELECT COUNT(*) FROM zc").fetchone()[0]
    con.execute(f"""CREATE TABLE vit AS
        SELECT stay_id, t, heartrate AS hrt, respiration AS rrt, sao2 AS spo2t, temperature AS tempt,
               systemicsystolic AS sbpt, systemicdiastolic AS dbpt, systemicmean AS mapt FROM {R('eicu_vitalp_sub')}
        UNION ALL
        SELECT stay_id, t, NULL, NULL, NULL, NULL, noninvasivesystolic, noninvasivediastolic, noninvasivemean FROM {R('eicu_vitala_sub')}""")
    vmap = {"hr": "hrt", "rr": "rrt", "spo2": "spo2t", "sbp": "sbpt", "dbp": "dbpt", "map": "mapt", "temp": "tempt"}
    aggs = []
    for x, c in vmap.items():
        aggs += [f"AVG({c}) AS {x}_mean", f"MIN({c}) AS {x}_min", f"MAX({c}) AS {x}_max",
                 f"arg_max({c}, t) FILTER (WHERE {c} IS NOT NULL) AS {x}_last", f"arg_min({c}, t) FILTER (WHERE {c} IS NOT NULL) AS {x}_first"]
    con.execute(f"""CREATE TABLE chart_feat AS SELECT zc.stay_id, {', '.join(aggs)}
        FROM vit JOIN zc ON zc.stay_id = vit.stay_id AND vit.t < {WIN * 60} AND vit.t >= 0 GROUP BY zc.stay_id""")
    con.execute(f"CREATE TABLE lab AS SELECT * FROM {R('eicu_lab_sub')}")
    aggs = []
    for x, nm in E_LAB.items():
        aggs += [f"arg_max(v, t) FILTER (WHERE labname='{nm}') AS {x}_last", f"MIN(v) FILTER (WHERE labname='{nm}') AS {x}_min",
                 f"MAX(v) FILTER (WHERE labname='{nm}') AS {x}_max", f"arg_min(v, t) FILTER (WHERE labname='{nm}') AS {x}_first"]
    con.execute(f"""CREATE TABLE lab_feat AS SELECT zc.stay_id, {', '.join(aggs)}
        FROM lab JOIN zc ON zc.stay_id = lab.stay_id AND lab.t < {WIN * 60} AND lab.t >= 0 GROUP BY zc.stay_id""")
    con.execute(f"""CREATE TABLE sofa_idx AS SELECT zc.stay_id, {SOFA_SEL}
        FROM zc JOIN {R('sofa_hourly_eicu')} s ON s.stay_id = zc.stay_id AND s.hr = {WIN - 1}""")
    df = con.execute(f"""SELECT zc.stay_id, zc.hospitalid, zc.age, zc.male, zc.index_hr, {WIN} AS hours_icu_to_pred_z, zc.shock_h,
                              ch.* EXCLUDE (stay_id), lb.* EXCLUDE (stay_id), so.* EXCLUDE (stay_id)
                       FROM zc LEFT JOIN chart_feat ch USING (stay_id) LEFT JOIN lab_feat lb USING (stay_id) LEFT JOIN sofa_idx so USING (stay_id)
                       ORDER BY zc.stay_id""").df()
    s3 = con.execute("SELECT stay_id, GREATEST(si_hr, sofa2_hr, 0) AS index_hr FROM sepsis3").df()
    shock_all = con.execute("SELECT s.stay_id, k.shock_hr AS shock_h FROM sepsis3 s LEFT JOIN shock k USING (stay_id)").df()
    return df, flow, s3, shock_all


# ------------------------------------------------------------------ self-checks
def check_features(tag, z, old, say):
    """S3: where the primary prediction time equals t0 + 24 h, the agg columns of the emulation equal the primary feature table"""
    old = old[old.hours_icu_to_pred == WIN]
    cols = [c for c in AGG_COLS if c in old.columns and c in z.columns]
    m = clean(old)[["stay_id"] + cols].merge(clean(z)[["stay_id"] + cols], on="stay_id", suffixes=("_old", "_new"))
    say(f"[{tag}] S3 feature check on {len(m):,} stays with primary prediction time = t0 + 24 h ({len(cols)} columns)")
    if len(m) == 0:
        raise RuntimeError(f"[{tag}] S3: no stay can be used to check the features (none with hours_icu_to_pred = 24 in the emulated cohort).")
    bad = []
    for c in cols:
        a, b = m[c + "_old"].astype(float), m[c + "_new"].astype(float)
        eq = (a.isna() & b.isna()) | ((a - b).abs() <= 1e-6 + 1e-6 * a.abs())
        tol = 0.02 if c.endswith("_last") else 5e-4          # arg_max may break timestamp ties differently
        if 1 - eq.mean() > tol: bad.append((c, round(float(1 - eq.mean()), 4)))
    if bad: raise RuntimeError(f"[{tag}] S3 FAILED: the emulation does not reproduce the primary features (column, mismatch fraction): {bad[:15]}")
    say(f"[{tag}] S3 PASSED")


def check_shock_mimic(z_all, cf, say):
    """S2 (MIMIC-IV): cohort_final_mimic shock / hours_to_shock reproduced from the hourly table"""
    m = cf[["stay_id", "index_time", "shock", "hours_to_shock"]].merge(z_all, on="stay_id", how="left")
    if m.t0.isna().any(): raise RuntimeError("S2 FAILED (MIMIC-IV): stays of cohort_final_mimic are missing from sepsis3_mimic.")
    index_rel = ((pd.to_datetime(m.index_time) - pd.to_datetime(m.t0)) / pd.Timedelta(hours=1)).round().astype(int)
    flag = m.shock_h.notna().astype(int)
    bad_flag = int((flag != m["shock"]).sum())
    ev = m["shock"] == 1
    bad_h = int(((m.shock_h - index_rel)[ev] != m.hours_to_shock[ev]).sum())
    say(f"[MIMIC] S2 shock onset check on {len(m):,} stays: flag mismatches {bad_flag}, hours_to_shock mismatches {bad_h} (among {int(ev.sum()):,} events)")
    if bad_flag or bad_h: raise RuntimeError("S2 FAILED (MIMIC-IV): shock onset is not reproduced from sofa_hourly.")
    say("[MIMIC] S2 PASSED")


def check_eicu(s3, shock_all, fe, say):
    """S1 + S2 (eICU-CRD)"""
    m = fe[["stay_id", "hours_icu_to_pred", "shock", "hours_to_shock"]].merge(s3, on="stay_id", how="left").merge(shock_all, on="stay_id", how="left")
    if m.index_hr.isna().any(): raise RuntimeError("S1 FAILED (eICU-CRD): stays of features_eicu are missing from the re-derived sepsis-3 cohort.")
    bad_idx = int((m.index_hr != m.hours_icu_to_pred - 1).sum())
    say(f"[eICU] S1 cohort check on {len(m):,} stays: index hour mismatches {bad_idx}")
    if bad_idx: raise RuntimeError("S1 FAILED (eICU-CRD): the re-derived sepsis-3 index hour differs from the primary cohort.")
    flag = m.shock_h.notna().astype(int); bad_flag = int((flag != m["shock"]).sum()); ev = m["shock"] == 1
    bad_h = int(((m.shock_h - m.index_hr)[ev] != m.hours_to_shock[ev]).sum())
    say(f"[eICU] S2 shock onset check: flag mismatches {bad_flag}, hours_to_shock mismatches {bad_h} (among {int(ev.sum()):,} events)")
    if bad_flag or bad_h: raise RuntimeError("S2 FAILED (eICU-CRD): shock onset is not reproduced.")
    say("[eICU] S1/S2 PASSED")


# ------------------------------------------------------------------ statistics
def make_boot(clusters, nb, seed):
    rng = np.random.default_rng(seed); codes = pd.factorize(pd.Series(clusters))[0]
    groups = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    return [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(nb)]


def metric(fn, y, p, mask, idx=None):
    """AUROC/AUPRC on the rows of `mask` (restricted to the resample `idx`); NaN if one class is missing"""
    if idx is None: sel = np.flatnonzero(mask)
    else: sel = idx[mask[idx]]
    yy = y[sel]
    return fn(yy, p[sel]) if 0 < yy.sum() < len(yy) else np.nan


def pct(v): v = v[~np.isnan(v)]; return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def fmt(pt, lo, hi): return f"{pt:.3f} ({lo:.3f}-{hi:.3f})"


def folds_by_group(groups, nf, seed):
    u = np.unique(groups); perm = np.random.default_rng(seed).permutation(len(u))
    fold_of = dict(zip(u[perm], np.arange(len(u)) % nf))
    return np.array([fold_of[g] for g in groups])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/drive/MyDrive/sepsis_project")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--rf_trees", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="RF", choices=["RF", "LightGBM"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max_hours", type=int, default=168, help="shock counts as an event only if it starts within this many hours of t0 "
                    "(168 h = the horizon of the eICU-CRD hourly table; the same rule is applied to MIMIC-IV)")
    ap.add_argument("--predictors", default="first,agg")
    ap.add_argument("--variants", default="full,PL", help="input sets: full; PL = without vasopressor-derived inputs and lactate; "
                    "L = without lactate only (SOFA and its cardiovascular component stay); must contain full")
    ap.add_argument("--tag", default="", help="suffix of the result files (zhou_results_<tag>.csv ...), so that a second run does not overwrite the first")
    ap.add_argument("--only_build", action="store_true", help="build the emulated cohorts + run the self-checks, then stop")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    OUT = args.out; ZD = os.path.join(OUT, "zhou_emulation"); os.makedirs(ZD, exist_ok=True)
    TG = ("_" + args.tag) if args.tag else ""
    LOG = open(os.path.join(ZD, f"zhou_log{TG}.txt"), "w")

    def say(*a):
        s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()
    import duckdb
    say("python", platform.python_version(), "| duckdb", duckdb.__version__, "| pandas", pd.__version__, "| numpy", np.__version__)
    say(f"model={args.model} nboot={args.nboot} rf_trees={args.rf_trees} folds={args.folds} max_hours={args.max_hours} predictors={args.predictors} variants={args.variants}")

    # ---- build + self-checks
    pm, pe = f"{ZD}/zhou_mimic_cohort.parquet", f"{ZD}/zhou_eicu_cohort.parquet"
    fm = pd.read_parquet(f"{OUT}/features_mimic.parquet"); fe = pd.read_parquet(f"{OUT}/features_eicu.parquet")
    if args.rebuild or not (os.path.exists(pm) and os.path.exists(pe) and os.path.exists(f"{ZD}/zhou_flow.csv")):
        zm, flow_m, shock_m = build_mimic_z(OUT); ze, flow_e, s3e, shock_e = build_eicu_z(OUT)
        check_shock_mimic(shock_m, pd.read_parquet(f"{OUT}/cohort_final_mimic.parquet"), say)
        check_features("MIMIC", zm, fm, say)
        check_eicu(s3e, shock_e, fe, say)
        check_features("eICU", ze, fe, say)
        zm.to_parquet(pm); ze.to_parquet(pe)
        pd.DataFrame([{"dataset": "MIMIC-IV", "step": k, "n": v} for k, v in flow_m.items()] +
                     [{"dataset": "eICU-CRD", "step": k, "n": v} for k, v in flow_e.items()]).to_csv(f"{ZD}/zhou_flow.csv", index=False)
    else:
        say(f"reusing {pm} and {pe} (use --rebuild to redo the build and the self-checks)")
    if args.only_build:
        say("--only_build: emulated cohorts built and self-checks passed."); return
    flow = pd.read_csv(f"{ZD}/zhou_flow.csv"); say(flow.to_string(index=False))
    zm, ze = clean(pd.read_parquet(pm)), clean(pd.read_parquet(pe))

    # ---- outcome definitions
    def outcome(d):
        s = d.shock_h
        any_ = (s.notna() & (s <= args.max_hours)).values.astype(int)
        early = (s.notna() & (s <= WIN - 1)).values                   # onset inside the predictor window (hour 23 or earlier, incl. before admission)
        return any_, early
    ym, em = outcome(zm); ye, ee = outcome(ze)
    ns = []
    for nm, d, y, e in (("MIMIC-IV", zm, ym, em), ("eICU-CRD", ze, ye, ee)):
        n, ev, ea = len(y), int(y.sum()), int((y.astype(bool) & e).sum())
        ns.append({"dataset": nm, "stays": n, "events_any": ev, "events_early_onset_in_window": ea, "events_late": ev - ea,
                   "prevalence_any": ev / n, "share_of_events_that_are_early": ea / max(ev, 1), "stays_E2": int((~e).sum()), "events_E2": ev - ea,
                   "prevalence_E2": (ev - ea) / max(int((~e).sum()), 1)})
        say(f"{nm}: n={n:,} events(any, <= {args.max_hours} h)={ev:,} ({ev / n:.1%}) | early (onset <= hour 23)={ea:,} ({ea / max(ev, 1):.1%} of events) | "
            f"E2 set n={int((~e).sum()):,} late events={ev - ea:,}")
    pd.DataFrame(ns).to_csv(f"{ZD}/zhou_design_ns{TG}.csv", index=False)

    # ---- exposure: how much of the outcome definition is already in the window
    exposure = []
    for nm, d, y, e in (("MIMIC-IV", zm, ym, em), ("eICU-CRD", ze, ye, ee)):
        for grp, mk in (("early events", (y == 1) & e), ("late events", (y == 1) & ~e), ("no event", y == 0)):
            q = d[mk]
            exposure.append({"dataset": nm, "group": grp, "n": int(mk.sum()),
                             "vasopressor_running_at_last_window_hour_%": 100 * q.on_pressor.mean(), "s_cv>=3_%": 100 * (q.s_cv >= 3).mean(),
                             "last_lactate>2_%": 100 * (q.lactate_last > 2).mean(), "sofa_total_median": q.sofa_total.median()})
    pd.DataFrame(exposure).to_csv(f"{ZD}/zhou_overlap_exposure{TG}.csv", index=False)
    say("\n=== exposure (what the predictor window already contains) ==="); say(pd.DataFrame(exposure).round(1).to_string(index=False))

    # ---- models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score

    def make_learner():
        if args.model == "RF":
            return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                 RandomForestClassifier(n_estimators=args.rf_trees, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42))
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)

    def fit_predict(Xtr, ytr, Xte):
        m = make_learner(); m.fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]

    PSETS = {"first": FIRST_COLS + STATIC, "agg": AGG_COLS + STATIC}
    psets = [p.strip() for p in args.predictors.split(",") if p.strip()]
    LAC = lambda cols: [c for c in cols if c.startswith("lactate_")]
    ALLV = {"full": lambda cols: cols, "PL": lambda cols: [c for c in cols if c not in PRESS and c not in LAC(cols)],
            "L": lambda cols: [c for c in cols if c not in LAC(cols)]}
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    if "full" not in want or any(v not in ALLV for v in want): raise SystemExit(f"--variants must contain 'full' and only {list(ALLV)}: got {want}")
    VARIANT = {v: ALLV[v] for v in want}
    fold = folds_by_group(zm.subject_id.values, args.folds, args.seed)
    P = {"MIMIC": {}, "eICU": {}}
    for ps in psets:
        for var, sel in VARIANT.items():
            cols = sel(PSETS[ps]); cols = [c for c in cols if c in zm.columns]
            for ex in ("E1", "E2"):
                popm = np.ones(len(zm), bool) if ex == "E1" else ~em
                oof = np.full(len(zm), np.nan)
                for f in range(args.folds):
                    tr = popm & (fold != f); te = popm & (fold == f)
                    oof[te] = fit_predict(zm.loc[tr, cols], ym[tr], zm.loc[te, cols])
                P["MIMIC"][(ps, var, ex)] = oof
                P["eICU"][(ps, var, ex)] = fit_predict(zm.loc[popm, cols], ym[popm], ze[cols])
                say(f"  [{ps:5s} {var:4s} {ex}] MIMIC OOF AUROC {roc_auc_score(ym[popm], oof[popm]):.3f} | "
                    f"eICU AUROC {roc_auc_score(ye[(~ee) if ex == 'E2' else np.ones(len(ze), bool)], P['eICU'][(ps, var, ex)][(~ee) if ex == 'E2' else np.ones(len(ze), bool)]):.3f}")

    # ---- metrics with paired cluster bootstrap
    rows, diffs = [], []
    for ds, y, e, clus in (("MIMIC-IV (out-of-fold)", ym, em, zm.subject_id.values), ("eICU-CRD external", ye, ee, ze.hospitalid.values)):
        key = "MIMIC" if ds.startswith("MIMIC") else "eICU"
        N = len(y); allm = np.ones(N, bool); m2 = ~e; mearly = (y == 0) | e          # populations
        idxs = make_boot(clus, args.nboot, args.seed + (0 if key == "MIMIC" else 1))
        ser = {}                                                                  # name -> (point, boot array)

        def add(name, fn, p, mask):
            pt = metric(fn, y, p, mask); v = np.array([metric(fn, y, p, mask, i) for i in idxs]); ser[name] = (pt, v)
            return pt, v
        for ps in psets:
            for var in VARIANT:
                p1, p2 = P[key][(ps, var, "E1")], P[key][(ps, var, "E2")]
                specs = [("E1", "AUROC", roc_auc_score, p1, allm, "E1: shock at any time (Zhou-like), all stays"),
                         ("E1", "AUPRC", average_precision_score, p1, allm, "E1: shock at any time (Zhou-like), all stays"),
                         ("E2", "AUROC", roc_auc_score, p2, m2, "E2: late shock only, stays with early shock removed (E2 model)"),
                         ("E2", "AUPRC", average_precision_score, p2, m2, "E2: late shock only, stays with early shock removed (E2 model)"),
                         ("E1-on-early", "AUROC", roc_auc_score, p1, mearly, "E1 model: non-events + early events"),
                         ("E1-on-late", "AUROC", roc_auc_score, p1, m2, "E1 model: non-events + late events")]
                for code, mn, fn, p, mask, label in specs:
                    pt, v = add((ps, var, code, mn), fn, p, mask); lo, hi = pct(v)
                    rows.append({"dataset": ds, "predictors": ps, "variant": var, "analysis": code, "label": label, "metric": mn, "value": pt, "lo": lo, "hi": hi,
                                 "formatted": fmt(pt, lo, hi), "n": int(mask.sum()), "events": int(y[mask].sum()), "prevalence": float(y[mask].mean())})
        def diff(name_a, name_b, text, ps, var=None):
            a, b = ser[name_a], ser[name_b]; d = a[1] - b[1]; pt = a[0] - b[0]; lo, hi = pct(d)
            diffs.append({"dataset": ds, "predictors": ps, "variant": var if var else "", "comparison": text, "metric": name_a[3], "difference": pt, "lo": lo, "hi": hi,
                          "formatted": fmt(pt, lo, hi)})
        for ps in psets:
            for var in VARIANT:
                for mn in ("AUROC", "AUPRC"):
                    diff((ps, var, "E1", mn), (ps, var, "E2", mn), "E1 minus E2 (early events counted vs removed)", ps, var)
                diff((ps, var, "E1-on-early", "AUROC"), (ps, var, "E1-on-late", "AUROC"), "same E1 model: early-event population minus late-event population", ps, var)
                diff((ps, var, "E2", "AUROC"), (ps, var, "E1-on-late", "AUROC"), "late events: E2 model (early events removed from training) minus E1 model", ps, var)
            for var in VARIANT:
                if var == "full": continue
                for ex in ("E1", "E2", "E1-on-early", "E1-on-late"):
                    diff((ps, var, ex, "AUROC"), (ps, "full", ex, "AUROC"), f"{var} minus full inputs ({ex})", ps)
    res = pd.DataFrame(rows); res.to_csv(f"{ZD}/zhou_results{TG}.csv", index=False)
    dd = pd.DataFrame(diffs); dd.to_csv(f"{ZD}/zhou_paired_differences{TG}.csv", index=False)
    first_run = f"{ZD}/zhou_results.csv"
    if TG and os.path.exists(first_run):                  # a second run must reproduce the shared ("full") rows of the first one exactly
        old = pd.read_csv(first_run); key = ["dataset", "predictors", "variant", "analysis", "metric"]
        mg = old[old.variant == "full"][key + ["value", "lo", "hi"]].merge(res[res.variant == "full"][key + ["value", "lo", "hi"]], on=key, suffixes=("_old", "_new"))
        dv = float(np.nanmax(np.abs(mg[["value_old", "lo_old", "hi_old"]].values - mg[["value_new", "lo_new", "hi_new"]].values))) if len(mg) else float("nan")
        say(f"reproducibility: {len(mg)} shared rows of the first run, largest difference in value / CI limits = {dv:.2e}" + ("" if dv < 1e-9 else "  <-- WARNING: not identical"))
    say("\n=== AUROC / AUPRC ==="); say(res[["dataset", "predictors", "variant", "analysis", "metric", "formatted", "events", "prevalence"]].to_string(index=False))
    say("\n=== paired differences (cluster bootstrap) ==="); say(dd[["dataset", "predictors", "variant", "comparison", "metric", "formatted"]].to_string(index=False))
    say(f"\nDone. Results in {ZD}"); LOG.close()


if __name__ == "__main__":
    main()
