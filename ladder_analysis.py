#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ladder_analysis.py
==================
Controlled experiment on predictor-outcome overlap, run inside the SAME cohort and pipeline as the primary analysis.

Question: how much of the apparent discrimination of a septic-shock model is explained by WHEN the predictors are
measured relative to the outcome window?  Exactly one factor is changed between designs: the end of the predictor window.

  Outcome (identical in every design): septic shock with onset in (index, index + H] hours, H in --horizons (default 6,24).
  Cohort  (identical in every design): primary-cohort stays whose ICU stay lasts until index + H + 1 h, so that every design
                                       has data; patients are split exactly as in the primary analysis (temporal split; eICU external).
  Design B  predictors end 1 h after the index (= the primary analysis: strictly before the outcome window).
  Design C  predictors end at the end of the outcome window, i.e. index + H + 1 h (concurrent classification).
  Design C-P  = C without the vasopressor-derived inputs (on_pressor, s_cv, sofa_total).
  Design C-L  = C without lactate.
  Design C-PL = C without both.
  Matched ablations at the predictive landmark (added in v2; they remove the SAME inputs from B, so that C-P/C-L/C-PL can be
  compared with a design that has lost the same information, not only with the full B):
  Design B-P, B-L, B-PL = B without the vasopressor-derived inputs / without lactate / without both.
  The paired contrasts C-P minus B-P, C-L minus B-L and C-PL minus B-PL isolate the effect of the predictor window with the
  input set held fixed.
  Model, hyper-parameters, imputation, cleaning and bootstrap are those of the primary analysis (random forest, 500 trees).

Step 1 (--only_build): re-derives the predictor tables at the required landmarks from the raw sub-tables written by
ICUmodelprediction.ipynb, with the same SQL logic, and SELF-CHECKS that the k = 1 table reproduces the existing
features_mimic.parquet / features_eicu.parquet.  The run stops if it does not.
Step 2: fits the designs and writes the tables/figure.

Inputs (in --out):  features_mimic, features_eicu, cohort_final_mimic, cohort_eicu, chart_sub, lab_sub, sofa_hourly,
                    eicu_vitalp_sub, eicu_vitala_sub, eicu_lab_sub, sofa_hourly_eicu   (all .parquet)
Outputs (in --out/ladder): features_{mimic,eicu}_k{K}.parquet, ladder_results.csv, ladder_paired_differences.csv,
                    ladder_overlap_exposure.csv, ladder_design_ns.csv, fig_ladder.png, ladder_log.txt
Nothing here is tuned on external data.  Patient-level files stay inside --out and must not be uploaded anywhere.
"""
import argparse, os, sys, json, warnings, platform
import numpy as np, pandas as pd
import duckdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/content/drive/MyDrive/sepsis_project")
ap.add_argument("--horizons", default="6,24")
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--rf_trees", type=int, default=500)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--model", default="RF", choices=["RF", "LightGBM"],
                help="learner used in every design (hyper-parameters fixed, identical to extended_analysis_v3.py); outputs go to ladder/ (RF) or ladder_lightgbm/")
ap.add_argument("--outcome", default="primary", choices=["primary", "vaso_any", "vaso_2h", "primary_kept_vaso_any", "primary_kept_vaso_2h"],
                help="outcome definition (sensitivity analysis, see outcome_definitions.py): primary = vasopressor and lactate > 2 (manuscript); "
                     "vaso_any / vaso_2h = vasopressor only (any / at least 2 consecutive hours), no lactate in the outcome. Same stays, same features; only the label "
                     "changes. Requires the landmark tables of the primary run in ladder/. Outputs go to ladder_outcome_<name>[_lightgbm]/. "
                     "primary_kept_<rule> is the CONTROL: the primary outcome (vasopressor and lactate > 2) on exactly the stays that remain under <rule>, so that the effect of the "
                     "stay selection can be separated from the effect of the outcome definition.")
ap.add_argument("--only_build", action="store_true", help="build the landmark tables + self-check, then stop")
ap.add_argument("--rebuild", action="store_true", help="rebuild landmark tables even if they already exist")
args = ap.parse_args()
OUT = args.out
LD = os.path.join(OUT, "ladder" if args.model == "RF" else "ladder_" + args.model.lower())
if args.outcome != "primary": LD = os.path.join(OUT, "ladder_outcome_" + args.outcome + ("" if args.model == "RF" else "_" + args.model.lower()))
os.makedirs(LD, exist_ok=True)
FEAT_DIR = os.path.join(OUT, "ladder"); os.makedirs(FEAT_DIR, exist_ok=True)   # landmark feature tables are shared by all learners
HORIZONS = [int(x) for x in args.horizons.split(",") if x.strip()]
B, SEED = args.nboot, args.seed
LOG = open(os.path.join(LD, "ladder_log.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()


say("python", platform.python_version(), "| duckdb", duckdb.__version__, "| pandas", pd.__version__)
say(f"horizons={HORIZONS} nboot={B} rf_trees={args.rf_trees} model={args.model} outcome={args.outcome}")

VARS = ["hr", "rr", "spo2", "sbp", "dbp", "map", "temp"]
M_LAB = {"creat": 50912, "bili": 50885, "plt": 51265, "lactate": 50813, "wbc": 51301, "hgb": 51222,
         "k": 50971, "na": 50983, "bicarb": 50882, "ph": 50820, "pao2": 50821, "base_excess": 50802}
E_LAB = {"creat": "creatinine", "bili": "total bilirubin", "plt": "platelets x 1000", "lactate": "lactate", "wbc": "wbc x 1000",
         "hgb": "hgb", "k": "potassium", "na": "sodium", "bicarb": "bicarbonate", "ph": "ph", "pao2": "pao2", "base_excess": "base excess"}
SOFA_SEL = "s.sofa AS sofa_total, s.s_resp, s.s_coag, s.s_liver, s.s_cv, s.s_cns, s.s_renal, s.gcs_min, s.pf_min, s.any_pressor AS on_pressor"
LIM = {"hr": (0, 300), "rr": (0, 100), "spo2": (0, 100), "sbp": (0, 350), "dbp": (0, 250), "map": (0, 300), "temp": (25, 45)}


def clean(d):                                   # identical to the primary analysis
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
    con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=2"); return con


# ------------------------------------------------------------------ landmark feature builders (same SQL logic as the notebook)
def build_mimic(k, static):
    """features at prediction time = index_time + k hours.  static: DataFrame with stay_id (cohort definition)."""
    con = newcon(); R = lambda f: f"read_parquet('{OUT}/{f}.parquet')"
    con.register("st", static[["stay_id"]])
    con.execute(f"""CREATE TABLE cf AS
        SELECT c.stay_id, c.subject_id, c.intime, c.index_time, c.index_time + INTERVAL '{k} hours' AS pred_time
        FROM {R('cohort_final_mimic')} c JOIN st USING (stay_id)""")
    items = [220045, 220210, 220277, 220179, 220050, 220180, 220051, 220181, 220052, 225312, 223762, 223761]
    con.execute(f"""CREATE TABLE chart_long AS
        SELECT cf.stay_id, ch.charttime,
          CASE WHEN ch.itemid=220045 THEN 'hr' WHEN ch.itemid=220210 THEN 'rr' WHEN ch.itemid=220277 THEN 'spo2'
               WHEN ch.itemid IN (220179,220050) THEN 'sbp' WHEN ch.itemid IN (220180,220051) THEN 'dbp'
               WHEN ch.itemid IN (220181,220052,225312) THEN 'map'
               WHEN ch.itemid IN (223762,223761) THEN 'temp' END AS var,
          CASE WHEN ch.itemid=223761 THEN (ch.valuenum-32)/1.8 ELSE ch.valuenum END AS v
        FROM {R('chart_sub')} ch JOIN cf USING (stay_id)
        WHERE ch.itemid IN ({','.join(map(str, items))})
          AND ch.charttime < cf.pred_time AND ch.charttime >= cf.pred_time - INTERVAL '24 hours'""")
    aggs = []
    for x in VARS:
        aggs += [f"AVG(v) FILTER (WHERE var='{x}') AS {x}_mean", f"MIN(v) FILTER (WHERE var='{x}') AS {x}_min",
                 f"MAX(v) FILTER (WHERE var='{x}') AS {x}_max", f"arg_max(v, charttime) FILTER (WHERE var='{x}') AS {x}_last"]
    con.execute(f"CREATE TABLE chart_feat AS SELECT stay_id, {', '.join(aggs)} FROM chart_long GROUP BY stay_id")
    case = "CASE " + " ".join(f"WHEN lb.itemid={i} THEN '{n}'" for n, i in M_LAB.items()) + " END"
    con.execute(f"""CREATE TABLE lab_long AS
        SELECT cf.stay_id, lb.charttime, {case} AS var, lb.valuenum AS v
        FROM {R('lab_sub')} lb JOIN cf ON cf.subject_id = lb.subject_id
        WHERE lb.itemid IN ({','.join(map(str, M_LAB.values()))})
          AND lb.charttime < cf.pred_time AND lb.charttime >= cf.pred_time - INTERVAL '24 hours'""")
    aggs = []
    for x in M_LAB:
        aggs += [f"arg_max(v, charttime) FILTER (WHERE var='{x}') AS {x}_last", f"MIN(v) FILTER (WHERE var='{x}') AS {x}_min",
                 f"MAX(v) FILTER (WHERE var='{x}') AS {x}_max"]
    con.execute(f"CREATE TABLE lab_feat AS SELECT stay_id, {', '.join(aggs)} FROM lab_long GROUP BY stay_id")
    con.execute(f"""CREATE TABLE sofa_idx AS SELECT cf.stay_id, {SOFA_SEL}
        FROM cf JOIN {R('sofa_hourly')} s ON s.stay_id = cf.stay_id AND s.hr = cf.pred_time - INTERVAL '1 hour'""")
    return con.execute("""SELECT cf.stay_id, date_diff('hour', cf.intime, cf.pred_time) AS hours_icu_to_pred,
                                 ch.* EXCLUDE (stay_id), lb.* EXCLUDE (stay_id), so.* EXCLUDE (stay_id)
                          FROM cf LEFT JOIN chart_feat ch USING (stay_id) LEFT JOIN lab_feat lb USING (stay_id)
                               LEFT JOIN sofa_idx so USING (stay_id)""").df()


def build_eicu(k, static):
    """static: DataFrame with stay_id, hours_icu_to_pred (primary design = index_hr + 1)."""
    con = newcon(); R = lambda f: f"read_parquet('{OUT}/{f}.parquet')"
    cf = static[["stay_id", "hours_icu_to_pred"]].assign(index_hr=lambda d: d.hours_icu_to_pred - 1)
    cf["pred_min"] = (cf.index_hr + k) * 60
    con.register("cf", cf[["stay_id", "index_hr", "pred_min"]])
    con.execute(f"""CREATE TABLE vit AS
        SELECT stay_id, t, heartrate AS hrt, respiration AS rrt, sao2 AS spo2t, temperature AS tempt,
               systemicsystolic AS sbpt, systemicdiastolic AS dbpt, systemicmean AS mapt FROM {R('eicu_vitalp_sub')}
        UNION ALL
        SELECT stay_id, t, NULL, NULL, NULL, NULL, noninvasivesystolic, noninvasivediastolic, noninvasivemean FROM {R('eicu_vitala_sub')}""")
    vmap = {"hr": "hrt", "rr": "rrt", "spo2": "spo2t", "sbp": "sbpt", "dbp": "dbpt", "map": "mapt", "temp": "tempt"}
    aggs = []
    for x, c in vmap.items():
        aggs += [f"AVG({c}) AS {x}_mean", f"MIN({c}) AS {x}_min", f"MAX({c}) AS {x}_max", f"arg_max({c}, t) FILTER (WHERE {c} IS NOT NULL) AS {x}_last"]
    con.execute(f"""CREATE TABLE chart_feat AS SELECT cf.stay_id, {', '.join(aggs)}
        FROM vit JOIN cf ON cf.stay_id = vit.stay_id AND vit.t < cf.pred_min AND vit.t >= cf.pred_min - 1440 GROUP BY cf.stay_id""")
    con.execute(f"CREATE TABLE lab AS SELECT * FROM {R('eicu_lab_sub')}")
    aggs = []
    for x, nm in E_LAB.items():
        aggs += [f"arg_max(v, t) FILTER (WHERE labname='{nm}') AS {x}_last", f"MIN(v) FILTER (WHERE labname='{nm}') AS {x}_min",
                 f"MAX(v) FILTER (WHERE labname='{nm}') AS {x}_max"]
    con.execute(f"""CREATE TABLE lab_feat AS SELECT cf.stay_id, {', '.join(aggs)}
        FROM lab JOIN cf ON cf.stay_id = lab.stay_id AND lab.t < cf.pred_min AND lab.t >= cf.pred_min - 1440 GROUP BY cf.stay_id""")
    con.execute(f"""CREATE TABLE sofa_idx AS SELECT cf.stay_id, {SOFA_SEL}
        FROM cf JOIN {R('sofa_hourly_eicu')} s ON s.stay_id = cf.stay_id AND s.hr = cf.index_hr + {k} - 1""")
    return con.execute(f"""SELECT cf.stay_id, cf.index_hr + {k} AS hours_icu_to_pred,
                                  ch.* EXCLUDE (stay_id), lb.* EXCLUDE (stay_id), so.* EXCLUDE (stay_id)
                           FROM cf LEFT JOIN chart_feat ch USING (stay_id) LEFT JOIN lab_feat lb USING (stay_id)
                                LEFT JOIN sofa_idx so USING (stay_id)""").df()


def selfcheck(tag, new, old, cols):
    """the k=1 table rebuilt here must reproduce the table the primary analysis used"""
    new, old = clean(new), clean(old)
    m = old[["stay_id"] + cols].merge(new[["stay_id"] + cols], on="stay_id", suffixes=("_old", "_new"))
    say(f"[{tag}] self-check on {len(m):,} of {len(old):,} stays")
    bad = []
    for c in cols:
        a, b = m[c + "_old"].astype(float), m[c + "_new"].astype(float)
        eq = (a.isna() & b.isna()) | ((a - b).abs() <= 1e-6 + 1e-6 * a.abs())
        frac = 1 - eq.mean()
        tol = 0.02 if c.endswith("_last") else 5e-4        # arg_max can break timestamp ties differently between runs
        if frac > tol: bad.append((c, round(float(frac), 4)))
    if len(m) < 0.999 * len(old): bad.append(("row coverage", round(len(m) / len(old), 4)))
    if bad:
        raise RuntimeError(f"[{tag}] the rebuilt k=1 table does NOT reproduce the primary features (column, mismatch fraction): {bad[:15]}. "
                           "Do not trust any landmark result until this is explained.")
    say(f"[{tag}] self-check PASSED (k=1 reproduces the primary feature table)")


# ------------------------------------------------------------------ data
fm_raw = pd.read_parquet(f"{OUT}/features_mimic.parquet")
fe_raw = pd.read_parquet(f"{OUT}/features_eicu.parquet")
cm = pd.read_parquet(f"{OUT}/cohort_final_mimic.parquet")[["stay_id", "index_time", "outtime"]]
ce = pd.read_parquet(f"{OUT}/cohort_eicu.parquet")[["stay_id", "outoffset"]]
if args.outcome != "primary":
    # ---- sensitivity analysis of the outcome definition: same stays, same predictors, only the label changes -------------
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import outcome_definitions as od
    if args.rebuild: sys.exit("--rebuild is not allowed with --outcome (the landmark tables of the primary run are reused unchanged).")
    for k_ in sorted(set([1] + [H + 1 for H in HORIZONS])):
        for ds_ in ("mimic", "eicu"):
            if not os.path.exists(f"{FEAT_DIR}/features_{ds_}_k{k_}.parquet"):
                sys.exit(f"Run the primary ladder_analysis.py first (landmark table features_{ds_}_k{k_}.parquet is missing).")
    con_o = newcon(); log_o = []
    def _l(*a):
        m = " ".join(str(x) for x in a); say(m); log_o.append(m)
    # MIMIC-IV: hr is a timestamp; hours_to_shock = hours from index_time
    src_m = f"read_parquet('{OUT}/sofa_hourly.parquet')"
    say(f"[MIMIC] stays with a gap in the hourly grid (must be 0): {od.grid_gaps(con_o, src_m, True)}")
    idx_m = cm.set_index("stay_id").index_time
    chk = od.relabel(fm_raw, idx_m, od.onset_frame(con_o, src_m, "primary", True), "primary re-derived (check)", log=lambda *a: None)
    same = (chk.shape[0] == fm_raw.shape[0]) and np.array_equal(chk.shock.values, fm_raw.shock.values) and \
           np.allclose(chk.hours_to_shock.fillna(-1).values, fm_raw.hours_to_shock.fillna(-1).values)
    say(f"[MIMIC] primary outcome re-derived from sofa_hourly reproduces shock and hours_to_shock of the primary features: {'OK' if same else 'MISMATCH - STOP'}")
    if not same: sys.exit("Outcome derivation does not reproduce the primary labels; do not use this run.")
    CONTROL = args.outcome.startswith("primary_kept_"); RULE = args.outcome.replace("primary_kept_", "") if CONTROL else args.outcome
    fm_prim = fm_raw.copy()
    fm_raw = od.relabel(fm_raw, idx_m, od.onset_frame(con_o, src_m, RULE, True), "MIMIC " + RULE, log=_l)
    if CONTROL: fm_raw = fm_prim[fm_prim.stay_id.isin(fm_raw.stay_id)].copy(); _l(f"[MIMIC control] primary outcome kept on the {len(fm_raw):,} stays that remain under {RULE}; events {int(fm_raw.shock.sum()):,}")
    # eICU-CRD: hr is an integer hour; index hour = hours_icu_to_pred - 1
    src_e = f"read_parquet('{OUT}/sofa_hourly_eicu.parquet')"
    say(f"[eICU] stays with a gap in the hourly grid (must be 0): {od.grid_gaps(con_o, src_e, False)}")
    idx_e = (fe_raw.set_index("stay_id").hours_icu_to_pred - 1)
    chk = od.relabel(fe_raw, idx_e, od.onset_frame(con_o, src_e, "primary", False), "primary re-derived (check)", log=lambda *a: None)
    same = (chk.shape[0] == fe_raw.shape[0]) and np.array_equal(chk.shock.values, fe_raw.shock.values) and \
           np.allclose(chk.hours_to_shock.fillna(-1).values, fe_raw.hours_to_shock.fillna(-1).values)
    say(f"[eICU] primary outcome re-derived from sofa_hourly_eicu reproduces shock and hours_to_shock of the primary features: {'OK' if same else 'MISMATCH - STOP'}")
    if not same: sys.exit("Outcome derivation does not reproduce the primary labels; do not use this run.")
    fe_prim = fe_raw.copy()
    fe_raw = od.relabel(fe_raw, idx_e, od.onset_frame(con_o, src_e, RULE, False), "eICU " + RULE, log=_l)
    if CONTROL: fe_raw = fe_prim[fe_prim.stay_id.isin(fe_raw.stay_id)].copy(); _l(f"[eICU control] primary outcome kept on the {len(fe_raw):,} stays that remain under {RULE}; events {int(fe_raw.shock.sum()):,}")
    summ_o = []                       # how the cohort and the events change (aggregated counts only)
    for nm_, prim_, alt_ in (("MIMIC-IV", fm_prim, fm_raw), ("eICU-CRD", fe_prim, fe_raw)):
        kept_ = prim_[prim_.stay_id.isin(alt_.stay_id)]
        r_ = {"dataset": nm_, "outcome": args.outcome, "stays_primary_cohort": len(prim_), "stays_removed": len(prim_) - len(alt_), "stays_kept": len(alt_),
              "events_primary_in_kept": int(kept_.shock.sum()), "events_alt": int(alt_.shock.sum())}
        for H_ in HORIZONS:
            ep_ = set(kept_.stay_id[(kept_.shock == 1) & (kept_.hours_to_shock <= H_)]); ea_ = set(alt_.stay_id[(alt_.shock == 1) & (alt_.hours_to_shock <= H_)])
            r_.update({f"events_primary_H{H_}": len(ep_), f"events_alt_H{H_}": len(ea_), f"events_in_both_H{H_}": len(ep_ & ea_)})
        summ_o.append(r_)
    pd.DataFrame(summ_o).to_csv(f"{LD}/outcome_cohort_summary.csv", index=False)
    say(pd.DataFrame(summ_o).to_string(index=False))
TRAIN_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
DROP = ["stay_id", "subject_id", "hadm_id", "anchor_year_group", "race", "gender", "shock", "hours_to_shock"]
FEATS = [c for c in fm_raw.columns if c not in DROP]
FEATS_STATIC = [c for c in FEATS if c in ("age", "male")]
STAT_M = ["stay_id", "subject_id", "anchor_year_group", "shock", "hours_to_shock", "age", "male"]
STAT_E = ["stay_id", "hospitalid", "shock", "hours_to_shock", "age", "male"]
say(f"MIMIC stays {len(fm_raw):,} | eICU stays {len(fe_raw):,} | {len(FEATS)} features")

KS = sorted(set([1] + [H + 1 for H in HORIZONS]))
FM, FE = {}, {}
for k in KS:
    pm, pe = f"{FEAT_DIR}/features_mimic_k{k}.parquet", f"{FEAT_DIR}/features_eicu_k{k}.parquet"
    if k == 1:           # primary tables, with a self-check of the builders
        for tag, builder, raw, path in [("MIMIC k=1", build_mimic, fm_raw, pm), ("eICU k=1", build_eicu, fe_raw, pe)]:
            if args.rebuild or not os.path.exists(path):
                new = builder(1, raw)
                selfcheck(tag, new, raw, [c for c in FEATS if c in new.columns])
                new.to_parquet(path)
            else:
                say(f"[{tag}] reusing {path} (use --rebuild to redo the self-check)")
        FM[1] = fm_raw[STAT_M + [c for c in FEATS if c not in STAT_M]].copy()
        FE[1] = fe_raw[STAT_E + [c for c in FEATS if c not in STAT_E]].copy()
        continue
    for tag, builder, raw, path, store, stat in [(f"MIMIC k={k}", build_mimic, fm_raw, pm, FM, STAT_M), (f"eICU k={k}", build_eicu, fe_raw, pe, FE, STAT_E)]:
        if args.rebuild or not os.path.exists(path):
            say(f"[{tag}] building ..."); builder(k, raw).to_parquet(path)
        new = pd.read_parquet(path)
        store[k] = raw[stat].merge(new, on="stay_id", how="inner")
        say(f"[{tag}] {len(store[k]):,} stays; median hours_icu_to_pred {store[k].hours_icu_to_pred.median():.0f}")
if args.only_build:
    say("--only_build: landmark tables built and self-check passed. Run again without --only_build for the experiment."); sys.exit(0)
FM = {k: clean(v) for k, v in FM.items()}
FE = {k: clean(v) for k, v in FE.items()}

# index / discharge information
cm = cm.set_index("stay_id"); ce = ce.set_index("stay_id")
fe_idx = fe_raw.set_index("stay_id").hours_icu_to_pred - 1            # eICU index hour


# ------------------------------------------------------------------ statistics helpers
def make_boot(clusters, nb, seed):
    rng = np.random.default_rng(seed); codes = pd.factorize(pd.Series(clusters))[0]
    groups = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    return [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(nb)]


def boot_vals(y, p, idxs, fn):
    out = []
    for i in idxs:
        yy = y[i]
        out.append(fn(yy, p[i]) if 0 < yy.sum() < len(yy) else np.nan)
    return np.array(out)


def pct(v): v = v[~np.isnan(v)]; return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def fmt(pt, lo, hi): return f"{pt:.3f} ({lo:.3f}-{hi:.3f})"


def make_learner():
    if args.model == "RF":     # primary analysis
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             RandomForestClassifier(n_estimators=args.rf_trees, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42))
    import lightgbm as lgb     # same fixed hyper-parameters as make_model("LightGBM") in extended_analysis_v3.py; NaN handled natively
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)


def fit_predict(train, ytr, cols, tests):
    m = make_learner()
    m.fit(train[cols], ytr)
    return [m.predict_proba(t[cols])[:, 1] for t in tests]


PRESS = [c for c in ["on_pressor", "s_cv", "sofa_total"] if c in FEATS]
LAC = [c for c in FEATS if c.startswith("lactate_")]
rows, diffs, exposure, ns = [], [], [], []
for H in HORIZONS:
    kC = H + 1
    # cohort: ICU stay lasts until the end of the concurrent predictor window (identical for every design)
    okm = cm.loc[FM[1].stay_id].pipe(lambda d: (d.outtime.values >= (d.index_time + pd.Timedelta(hours=kC)).values))
    oke = ((ce.loc[FE[1].stay_id].outoffset.values) >= (fe_idx.loc[FE[1].stay_id].values + kC) * 60)
    keepM, keepE = set(FM[1].stay_id[okm]), set(FE[1].stay_id[oke])

    def sub(store, k, keep):
        d = store[k]; return d[d.stay_id.isin(keep)].sort_values("stay_id").reset_index(drop=True)

    D = {"B": (1, FEATS), "C": (kC, FEATS), "C-P": (kC, [c for c in FEATS if c not in PRESS]),
         "C-L": (kC, [c for c in FEATS if c not in LAC]), "C-PL": (kC, [c for c in FEATS if c not in PRESS + LAC]),
         "B-P": (1, [c for c in FEATS if c not in PRESS]), "B-L": (1, [c for c in FEATS if c not in LAC]),
         "B-PL": (1, [c for c in FEATS if c not in PRESS + LAC])}
    LABEL = {"B": "B: predictors end 1 h after index (primary design)", "C": f"C: predictors end with the {H}-h outcome window",
             "C-P": "C-P: C without vasopressor-derived inputs", "C-L": "C-L: C without lactate", "C-PL": "C-PL: C without both",
             "B-P": "B-P: B without vasopressor-derived inputs", "B-L": "B-L: B without lactate", "B-PL": "B-PL: B without both"}
    sets = {}
    for k in sorted(set(v[0] for v in D.values())):
        m_, e_ = sub(FM, k, keepM), sub(FE, k, keepE)
        sets[k] = {"tr": m_[m_.anchor_year_group.isin(TRAIN_GROUPS)].reset_index(drop=True), "te": m_[~m_.anchor_year_group.isin(TRAIN_GROUPS)].reset_index(drop=True), "ev": e_}
    ref = sets[1]
    for k in sets:        # every design must contain exactly the same stays in the same order
        for part in ("tr", "te", "ev"):
            assert (sets[k][part].stay_id.values == ref[part].stay_id.values).all(), f"stay mismatch between landmarks (H={H}, {part})"
    Y = {part: ((ref[part].shock == 1) & (ref[part].hours_to_shock <= H)).astype(int).values for part in ("tr", "te", "ev")}
    CL = {"te": ref["te"].subject_id.values, "ev": ref["ev"].hospitalid.values}
    for part, nm in (("tr", "MIMIC development"), ("te", "MIMIC temporal test"), ("ev", "eICU external")):
        ns.append({"H": H, "dataset": nm, "n": len(Y[part]), "events": int(Y[part].sum()), "prevalence": Y[part].mean()})
        say(f"H={H}h {nm:22s} n={len(Y[part]):6,} events={int(Y[part].sum()):5,} ({Y[part].mean():.1%})")
    P = {"te": {}, "ev": {}}
    for name, (k, cols) in D.items():
        p_te, p_ev = fit_predict(sets[k]["tr"], Y["tr"], cols, [sets[k]["te"], sets[k]["ev"]])
        P["te"][name], P["ev"][name] = p_te, p_ev
        say(f"  H={H} design {name:5s} AUROC temporal {roc_auc_score(Y['te'], p_te):.3f} | eICU {roc_auc_score(Y['ev'], p_ev):.3f}")
    for part, nm in (("te", "MIMIC temporal test"), ("ev", "eICU external")):
        y = Y[part]; idxs = make_boot(CL[part], B, SEED + H * 7 + (0 if part == "te" else 1))
        met = {}
        for name in D:
            met[name] = {}
            for mn, fn in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
                v = boot_vals(y, P[part][name], idxs, fn); met[name][mn] = v
                pt = fn(y, P[part][name]); lo, hi = pct(v)
                rows.append({"H": H, "dataset": nm, "design": LABEL[name], "code": name, "metric": mn, "value": pt, "lo": lo, "hi": hi,
                             "formatted": fmt(pt, lo, hi), "n": len(y), "events": int(y.sum()), "prevalence": y.mean()})
        for name in D:                       # comparators evaluated at each landmark
            if name not in ("B", "C"): continue
            k = D[name][0]; d = sets[k][part]; trd = sets[k]["tr"]
            for cn, p in (("SOFA only", d.sofa_total.fillna(0).values), ("Lactate only", d.lactate_last.fillna(trd.lactate_last.median()).values)):
                for mn, fn in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
                    v = boot_vals(y, p, idxs, fn); pt = fn(y, p); lo, hi = pct(v)
                    rows.append({"H": H, "dataset": nm, "design": f"{cn} at landmark of {name}", "code": f"{cn} ({name})", "metric": mn, "value": pt, "lo": lo, "hi": hi,
                                 "formatted": fmt(pt, lo, hi), "n": len(y), "events": int(y.sum()), "prevalence": y.mean()})
        for name in D:                       # paired differences against B (same stays, same resamples)
            if name == "B": continue
            for mn, fn in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
                d = met[name][mn] - met["B"][mn]; pt = fn(y, P[part][name]) - fn(y, P[part]["B"]); lo, hi = pct(d)
                diffs.append({"H": H, "dataset": nm, "comparison": f"{name} minus B", "metric": mn, "difference": pt, "lo": lo, "hi": hi, "formatted": fmt(pt, lo, hi)})
        for a_, b_ in (("C-P", "B-P"), ("C-L", "B-L"), ("C-PL", "B-PL")):   # window effect with the input set held fixed
            for mn, fn in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
                d = met[a_][mn] - met[b_][mn]; pt = fn(y, P[part][a_]) - fn(y, P[part][b_]); lo, hi = pct(d)
                diffs.append({"H": H, "dataset": nm, "comparison": f"{a_} minus {b_} (matched inputs)", "metric": mn, "difference": pt, "lo": lo, "hi": hi, "formatted": fmt(pt, lo, hi)})
        for name in ("B", "C"):              # how much of the outcome information is already inside the predictors
            k = D[name][0]; d = sets[k][part]; ev_ = y == 1
            exposure.append({"H": H, "dataset": nm, "design": name, "events": int(y.sum()),
                             "events_with_vasopressor_running_at_last_predictor_hour_%": 100 * d.on_pressor[ev_].mean(),
                             "non_events_with_vasopressor_%": 100 * d.on_pressor[~ev_].mean(),
                             "events_with_last_lactate_>2_%": 100 * (d.lactate_last[ev_] > 2).mean(),
                             "non_events_with_last_lactate_>2_%": 100 * (d.lactate_last[~ev_] > 2).mean(),
                             "events_with_s_cv>=3_%": 100 * (d.s_cv[ev_] >= 3).mean(), "non_events_with_s_cv>=3_%": 100 * (d.s_cv[~ev_] >= 3).mean()})

res = pd.DataFrame(rows); res.to_csv(f"{LD}/ladder_results.csv", index=False)
pd.DataFrame(diffs).to_csv(f"{LD}/ladder_paired_differences.csv", index=False)
pd.DataFrame(exposure).to_csv(f"{LD}/ladder_overlap_exposure.csv", index=False)
pd.DataFrame(ns).to_csv(f"{LD}/ladder_design_ns.csv", index=False)
say("\n=== AUROC by design ===")
say(res[res.metric == "AUROC"][["H", "dataset", "design", "formatted", "events", "prevalence"]].to_string(index=False))
say("\n=== paired differences vs B (cluster bootstrap) ===")
say(pd.DataFrame(diffs)[["H", "dataset", "comparison", "metric", "formatted"]].to_string(index=False))
say("\n=== overlap exposure ===")
say(pd.DataFrame(exposure).round(1).to_string(index=False))

# figure
codes = ["B", "B-P", "B-L", "B-PL", "C", "C-P", "C-L", "C-PL"]
fig, axs = plt.subplots(len(HORIZONS), 2, figsize=(13, 3.6 * len(HORIZONS)), squeeze=False, sharey=True)
for i, H in enumerate(HORIZONS):
    for j, nm in enumerate(["MIMIC temporal test", "eICU external"]):
        d = res[(res.H == H) & (res.dataset == nm) & (res.metric == "AUROC")].set_index("code")
        ax = axs[i, j]; x = np.arange(len(codes)); v = d.loc[codes]
        ax.errorbar(x, v.value, yerr=[v.value - v.lo, v.hi - v.value], fmt="o", capsize=3)
        ax.axhline(0.5, ls=":", c="grey"); ax.set_xticks(x); ax.set_xticklabels(codes); ax.set_title(f"{nm} - outcome within {H} h"); ax.set_ylabel("AUROC")
plt.tight_layout(); plt.savefig(f"{LD}/fig_ladder.png", dpi=300); plt.close()
say(f"\nDone. Results in {LD}")
LOG.close()
