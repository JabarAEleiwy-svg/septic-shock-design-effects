#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
window_cumulative.py
====================
Extension of ladder_analysis.py for the peer-review revision (v4 -> v5).  Runs on the SAME files in --out
(sepsis_project/) with the SAME SQL logic, cleaning, learner, split, labels and cluster bootstrap.

PART A  Window designs that separate "the end of the window moved" from "the window got longer / shorter".
  For horizon H (index = t0; predictor windows are [pred_time - win, pred_time), pred_time = t0 + k h):
     B        k=1    win=24        primary design (predictors end 1 h after the index)
     B_wide   k=1    win=24+H      B plus H extra hours of EARLIER history
     C        k=H+1  win=24        fixed-length window shifted to the end of the outcome window   (as in the manuscript)
     C_star   k=H+1  win=24+H      B's whole window PLUS the H later hours (cumulative window; information superset of B)
     B_short  k=1    win=--short_win    short windows: amount of recent data matched
     C_short  k=H+1  win=--short_win
  Primary contrast: C_star - B_wide (equal window length, the extra H hours lie AFTER vs BEFORE the prediction time).
  Also reported: C_star - B (what the later hours add on top of everything B knows), B_wide - B, C - C_star, C - B,
  C_short - B_short.  Every contrast is also computed without the outcome-defining inputs (-PL).
  IMPORTANT data-availability fact: ICU charting starts at ICU admission (median prediction time 3 h MIMIC / 1 h eICU
  after admission), so B_wide gets little extra history; window_data_availability.csv quantifies this.

PART B  Lead-time analysis with SYMMETRIC conditioning (evaluation only, primary model, no retraining).
  original  : drop events with onset <= g h; keep ALL non-events (manuscript v4)
  symmetric : drop events with onset <= g h; keep only non-events still in the ICU after g h
  Reads the saved private predictions (results/private/predictions_*.parquet).

Self-checks (the run STOPS if they fail):
  1. my builder with k=1, win=24 reproduces features_mimic / features_eicu;
  2. my builder with k=H+1, win=24 reproduces ladder/features_*_k{H+1}.parquet when that file exists.

Outputs (aggregated only) in <out>/window_cum/: window_results.csv, window_paired_differences.csv, window_seeds.csv,
  window_design_ns.csv, window_data_availability.csv, leadtime_symmetric.csv, window_log.txt.  Feature caches stay in that
  folder too: they are patient-level, do NOT upload them anywhere.
"""
import argparse, os, sys, warnings, platform, time
import numpy as np, pandas as pd, duckdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/content/drive/MyDrive/sepsis_project")
ap.add_argument("--horizons", default="6,24")
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--rf_trees", type=int, default=500)
ap.add_argument("--seeds", default="42,43,44", help="training seeds; predictions are averaged, per-seed AUROC is reported")
ap.add_argument("--short_win", type=int, default=6, help="length (h) of the short windows B_short / C_short; 0 = skip")
ap.add_argument("--drop_hours", action="store_true", help="drop hours_icu_to_pred from every design (it moves with the window end)")
ap.add_argument("--skip_lead", action="store_true")
ap.add_argument("--only_lead", action="store_true")
ap.add_argument("--fast", action="store_true", help="smoke run: 50 trees, 1 seed, 100 bootstrap resamples")
ap.add_argument("--test_groups", default="", help="evaluate the MIMIC-IV test set only in these anchor_year_group values, comma separated, "
                "e.g. '2017 - 2019' (training is unchanged); Part B is skipped; use with --tag")
ap.add_argument("--tag", default="", help="suffix of the output folder window_cum<tag>")
args = ap.parse_args()
OUT = args.out
WD = os.path.join(OUT, "window_cum" + args.tag); os.makedirs(WD, exist_ok=True)
TEST_GROUPS = [x.strip() for x in args.test_groups.split(",") if x.strip()]
if TEST_GROUPS: args.skip_lead = True
FEAT_DIR = os.path.join(OUT, "ladder")
HORIZONS = [int(x) for x in args.horizons.split(",") if x.strip()]
SEEDS = [int(x) for x in args.seeds.split(",") if x.strip()]
B, TREES = args.nboot, args.rf_trees
if args.fast: B, TREES, SEEDS = 100, 50, SEEDS[:1]
LOG = open(os.path.join(WD, "window_log.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


say("python", platform.python_version(), "| duckdb", duckdb.__version__, "| pandas", pd.__version__)
say(f"test_groups={TEST_GROUPS or 'all'} tag='{args.tag}'")
say(f"horizons={HORIZONS} nboot={B} trees={TREES} seeds={SEEDS} short_win={args.short_win} drop_hours={args.drop_hours}")

# ------------------------------------------------------------------ constants identical to ladder_analysis.py
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


# ------------------------------------------------------------------ builders: ladder_analysis.py logic with a window length `win`
def build_mimic(k, static, win):
    """features from [pred_time - win h, pred_time), pred_time = index_time + k h; SOFA at pred_time - 1 h."""
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
          AND ch.charttime < cf.pred_time AND ch.charttime >= cf.pred_time - INTERVAL '{win} hours'""")
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
          AND lb.charttime < cf.pred_time AND lb.charttime >= cf.pred_time - INTERVAL '{win} hours'""")
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


def build_eicu(k, static, win):
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
        FROM vit JOIN cf ON cf.stay_id = vit.stay_id AND vit.t < cf.pred_min AND vit.t >= cf.pred_min - {win * 60} GROUP BY cf.stay_id""")
    con.execute(f"CREATE TABLE lab AS SELECT * FROM {R('eicu_lab_sub')}")
    aggs = []
    for x, nm in E_LAB.items():
        aggs += [f"arg_max(v, t) FILTER (WHERE labname='{nm}') AS {x}_last", f"MIN(v) FILTER (WHERE labname='{nm}') AS {x}_min",
                 f"MAX(v) FILTER (WHERE labname='{nm}') AS {x}_max"]
    con.execute(f"""CREATE TABLE lab_feat AS SELECT cf.stay_id, {', '.join(aggs)}
        FROM lab JOIN cf ON cf.stay_id = lab.stay_id AND lab.t < cf.pred_min AND lab.t >= cf.pred_min - {win * 60} GROUP BY cf.stay_id""")
    con.execute(f"""CREATE TABLE sofa_idx AS SELECT cf.stay_id, {SOFA_SEL}
        FROM cf JOIN {R('sofa_hourly_eicu')} s ON s.stay_id = cf.stay_id AND s.hr = cf.index_hr + {k} - 1""")
    return con.execute(f"""SELECT cf.stay_id, cf.index_hr + {k} AS hours_icu_to_pred,
                                  ch.* EXCLUDE (stay_id), lb.* EXCLUDE (stay_id), so.* EXCLUDE (stay_id)
                           FROM cf LEFT JOIN chart_feat ch USING (stay_id) LEFT JOIN lab_feat lb USING (stay_id)
                                LEFT JOIN sofa_idx so USING (stay_id)""").df()


def compare(tag, new, old, cols):
    """returns list of (column, mismatch fraction) beyond tolerance"""
    new, old = clean(new), clean(old)
    m = old[["stay_id"] + cols].merge(new[["stay_id"] + cols], on="stay_id", suffixes=("_old", "_new"))
    bad = []
    for c in cols:
        a, b = m[c + "_old"].astype(float), m[c + "_new"].astype(float)
        eq = (a.isna() & b.isna()) | ((a - b).abs() <= 1e-6 + 1e-6 * a.abs())
        frac = 1 - eq.mean()
        if frac > (0.02 if c.endswith("_last") else 5e-4): bad.append((c, round(float(frac), 4)))
    if len(m) < 0.999 * len(old): bad.append(("row coverage", round(len(m) / len(old), 4)))
    say(f"[{tag}] compared {len(m):,} of {len(old):,} stays on {len(cols)} columns: {'PASSED' if not bad else 'FAILED ' + str(bad[:12])}")
    return bad


# ------------------------------------------------------------------ statistics helpers
def make_boot(clusters, nb, seed):
    rng = np.random.default_rng(seed); codes = pd.factorize(pd.Series(clusters))[0]
    order = np.argsort(codes, kind="stable"); cut = np.flatnonzero(np.diff(codes[order])) + 1
    groups = np.split(order, cut)
    return [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(nb)]


def boot_auc(y, p, idxs):
    out = []
    for i in idxs:
        yy = y[i]
        out.append(roc_auc_score(yy, p[i]) if 0 < yy.sum() < len(yy) else np.nan)
    return np.array(out)


def pct(v): v = v[~np.isnan(v)]; return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def fmt(pt, lo, hi): return f"{pt:.3f} ({lo:.3f}-{hi:.3f})"


def make_learner(seed):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         RandomForestClassifier(n_estimators=TREES, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=seed))


# ------------------------------------------------------------------ data
fm_raw = pd.read_parquet(f"{OUT}/features_mimic.parquet")
fe_raw = pd.read_parquet(f"{OUT}/features_eicu.parquet")
cm = pd.read_parquet(f"{OUT}/cohort_final_mimic.parquet")[["stay_id", "index_time", "outtime"]]
ce = pd.read_parquet(f"{OUT}/cohort_eicu.parquet")[["stay_id", "outoffset"]]
TRAIN_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
DROP = ["stay_id", "subject_id", "hadm_id", "anchor_year_group", "race", "gender", "shock", "hours_to_shock"]
FEATS = [c for c in fm_raw.columns if c not in DROP]
if args.drop_hours: FEATS = [c for c in FEATS if c != "hours_icu_to_pred"]
STAT_M = ["stay_id", "subject_id", "anchor_year_group", "shock", "hours_to_shock", "age", "male"]
STAT_E = ["stay_id", "hospitalid", "shock", "hours_to_shock", "age", "male"]
say(f"MIMIC stays {len(fm_raw):,} | eICU stays {len(fe_raw):,} | {len(FEATS)} features")
cm = cm.set_index("stay_id"); ce = ce.set_index("stay_id")
fe_idx = fe_raw.set_index("stay_id").hours_icu_to_pred - 1

CACHE = {}


def get_table(ds, k, win):
    """predictor table (stay_id + static + FEATS) of dataset ds ('mimic'/'eicu') at landmark k with window win"""
    key = (ds, k, win)
    if key in CACHE: return CACHE[key]
    raw, stat, builder = (fm_raw, STAT_M, build_mimic) if ds == "mimic" else (fe_raw, STAT_E, build_eicu)
    if (k, win) == (1, 24):                                  # the primary table itself
        t = raw[stat + [c for c in FEATS if c not in stat]].copy()
    else:
        p_lad, p = f"{FEAT_DIR}/features_{ds}_k{k}.parquet", f"{WD}/features_{ds}_k{k}_w{win}.parquet"
        if win == 24 and os.path.exists(p_lad): new = pd.read_parquet(p_lad)
        elif os.path.exists(p): new = pd.read_parquet(p)
        else:
            say(f"[{ds} k={k} win={win}] building ..."); new = builder(k, raw, win); new.to_parquet(p)
        t = raw[stat].merge(new, on="stay_id", how="inner")
        for c in FEATS:
            if c not in t.columns: t[c] = np.nan
    t = clean(t); CACHE[key] = t; return t


def verify_builders(H):
    for ds, raw, builder in (("mimic", fm_raw, build_mimic), ("eicu", fe_raw, build_eicu)):
        cols = [c for c in FEATS if c != "hours_icu_to_pred"]
        if H == HORIZONS[0]:
            new = builder(1, raw, 24); cc = [c for c in cols if c in new.columns]
            if compare(f"{ds} k=1 win=24 vs primary features", new, raw, cc): sys.exit("my builder does not reproduce the primary features; stop.")
        p_lad = f"{FEAT_DIR}/features_{ds}_k{H + 1}.parquet"
        if os.path.exists(p_lad):
            new = builder(H + 1, raw, 24); old = pd.read_parquet(p_lad); cc = [c for c in cols if c in new.columns and c in old.columns]
            if compare(f"{ds} k={H + 1} win=24 vs ladder table", new, old, cc): sys.exit("my builder does not reproduce the ladder table; stop.")


# ================================================================== PART A
PRESS = [c for c in ["on_pressor", "s_cv", "sofa_total"] if c in FEATS]
LAC = [c for c in FEATS if c.startswith("lactate_")]
rows, diffs, ns, seed_rows, avail = [], [], [], [], []
t0 = time.time()
for H in ([] if args.only_lead else HORIZONS):
    verify_builders(H)
    kC = H + 1; w = args.short_win
    DES = {"B": (1, 24), "B_wide": (1, 24 + H), "C": (kC, 24), "C_star": (kC, 24 + H)}
    if w > 0: DES.update({"B_short": (1, w), "C_short": (kC, w)})
    FS = {"": FEATS, "-PL": [c for c in FEATS if c not in PRESS + LAC]}
    okm = cm.loc[get_table("mimic", 1, 24).stay_id].pipe(lambda d: (d.outtime.values >= (d.index_time + pd.Timedelta(hours=kC)).values))
    oke = (ce.loc[get_table("eicu", 1, 24).stay_id].outoffset.values >= (fe_idx.loc[get_table("eicu", 1, 24).stay_id].values + kC) * 60)
    keepM, keepE = set(get_table("mimic", 1, 24).stay_id[okm]), set(get_table("eicu", 1, 24).stay_id[oke])
    sets = {}
    for name, (k, win) in DES.items():
        m_, e_ = [get_table(ds, k, win) for ds in ("mimic", "eicu")]
        m_, e_ = m_[m_.stay_id.isin(keepM)].sort_values("stay_id").reset_index(drop=True), e_[e_.stay_id.isin(keepE)].sort_values("stay_id").reset_index(drop=True)
        sets[name] = {"tr": m_[m_.anchor_year_group.isin(TRAIN_GROUPS)].reset_index(drop=True), "te": m_[~m_.anchor_year_group.isin(TRAIN_GROUPS) & (m_.anchor_year_group.isin(TEST_GROUPS) if TEST_GROUPS else True)].reset_index(drop=True), "ev": e_}
    ref = sets["B"]
    for name in sets:
        for part in ("tr", "te", "ev"):
            assert (sets[name][part].stay_id.values == ref[part].stay_id.values).all(), f"stay mismatch: design {name}, {part}"
    Y = {part: ((ref[part].shock == 1) & (ref[part].hours_to_shock <= H)).astype(int).values for part in ("tr", "te", "ev")}
    CL = {"te": ref["te"].subject_id.values, "ev": ref["ev"].hospitalid.values}
    for part, nm in (("tr", "MIMIC development"), ("te", "MIMIC temporal test"), ("ev", "eICU external")):
        ns.append({"H": H, "dataset": nm, "n": len(Y[part]), "events": int(Y[part].sum()), "prevalence": Y[part].mean()})
        say(f"H={H}h {nm:22s} n={len(Y[part]):6,} events={int(Y[part].sum()):5,} ({Y[part].mean():.1%})")
    for name, (k, win) in DES.items():           # how much recent data / extra history each design actually has
        for part, nm in (("te", "MIMIC temporal test"), ("ev", "eICU external")):
            d = sets[name][part]
            avail.append({"H": H, "dataset": nm, "design": name, "k": k, "window_h": win,
                          "median_hours_icu_to_pred": float(d.hours_icu_to_pred.median()),
                          "pct_stays_with_>=window_hours_since_admission": 100 * float((d.hours_icu_to_pred >= win).mean()),
                          "pct_missing_hr_mean": 100 * float(d.hr_mean.isna().mean()) if "hr_mean" in d else np.nan,
                          "pct_missing_lactate_last": 100 * float(d.lactate_last.isna().mean()) if "lactate_last" in d else np.nan,
                          "pct_vasopressor_running_events": 100 * float(d.on_pressor[Y[part] == 1].mean()) if "on_pressor" in d else np.nan})
    P = {"te": {}, "ev": {}}
    for fsn, cols in FS.items():
        for name, (k, win) in DES.items():
            code = name + fsn
            ps_te, ps_ev = [], []
            for sd in SEEDS:
                m = make_learner(sd).fit(sets[name]["tr"][cols], Y["tr"])
                ps_te.append(m.predict_proba(sets[name]["te"][cols])[:, 1]); ps_ev.append(m.predict_proba(sets[name]["ev"][cols])[:, 1])
                seed_rows.append({"H": H, "design": code, "seed": sd, "AUROC_MIMIC_test": roc_auc_score(Y["te"], ps_te[-1]), "AUROC_eICU": roc_auc_score(Y["ev"], ps_ev[-1])})
            P["te"][code], P["ev"][code] = np.mean(ps_te, axis=0), np.mean(ps_ev, axis=0)
            say(f"  H={H} {code:14s} AUROC temporal {roc_auc_score(Y['te'], P['te'][code]):.3f} | eICU {roc_auc_score(Y['ev'], P['ev'][code]):.3f}   [{time.time() - t0:.0f}s]")
    pairs = [("C_star", "B_wide"), ("B_wide", "B"), ("C_star", "B"), ("C", "C_star"), ("C", "B")] + ([("C_short", "B_short")] if w > 0 else [])
    contrasts = [(a + fs, b + fs) for fs in FS for a, b in pairs] + [(d + "-PL", d) for d in DES]
    for part, nm in (("te", "MIMIC temporal test"), ("ev", "eICU external")):
        y = Y[part]; idxs = make_boot(CL[part], B, 1000 + H * 7 + (0 if part == "te" else 1)); bv = {}
        for code in P[part]:
            bv[code] = boot_auc(y, P[part][code], idxs); pt = roc_auc_score(y, P[part][code]); lo, hi = pct(bv[code])
            rows.append({"H": H, "dataset": nm, "code": code, "AUROC": pt, "lo": lo, "hi": hi, "formatted": fmt(pt, lo, hi),
                         "AUPRC_point": average_precision_score(y, P[part][code]), "n": len(y), "events": int(y.sum()), "prevalence": y.mean()})
        for a_, b_ in contrasts:
            d = bv[a_] - bv[b_]; pt = roc_auc_score(y, P[part][a_]) - roc_auc_score(y, P[part][b_]); lo, hi = pct(d)
            diffs.append({"H": H, "dataset": nm, "comparison": f"{a_} minus {b_}", "difference": pt, "lo": lo, "hi": hi, "formatted": fmt(pt, lo, hi)})
    res = pd.DataFrame(rows); df_ = pd.DataFrame(diffs)
    res.to_csv(f"{WD}/window_results.csv", index=False); df_.to_csv(f"{WD}/window_paired_differences.csv", index=False)
    pd.DataFrame(ns).to_csv(f"{WD}/window_design_ns.csv", index=False); pd.DataFrame(seed_rows).to_csv(f"{WD}/window_seeds.csv", index=False)
    pd.DataFrame(avail).to_csv(f"{WD}/window_data_availability.csv", index=False)
    say(f"\n=== H={H}: paired AUROC differences (cluster bootstrap) ===")
    say(df_[df_.H == H][["dataset", "comparison", "formatted"]].to_string(index=False))

# ================================================================== PART B
if not args.skip_lead:
    PRIV = f"{OUT}/results/private"; lt_rows = []
    srcs = [("MIMIC-IV temporal test", f"{PRIV}/predictions_temporal_test.parquet", fm_raw, "subject_id"),
            ("eICU-CRD external", f"{PRIV}/predictions_eicu.parquet", fe_raw, "hospitalid")]
    for ds, path, raw, clcol in srcs:
        if not os.path.exists(path): say(f"[lead time] {path} not found - skipped"); continue
        pr = pd.read_parquet(path)
        need = [c for c in (clcol, "hours_to_shock", "sofa_total", "lactate_last") if c not in pr.columns]
        d = pr.merge(raw[["stay_id"] + need + (["shock"] if "shock" not in pr.columns else [])], on="stay_id", how="left")
        chk = pr[["stay_id", "shock"]].merge(raw[["stay_id", "shock"]], on="stay_id", suffixes=("_pred", "_raw"))
        assert len(chk) == len(pr) and (chk.shock_pred.values == chk.shock_raw.values).all(), "predictions file does not match the feature table labels"
        if ds.startswith("MIMIC"):
            c = cm.reindex(d.stay_id.values); d["exit_h"] = ((c.outtime.values - c.index_time.values) / np.timedelta64(1, "h")).astype(float)
        else:
            d["exit_h"] = ce.reindex(d.stay_id.values).outoffset.values / 60.0 - fe_idx.reindex(d.stay_id.values).values
        assert d.exit_h.notna().all() and d[clcol].notna().all(), "missing exit time or cluster"
        d = clean(d)
        pcols = [c for c in d.columns if c.startswith("p_")]
        S = {c[2:]: d[c].values for c in pcols}; S["SOFA only"] = d.sofa_total.fillna(0).values; S["Lactate only"] = d.lactate_last.fillna(d.lactate_last.median()).values
        shock, hts, ex, cl = d.shock.values.astype(int), d.hours_to_shock.values, d.exit_h.values, d[clcol].values
        say(f"[lead time] {ds}: {len(d):,} stays, {int(shock.sum())} events; models: {list(S)}")
        for g in [0, 1, 2, 3, 6, 12]:
            pos = (shock == 1) & (hts > g); neg = shock == 0
            for mode, keep in (("original", pos | neg), ("symmetric", pos | (neg & (ex > g)))):
                yy = pos[keep].astype(int); idxs = make_boot(cl[keep], B, 5000 + g)
                for nmod, sc in S.items():
                    p = sc[keep]; v = boot_auc(yy, p, idxs); lo, hi = pct(v); pt = roc_auc_score(yy, p)
                    lt_rows.append({"dataset": ds, "g_hours": g, "mode": mode, "model": nmod, "AUROC": pt, "lo": lo, "hi": hi, "formatted": fmt(pt, lo, hi),
                                    "n_pos": int(yy.sum()), "n_neg": int((yy == 0).sum()), "negatives_dropped_vs_original": int((pos | neg).sum() - keep.sum())})
    lt = pd.DataFrame(lt_rows); lt.to_csv(f"{WD}/leadtime_symmetric.csv", index=False)
    if len(lt):
        say("\n=== lead time: RF, original vs symmetric (g = 0 must equal the primary 0.778 / 0.783) ===")
        mm = "RF" if "RF" in set(lt.model) else sorted(set(lt.model))[0]
        say(lt[lt.model == mm][["dataset", "g_hours", "mode", "formatted", "n_pos", "n_neg", "negatives_dropped_vs_original"]].to_string(index=False))
say(f"\nDone in {time.time() - t0:.0f}s. Aggregated outputs in {WD}")
LOG.close()
