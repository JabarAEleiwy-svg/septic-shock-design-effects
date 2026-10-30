#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extended_analysis_v2.py
=======================
Additional analyses for the septic-shock prediction paper (MIMIC-IV -> eICU-CRD).
Revision of extended_analysis.py after a line-by-line audit against ICUmodelprediction.ipynb.

Input  (produced by ICUmodelprediction.ipynb, in --out):
    features_mimic.parquet, features_eicu.parquet                         (required)
    cohort_final_mimic.parquet                                            (optional: mortality, alt. outcome)
    sofa_hourly.parquet, sofa_hourly_eicu.parquet                         (optional: alt. outcome, SOFA audit)
Output (written to --out/results):
    one CSV per analysis + PNG figures + results_summary.txt (+ results_status.json)

Run in Colab (after mounting Drive):
    !python extended_analysis_v2.py --out /content/drive/MyDrive/sepsis_project --nboot 1000

What changed relative to extended_analysis.py (see CHANGES at the end of this docstring):
  FIXES
   1. SOFA-only comparator: missing SOFA -> 0 (exactly as in the notebook that produced 0.495 / 0.652);
      v1 imputed the training median, which would not reproduce the reported numbers.
   2. Reproducibility check: the primary RF must reproduce the point estimates in the manuscript
      (0.778 temporal, 0.783 eICU) before any new number is trusted.
   3. Operating threshold: manuscript v3 picked the 80%-specificity threshold on the TEST set. v2 reports it
      that way (for comparison) AND from out-of-fold predictions in the development period (patient-grouped
      5-fold), which is the legitimate choice.
   4. Patient-level prediction files are NOT written unless --save_preds is given (PhysioNet data-use terms).
   5. Hospital meta-analysis is run with >=10 and >=5 events per hospital and reports how many hospitals qualify.
   6. alt_outcome rewritten without groupby.apply (faster, no pandas-version dependence).
   7. A failed section is listed at the end and in results_status.json instead of being buried in the log.
  NEW ANALYSES
   8.  outcome_composition : how many events already had a vasopressor running at the index hour / lactate >2 before
                             the prediction time (overlap of predictors and outcome definition).
   9.  gap_sensitivity     : performance after removing events with onset <= g hours after the index.
   10. pressor_excluded    : primary outcome restricted to stays with NO vasopressor at the index hour
                             (evaluation-only and retrained).
   11. landmark_strata     : performance by hours from ICU admission to prediction time (eICU median 1 h vs MIMIC 3 h).
   12. sofa_audit          : component-wise SOFA distributions in both databases, imputed-default vasopressor dose share.
   13. qSOFA (approx.) and lactate-only comparators added to every discrimination/DeLong table.

NOTHING here is tuned on the external (eICU) data. eICU labels are used only for evaluation and, where stated, for
cross-fitted recalibration that is reported as an illustration, not as a result of the original model.
"""
import argparse, os, sys, warnings, json, traceback, platform, importlib
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss
from sklearn.model_selection import GroupKFold
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/content/drive/MyDrive/sepsis_project")
ap.add_argument("--nboot", type=int, default=1000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--shap_n", type=int, default=1500)
ap.add_argument("--rf_trees", type=int, default=500, help="500 reproduces the manuscript; lower only for dry runs")
ap.add_argument("--gap_hours", default="1,2,3,6,12")
ap.add_argument("--save_preds", action="store_true", help="write patient-level predictions (do NOT upload them anywhere)")
ap.add_argument("--skip", default="", help="comma-separated section names to skip, e.g. shap,alt_outcome")
ap.add_argument("--only", default="", help="comma-separated section names to run (all others skipped); reproducibility_check always runs")
args = ap.parse_args()
OUT, RES = args.out, os.path.join(args.out, "results")
os.makedirs(RES, exist_ok=True)
B, SEED = args.nboot, args.seed
SKIP = set(s.strip() for s in args.skip.split(",") if s.strip())
ONLY = set(s.strip() for s in args.only.split(",") if s.strip())
LOG = open(os.path.join(RES, "results_summary.txt"), "w")
STATUS = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


def section(name):
    def deco(fn):
        def run(*a, **k):
            if name in SKIP or (ONLY and name not in ONLY and name != "reproducibility_check"):
                say(f"\n[skipped] {name}"); STATUS[name] = "skipped"; return None
            say(f"\n=== {name} ===")
            STATUS.pop(name, None)
            try:
                r = fn(*a, **k); STATUS.setdefault(name, "ok"); return r
            except Exception as e:
                say(f"[FAILED] {name}: {type(e).__name__}: {e}"); traceback.print_exc(); STATUS[name] = f"FAILED: {type(e).__name__}: {e}"; return None
        return run
    return deco


say("python", platform.python_version())
for _m in ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "lightgbm", "xgboost", "shap", "duckdb"]:
    try:
        say(f"  {_m} {importlib.import_module(_m).__version__}")
    except Exception:
        say(f"  {_m} not installed")
say(f"args: nboot={B} seed={SEED} rf_trees={args.rf_trees} shap_n={args.shap_n}")

# ------------------------------------------------------------------ data
LIM = {"hr": (0, 300), "rr": (0, 100), "spo2": (0, 100), "sbp": (0, 350), "dbp": (0, 250), "map": (0, 300), "temp": (25, 45)}


def clean(d):                                   # identical to the notebook (cells 30/46/47)
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


def race_group(s):
    s = str(s).lower()
    if "white" in s or "caucasian" in s: return "White"
    if "black" in s or "african" in s: return "Black"
    if "hispanic" in s or "latino" in s: return "Hispanic"
    if "asian" in s: return "Asian"
    return "Other/unknown"


mi = clean(pd.read_parquet(f"{OUT}/features_mimic.parquet"))
ev = clean(pd.read_parquet(f"{OUT}/features_eicu.parquet"))
TRAIN_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
tr = mi[mi.anchor_year_group.isin(TRAIN_GROUPS)].reset_index(drop=True)
te = mi[~mi.anchor_year_group.isin(TRAIN_GROUPS)].reset_index(drop=True)
ev = ev.reset_index(drop=True)
DROP = ["stay_id", "subject_id", "hadm_id", "anchor_year_group", "race", "gender", "shock", "hours_to_shock"]
FEATS = [c for c in mi.columns if c not in DROP]
for c in FEATS:
    if c not in ev.columns:
        ev[c] = np.nan
say(f"train {len(tr):,} (shock {tr.shock.mean():.3%}) | temporal test {len(te):,} ({te.shock.mean():.3%}) | "
    f"eICU {len(ev):,} ({ev.shock.mean():.3%}, {ev.hospitalid.nunique()} hospitals) | {len(FEATS)} features")
if set(tr.subject_id) & set(te.subject_id):
    say("WARNING: patients appear in both the development and the temporal test set")

y_tr, y_te, y_ev = tr.shock.values.astype(int), te.shock.values.astype(int), ev.shock.values.astype(int)
CL_TR, CL_TE, CL_EV = tr.subject_id.values, te.subject_id.values, ev.hospitalid.values

SOFA_COLS = [c for c in ["sofa_total", "s_resp", "s_coag", "s_liver", "s_cv", "s_cns", "s_renal", "gcs_min", "pf_min", "on_pressor"] if c in FEATS]
LAC_COLS = [c for c in FEATS if c.startswith("lactate_")]
TEMP_COLS = [c for c in FEATS if c.startswith("temp_")]


# ------------------------------------------------------------------ models
def make_model(kind, indicator=True):
    if kind == "RF":     # identical to the notebook (n_estimators=500)
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=indicator),
                             RandomForestClassifier(n_estimators=args.rf_trees, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42))
    if kind == "LR":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=indicator), StandardScaler(),
                             LogisticRegression(max_iter=3000, C=1.0))
    if kind == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
    if kind == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                 tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=-1)
    raise ValueError(kind)


def fit_predict(kind, cols, train, tests, indicator=True, label="shock"):
    m = make_model(kind, indicator)
    m.fit(train[cols], train[label])
    return m, [m.predict_proba(t[cols])[:, 1] for t in tests]


PRED_TE, PRED_EV, MODELS = {}, {}, {}
for kind in ["RF", "LR", "LightGBM", "XGBoost"]:
    try:
        m, (a, b) = fit_predict(kind, FEATS, tr, [te, ev])
        MODELS[kind], PRED_TE[kind], PRED_EV[kind] = m, a, b
        say(f"{kind:9s} AUROC temporal {roc_auc_score(y_te, a):.3f} | eICU {roc_auc_score(y_ev, b):.3f}")
    except Exception as e:
        say(f"[model {kind} failed] {e}")

# ---- reproducibility check against the manuscript (Section 2.8: point estimates on the full test sets)
if "RF" in PRED_TE:
    r_te, r_ev = roc_auc_score(y_te, PRED_TE["RF"]), roc_auc_score(y_ev, PRED_EV["RF"])
    ok = abs(r_te - 0.778) <= 0.005 and abs(r_ev - 0.783) <= 0.005
    say(f"REPRODUCIBILITY: RF AUROC temporal {r_te:.3f} (manuscript 0.778), eICU {r_ev:.3f} (manuscript 0.783) -> "
        + ("OK" if ok else "MISMATCH: do NOT use the numbers from this run until the difference is explained "
                           "(different features parquet? different clean()? --rf_trees != 500?)"))
    STATUS["reproducibility_check"] = "ok" if ok else "MISMATCH"

# ---- comparators (no fitting on external data)
COMPARATORS = []
med = tr[FEATS].median()
if "sofa_total" in FEATS:      # missing SOFA -> 0, as in the notebook
    PRED_TE["SOFA only"], PRED_EV["SOFA only"] = te.sofa_total.fillna(0).values.astype(float), ev.sofa_total.fillna(0).values.astype(float)
    COMPARATORS.append("SOFA only")
if "lactate_last" in FEATS:
    PRED_TE["Lactate only"], PRED_EV["Lactate only"] = te.lactate_last.fillna(med.lactate_last).values.astype(float), ev.lactate_last.fillna(med.lactate_last).values.astype(float)
    COMPARATORS.append("Lactate only")
if all(c in FEATS for c in ("rr_max", "sbp_min", "gcs_min")):     # qSOFA from 24-h worst values (approximation of the bedside score)
    q = lambda d: ((d.rr_max >= 22).astype(int) + (d.sbp_min <= 100).astype(int) + (d.gcs_min < 15).astype(int)).values.astype(float)
    PRED_TE["qSOFA (approx.)"], PRED_EV["qSOFA (approx.)"] = q(te), q(ev)
    COMPARATORS.append("qSOFA (approx.)")
core = [c for c in ["age", "male", "sofa_total", "lactate_last", "map_min", "creat_last", "on_pressor"] if c in FEATS]
try:
    m, (a, b) = fit_predict("LR", core, tr, [te, ev])
    MODELS["Core LR"], PRED_TE["Core LR"], PRED_EV["Core LR"] = m, a, b
    say(f"Core LR ({', '.join(core)}): temporal {roc_auc_score(y_te, a):.3f} | eICU {roc_auc_score(y_ev, b):.3f}")
except Exception as e:
    say(f"[core LR failed] {e}")

if args.save_preds:
    PRIV = os.path.join(RES, "private"); os.makedirs(PRIV, exist_ok=True)
    pd.DataFrame({"stay_id": te.stay_id, "shock": y_te, **{f"p_{k}": v for k, v in PRED_TE.items()}}).to_parquet(f"{PRIV}/predictions_temporal_test.parquet")
    pd.DataFrame({"stay_id": ev.stay_id, "hospitalid": ev.hospitalid, "shock": y_ev, **{f"p_{k}": v for k, v in PRED_EV.items()}}).to_parquet(f"{PRIV}/predictions_eicu.parquet")
    say(f"patient-level predictions written to {PRIV} -- keep private (PhysioNet data-use agreement)")
PROB_MODELS = [k for k in ["RF", "LR", "LightGBM", "XGBoost", "Core LR"] if k in PRED_EV]
ALL = [k for k in PRED_EV.keys()]


# ------------------------------------------------------------------ bootstrap machinery
def make_boot(clusters, B, seed):
    rng = np.random.default_rng(seed)
    codes = pd.factorize(pd.Series(clusters))[0]
    groups = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    G = len(groups)
    return [np.concatenate([groups[g] for g in rng.integers(0, G, G)]) for _ in range(B)]


def naive_boot(n, B, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(B)]


BOOT = {"te_cluster": make_boot(CL_TE, B, SEED), "ev_cluster": make_boot(CL_EV, B, SEED + 1),
        "te_naive": naive_boot(len(te), B, SEED + 2), "ev_naive": naive_boot(len(ev), B, SEED + 3)}


def boot_metric(y, p, idxs, fn):
    v = []
    for i in idxs:
        yy = y[i]
        if yy.sum() == 0 or yy.sum() == len(yy): v.append(np.nan); continue
        v.append(fn(yy, p[i]))
    return np.array(v)


def ci(v):
    v = np.asarray(v, dtype=float); v = v[~np.isnan(v)]
    if len(v) == 0: return np.nan, np.nan
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def fmt(point, lo, hi, d=3):
    return f"{point:.{d}f} ({lo:.{d}f}-{hi:.{d}f})"


def sens_at_spec(y, p, spec):
    fpr, tpr, _ = roc_curve(y, p)
    return tpr[np.searchsorted(fpr, 1 - spec, side="right") - 1]


def cluster_ci(y, p, clusters, fn, nb=None, seed=0):
    """cluster-bootstrap percentile CI of fn(y, p) for an arbitrary subset (y, p, clusters are aligned arrays)"""
    nb = min(B, 500) if nb is None else nb
    if y.sum() < 2 or y.sum() == len(y): return "NA"
    codes = pd.factorize(pd.Series(clusters))[0]
    groups = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    rng = np.random.default_rng(seed); vals = []
    for _ in range(nb):
        s = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if 0 < y[s].sum() < len(s): vals.append(fn(y[s], p[s]))
    pt = fn(y, p)
    return fmt(pt, *ci(vals)) if vals else f"{pt:.3f} (NA)"


# ------------------------------------------------------------------ 1. discrimination table
@section("discrimination")
def discrimination():
    rows, deltas, store = [], [], {}
    for ds, y, P, key in [("MIMIC-IV temporal test", y_te, PRED_TE, "te"), ("eICU-CRD external", y_ev, PRED_EV, "ev")]:
        for k in ALL:
            p = P[k]
            r = {"dataset": ds, "model": k, "n": len(y), "events": int(y.sum()), "prevalence": y.mean()}
            for nm, fn in [("AUROC", roc_auc_score), ("AUPRC", average_precision_score),
                           ("Sens@80%spec", lambda a, b: sens_at_spec(a, b, .8)), ("Sens@90%spec", lambda a, b: sens_at_spec(a, b, .9))]:
                pt = fn(y, p)
                vc = boot_metric(y, p, BOOT[key + "_cluster"], fn)
                r[nm] = pt; r[nm + "_CI_cluster"] = fmt(pt, *ci(vc))
                if nm in ("AUROC", "AUPRC"):
                    vn = boot_metric(y, p, BOOT[key + "_naive"], fn); r[nm + "_CI_naive"] = fmt(pt, *ci(vn))
                    store[(ds, k, nm)] = vc
            rows.append(r)
        for k in ALL:           # paired cluster-bootstrap differences RF - comparator (same resamples for both models)
            if k == "RF": continue
            for nm in ("AUROC", "AUPRC"):
                d = store[(ds, "RF", nm)] - store[(ds, k, nm)]
                fn = roc_auc_score if nm == "AUROC" else average_precision_score
                pt = fn(y, P["RF"]) - fn(y, P[k]); lo, hi = ci(d)
                deltas.append({"dataset": ds, "metric": nm, "comparison": f"RF - {k}", "difference": pt, "CI_cluster": fmt(pt, lo, hi)})
    pd.DataFrame(rows).to_csv(f"{RES}/discrimination_all_models.csv", index=False)
    pd.DataFrame(deltas).to_csv(f"{RES}/discrimination_paired_differences.csv", index=False)
    say(pd.DataFrame(rows)[["dataset", "model", "AUROC", "AUROC_CI_cluster", "AUROC_CI_naive", "AUPRC_CI_cluster"]].to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, (ds, y, P) in zip(ax, [("MIMIC-IV temporal test", y_te, PRED_TE), ("eICU-CRD external", y_ev, PRED_EV)]):
        for k in ALL:
            f, t, _ = roc_curve(y, P[k]); a.plot(f, t, label=f"{k} ({roc_auc_score(y, P[k]):.3f})", ls="--" if k in COMPARATORS else "-")
        a.plot([0, 1], [0, 1], ":", c="grey"); a.set_title(ds); a.set_xlabel("1 - Specificity"); a.set_ylabel("Sensitivity"); a.legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{RES}/fig_roc_all_models.png", dpi=300); plt.close()


# ------------------------------------------------------------------ 2. DeLong
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1); i = j
    T2 = np.empty(N); T2[J] = T + 1
    return T2


def _fast_delong(preds_sorted, m):
    k, n = preds_sorted.shape; n_neg = n - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    tx, ty, tz = (np.empty((k, m)), np.empty((k, n_neg)), np.empty((k, n)))
    for r in range(k):
        tx[r], ty[r], tz[r] = _midrank(pos[r]), _midrank(neg[r]), _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n_neg - (m + 1.0) / 2.0 / n_neg
    v01 = (tz[:, :m] - tx) / n_neg; v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n_neg
    return aucs, cov


def delong_p(y, p1, p2):
    order = np.argsort(-y, kind="stable"); m = int(y.sum())
    aucs, cov = _fast_delong(np.vstack([p1, p2])[:, order], m)
    l = np.array([1, -1]); var = l @ np.atleast_2d(cov) @ l
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else np.nan
    return aucs[0], aucs[1], 2 * stats.norm.sf(abs(z))


@section("delong")
def delong():
    rows = []
    for ds, y, P in [("MIMIC-IV temporal test", y_te, PRED_TE), ("eICU-CRD external", y_ev, PRED_EV)]:
        for k in ALL:
            if k == "RF": continue
            a1, a2, p = delong_p(y, P["RF"], P[k])
            rows.append({"dataset": ds, "comparison": f"RF vs {k}", "AUROC_RF": a1, "AUROC_other": a2, "p_raw": p})
    d = pd.DataFrame(rows); d["p_holm"] = np.nan
    for ds in d.dataset.unique():
        msk = d.dataset == ds
        d.loc[msk, "p_holm"] = multipletests(d.loc[msk, "p_raw"], method="holm")[1]
    d.to_csv(f"{RES}/delong_paired_tests.csv", index=False)
    say(d.to_string(index=False))
    say("NOTE: DeLong treats stays as independent. For eICU-CRD (hospital clustering) the cluster-bootstrap differences in "
        "discrimination_paired_differences.csv are the primary inference.")


# ------------------------------------------------------------------ 3. calibration
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def calib_stats(y, p, bins=10):
    lp = logit(p)
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    df = pd.DataFrame({"y": y, "p": p, "b": q})
    g = df.groupby("b").agg(obs=("y", "mean"), pred=("p", "mean"), n=("y", "size"))
    ece = float((g.n * (g.obs - g.pred).abs()).sum() / g.n.sum())
    return {"Brier": brier_score_loss(y, p), "CITL": citl, "slope": slope, "ECE": ece, "O:E": y.mean() / p.mean()}, g


def crossfit_recal(y, p, groups, k=5):
    out = np.zeros(len(y)); X = logit(p).reshape(-1, 1)
    for a, b in GroupKFold(n_splits=k).split(X, y, groups):
        out[b] = LogisticRegression(C=1e6, max_iter=1000).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return out


RECAL = {}


@section("calibration")
def calibration():
    rows, curves = [], []
    for ds, y, P, key, cl in [("MIMIC-IV temporal test", y_te, PRED_TE, "te", CL_TE), ("eICU-CRD external", y_ev, PRED_EV, "ev", CL_EV)]:
        null_brier = y.mean() * (1 - y.mean())
        for k in PROB_MODELS:
            p = P[k]
            s, g = calib_stats(y, p)
            pr = crossfit_recal(y, p, cl)
            RECAL[(ds, k)] = pr
            s2, _ = calib_stats(y, pr)
            r = {"dataset": ds, "model": k, "Brier_null_model(prevalence only)": null_brier, **{f"{a}": v for a, v in s.items()},
                 **{f"{a} (cross-fitted recalibrated)": v for a, v in s2.items()}}
            if k == "RF":
                vals = {a: [] for a in s}
                for i in BOOT[key + "_cluster"][:min(B, 500)]:
                    if y[i].sum() < 5: continue
                    try:
                        bs, _ = calib_stats(y[i], p[i])
                        for a in s: vals[a].append(bs[a])
                    except Exception:
                        pass
                for a in s:
                    r[a + "_CI"] = fmt(s[a], *ci(np.array(vals[a])), d=3)
            rows.append(r)
            g["dataset"], g["model"] = ds, k; curves.append(g.reset_index())
    pd.DataFrame(rows).to_csv(f"{RES}/calibration_metrics.csv", index=False)
    pd.concat(curves).to_csv(f"{RES}/calibration_curves.csv", index=False)
    say(pd.DataFrame(rows)[["dataset", "model", "Brier", "Brier_null_model(prevalence only)", "CITL", "slope", "ECE", "O:E"]].round(3).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.8))
    for a, ds in zip(ax, ["MIMIC-IV temporal test", "eICU-CRD external"]):
        for k in ["RF", "LR", "LightGBM", "XGBoost"]:
            c = pd.concat(curves); c = c[(c.dataset == ds) & (c.model == k)]
            if len(c): a.plot(c.pred, c.obs, marker="o", ms=3, label=k)
        a.plot([0, 1], [0, 1], ":", c="grey"); a.set_xlabel("Mean predicted probability"); a.set_ylabel("Observed proportion"); a.set_title(ds); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{RES}/fig_calibration.png", dpi=300); plt.close()


# ------------------------------------------------------------------ 4. decision curves
@section("decision_curve")
def decision_curve():
    ts = np.arange(0.02, 0.51, 0.01); rows = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, (ds, y) in zip(ax, [("MIMIC-IV temporal test", y_te), ("eICU-CRD external", y_ev)]):
        n, prev = len(y), y.mean()
        a.plot(ts, [prev - (1 - prev) * t / (1 - t) for t in ts], c="grey", label="Treat all"); a.axhline(0, c="k", lw=.8, label="Treat none")
        for k in ["RF", "Core LR"]:
            if (ds, k) not in RECAL: continue
            for lab, p in [(f"{k} (raw)", (PRED_TE if "MIMIC" in ds else PRED_EV)[k]), (f"{k} (recalibrated)", RECAL[(ds, k)])]:
                nb = []
                for t in ts:
                    pred = p >= t; tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
                    nb.append(tp / n - fp / n * t / (1 - t))
                rows += [{"dataset": ds, "model": lab, "threshold": t, "net_benefit": v} for t, v in zip(ts, nb)]
                a.plot(ts, nb, label=lab, ls="--" if "raw" in lab else "-")
        a.set_ylim(-0.02, max(prev, .05) * 1.1); a.set_xlabel("Threshold probability"); a.set_ylabel("Net benefit"); a.set_title(ds); a.legend(fontsize=7)
    pd.DataFrame(rows).to_csv(f"{RES}/decision_curve.csv", index=False)
    plt.tight_layout(); plt.savefig(f"{RES}/fig_decision_curve.png", dpi=300); plt.close()
    say("saved decision_curve.csv / fig_decision_curve.png (raw RF probabilities are class-weighted and not calibrated: "
        "interpret the raw curve with care; the recalibrated curve is cross-fitted by patient/hospital)")


# ------------------------------------------------------------------ 5. operating points (test-derived vs out-of-fold-derived threshold)
_OOF = {}


def get_oof():
    if "p" not in _OOF:
        oof = np.full(len(tr), np.nan)
        for a, b in GroupKFold(n_splits=5).split(tr[FEATS], y_tr, CL_TR):
            m = make_model("RF"); m.fit(tr.iloc[a][FEATS], y_tr[a]); oof[b] = m.predict_proba(tr.iloc[b][FEATS])[:, 1]
        _OOF["p"] = oof
    return _OOF["p"]


def thr_at_spec(y, p, spec):
    fpr, tpr, thr = roc_curve(y, p)
    return thr[np.searchsorted(fpr, 1 - spec, side="right") - 1]


def op_metrics(yy, pp, t):
    pr = pp >= t; tp = (pr & (yy == 1)).sum(); fp = (pr & (yy == 0)).sum(); fn = ((~pr) & (yy == 1)).sum(); tn = ((~pr) & (yy == 0)).sum()
    return np.array([tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1) if (tp + fp) else np.nan, tn / max(tn + fn, 1), pr.mean()])


@section("operating_points")
def operating_points():
    rows = []
    oof = get_oof()
    say(f"out-of-fold AUROC in the development period: {roc_auc_score(y_tr, oof):.3f}")
    # TEMPORAL threshold: model A trained on 2008-2013, threshold chosen on the later development years (2014-2016, never used to fit model A),
    # then model A + that threshold are applied unchanged to the temporal test period and to eICU.  This mimics prospective deployment.
    early = tr.anchor_year_group.isin(["2008 - 2010", "2011 - 2013"]).values; late = ~early
    mA = make_model("RF"); mA.fit(tr[early][FEATS], y_tr[early])
    pA_late, pA_te, pA_ev = (mA.predict_proba(d[FEATS])[:, 1] for d in (tr[late], te, ev))
    say(f"temporal-threshold model A: trained on 2008-2013 (n={early.sum():,}), threshold set on 2014-2016 (n={late.sum():,}, events={int(y_tr[late].sum())}); "
        f"AUROC on 2014-2016 {roc_auc_score(y_tr[late], pA_late):.3f} | temporal test {roc_auc_score(y_te, pA_te):.3f} | eICU {roc_auc_score(y_ev, pA_ev):.3f}")
    for spec in (0.80, 0.90):
        for src, t, pte, pev in [("TEST-derived (as in manuscript v3; optimistic)", thr_at_spec(y_te, PRED_TE["RF"], spec), PRED_TE["RF"], PRED_EV["RF"]),
                                 ("OUT-OF-FOLD development-derived (random folds)", thr_at_spec(y_tr, oof, spec), PRED_TE["RF"], PRED_EV["RF"]),
                                 ("TEMPORAL development-derived (model A, 2008-2013 -> 2014-2016; recommended)", thr_at_spec(y_tr[late], pA_late, spec), pA_te, pA_ev)]:
            for ds, y, p, key in [("MIMIC-IV temporal test", y_te, pte, "te"), ("eICU-CRD external", y_ev, pev, "ev")]:
                pt = op_metrics(y, p, t)
                bs = np.array([op_metrics(y[i], p[i], t) for i in BOOT[key + "_cluster"][:min(B, 500)]])
                r = {"nominal_specificity": spec, "threshold_source": src, "threshold": t, "dataset": ds}
                for j, nm in enumerate(["Sensitivity", "Specificity", "PPV", "NPV", "Alert rate"]):
                    r[nm] = fmt(pt[j], *np.nanpercentile(bs[:, j], [2.5, 97.5]))
                r["Alerts per true positive (1/PPV)"] = 1 / pt[2] if pt[2] and pt[2] > 0 else np.nan
                rows.append(r)
    pd.DataFrame(rows).to_csv(f"{RES}/operating_points.csv", index=False)
    say(pd.DataFrame(rows).to_string(index=False))


# ------------------------------------------------------------------ 6. horizon-specific discrimination
@section("horizon")
def horizon():
    rows = []
    for ds, d, P in [("MIMIC-IV temporal test", te, PRED_TE), ("eICU-CRD external", ev, PRED_EV)]:
        for H in (6, 12, 24, 48):
            keep = ((d.shock == 0) | (d.hours_to_shock <= H)).values       # later events removed (label ambiguous at this horizon)
            yy = d.shock.values[keep].astype(int)
            if yy.sum() < 10: continue
            for k in ["RF"] + COMPARATORS:
                if k in P:
                    rows.append({"dataset": ds, "horizon_h_after_index": H, "model": k, "events": int(yy.sum()), "n": len(yy), "AUROC": roc_auc_score(yy, P[k][keep])})
    pd.DataFrame(rows).to_csv(f"{RES}/discrimination_by_horizon.csv", index=False)
    say(pd.DataFrame(rows).round(3).to_string(index=False))


# ------------------------------------------------------------------ 7. hospital heterogeneity (eICU)
def hanley_mcneil_se(a, n1, n0):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))


def hospital_meta(k, min_events, min_nonevents=10, plot=False):
    rows = []
    for h, g in ev.assign(p=PRED_EV[k]).groupby("hospitalid"):
        n1, n0 = int(g.shock.sum()), int((g.shock == 0).sum())
        if n1 >= min_events and n0 >= min_nonevents:
            a = roc_auc_score(g.shock, g.p)
            if 0 < a < 1:
                rows.append({"hospitalid": h, "n": len(g), "events": n1, "AUROC": a, "se": hanley_mcneil_se(a, n1, n0)})
    d = pd.DataFrame(rows)
    if len(d) < 3:
        say(f"{k}: fewer than 3 hospitals with >= {min_events} events ({len(d)}); meta-analysis not done"); return None
    yv = np.log(d.AUROC / (1 - d.AUROC)); sev = d.se / (d.AUROC * (1 - d.AUROC)); w = 1 / sev ** 2
    mu_fe = (w * yv).sum() / w.sum(); Q = (w * (yv - mu_fe) ** 2).sum(); kk = len(d)
    tau2 = max(0, (Q - (kk - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (sev ** 2 + tau2); mu = (wr * yv).sum() / wr.sum(); se_mu = np.sqrt(1 / wr.sum())
    I2 = max(0, (Q - (kk - 1)) / Q) * 100 if Q > 0 else 0
    tcrit = stats.t.ppf(0.975, max(kk - 2, 1)); pi = tcrit * np.sqrt(tau2 + se_mu ** 2)
    inv = lambda x: 1 / (1 + np.exp(-x))
    res = {"model": k, "min_events_per_hospital": min_events, "hospitals_included": kk, "events_included": int(d.events.sum()), "pooled_AUROC": inv(mu),
           "pooled_95CI": f"{inv(mu-1.96*se_mu):.3f}-{inv(mu+1.96*se_mu):.3f}", "tau2_logit": tau2, "I2_percent": I2,
           "prediction_interval_95": f"{inv(mu-pi):.3f}-{inv(mu+pi):.3f}", "median_hospital_AUROC": d.AUROC.median(),
           "IQR_hospital_AUROC": f"{d.AUROC.quantile(.25):.3f}-{d.AUROC.quantile(.75):.3f}"}
    d.to_csv(f"{RES}/hospital_auroc_{k.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')}_min{min_events}.csv", index=False)
    if plot:
        dd = d.sort_values("AUROC").reset_index(drop=True)
        plt.figure(figsize=(6, max(3, .25 * len(dd) + 1)))
        plt.errorbar(dd.AUROC, range(len(dd)), xerr=1.96 * dd.se, fmt="o", ms=3, c="C0", ecolor="C0", lw=1)
        plt.axvline(res["pooled_AUROC"], c="r", ls="--", label=f"Pooled {res['pooled_AUROC']:.3f}")
        plt.yticks(range(len(dd)), [f"H{int(h)} (n={n}, e={e})" for h, n, e in zip(dd.hospitalid, dd.n, dd.events)], fontsize=6)
        plt.xlabel("AUROC (95% CI)"); plt.legend(); plt.tight_layout(); plt.savefig(f"{RES}/fig_hospital_forest_{k}_min{min_events}.png", dpi=300); plt.close()
    return res


@section("hospital_heterogeneity")
def hospital_heterogeneity():
    ev_per_h = ev.groupby("hospitalid").shock.agg(["sum", "size"])
    say(f"eICU hospitals: {len(ev_per_h)} | with >=1 event {int((ev_per_h['sum'] >= 1).sum())} | >=5 events {int((ev_per_h['sum'] >= 5).sum())} | "
        f">=10 events {int((ev_per_h['sum'] >= 10).sum())} | median events per hospital {ev_per_h['sum'].median():.0f}")
    out = []
    for me in (10, 5):
        for k in ["RF"] + [c for c in COMPARATORS if c in PRED_EV]:
            r = hospital_meta(k, me, plot=(k == "RF"))
            if r: out.append(r)
    pd.DataFrame(out).to_csv(f"{RES}/hospital_heterogeneity_meta.csv", index=False)
    if out: say(pd.DataFrame(out).to_string(index=False))


# ------------------------------------------------------------------ 8. dataset shift & missingness
def smd(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2: return np.nan
    s = np.sqrt((a.var() + b.var()) / 2); return (a.mean() - b.mean()) / s if s > 0 else np.nan


@section("covariate_shift")
def covariate_shift():
    rows = []
    for c in FEATS:
        a, b, t = tr[c], te[c], ev[c]
        ks = stats.ks_2samp(a.dropna(), t.dropna()).statistic if a.notna().sum() > 2 and t.notna().sum() > 2 else np.nan
        ks2 = stats.ks_2samp(a.dropna(), b.dropna()).statistic if a.notna().sum() > 2 and b.notna().sum() > 2 else np.nan
        rows.append({"feature": c, "missing_train_%": 100 * a.isna().mean(), "missing_temporal_test_%": 100 * b.isna().mean(), "missing_eICU_%": 100 * t.isna().mean(),
                     "median_train": a.median(), "median_eICU": t.median(),
                     "SMD_train_vs_temporal_test": smd(a, b), "KS_train_vs_temporal_test": ks2, "SMD_train_vs_eICU": smd(a, t), "KS_train_vs_eICU": ks})
    d = pd.DataFrame(rows); d["abs_SMD_eICU"] = d.SMD_train_vs_eICU.abs()
    d.sort_values("abs_SMD_eICU", ascending=False).drop(columns="abs_SMD_eICU").to_csv(f"{RES}/covariate_shift.csv", index=False)
    say(d.sort_values("abs_SMD_eICU", ascending=False).head(15)[["feature", "missing_train_%", "missing_eICU_%", "median_train", "median_eICU", "SMD_train_vs_eICU", "KS_train_vs_eICU"]].round(2).to_string(index=False))


# ------------------------------------------------------------------ 9. Table 1 by outcome
@section("table1")
def table1():
    p = f"{OUT}/cohort_final_mimic.parquet"
    mi2 = mi.copy(); ev2 = ev.copy()
    if os.path.exists(p):
        co = pd.read_parquet(p)[["stay_id", "hospital_expire_flag"]]
        mi2 = mi2.merge(co, on="stay_id", how="left"); mi2["died"] = mi2.hospital_expire_flag.astype(float)
    if "hospitaldischargestatus" in ev2.columns:
        ev2["died"] = (ev2.hospitaldischargestatus == "Expired").astype(float)
    cont = [c for c in ["age", "hours_icu_to_pred", "sofa_total", "s_cv", "gcs_min", "pf_min", "lactate_last", "lactate_max", "map_min", "sbp_mean", "hr_mean", "rr_mean",
                        "spo2_mean", "creat_last", "bili_last", "plt_last", "wbc_last", "k_last", "bicarb_last", "ph_last", "pao2_last", "base_excess_last"] if c in FEATS]
    binv = [c for c in ["male", "on_pressor"] if c in FEATS] + ["died"]
    rows = []
    for ds, d in [("MIMIC-IV", mi2), ("eICU-CRD", ev2)]:
        n1, n0 = int((d.shock == 1).sum()), int((d.shock == 0).sum())
        rows.append({"cohort": ds, "variable": "N", "no_shock": f"{n0:,}", "shock": f"{n1:,}", "SMD": "", "p": ""})
        for c in cont:
            a, b = d.loc[d.shock == 0, c], d.loc[d.shock == 1, c]
            q = lambda s: f"{s.median():.1f} ({s.quantile(.25):.1f}-{s.quantile(.75):.1f})" if s.notna().any() else "NA"
            pv = stats.mannwhitneyu(a.dropna(), b.dropna()).pvalue if a.notna().sum() and b.notna().sum() else np.nan
            rows.append({"cohort": ds, "variable": c, "no_shock": q(a), "shock": q(b), "SMD": round(smd(b, a), 2), "p": f"{pv:.2g}", "missing_%": round(100 * d[c].isna().mean(), 1)})
        for c in binv:
            if c not in d.columns or d[c].isna().all(): continue
            a, b = d.loc[d.shock == 0, c], d.loc[d.shock == 1, c]
            tab = pd.crosstab(d.shock, d[c].fillna(-1))
            pv = stats.chi2_contingency(tab)[1] if tab.shape[1] > 1 else np.nan
            rows.append({"cohort": ds, "variable": c + " (%)", "no_shock": f"{100*a.mean():.1f}", "shock": f"{100*b.mean():.1f}", "SMD": round(smd(b, a), 2), "p": f"{pv:.2g}"})
    pd.DataFrame(rows).to_csv(f"{RES}/table1_by_outcome.csv", index=False)
    say(f"saved table1_by_outcome.csv ({len(rows)} rows)")


# ------------------------------------------------------------------ 10. subgroups
@section("subgroups")
def subgroups():
    rows = []
    for ds, d, P, CL in [("MIMIC-IV temporal test", te, PRED_TE, CL_TE), ("eICU-CRD external", ev, PRED_EV, CL_EV)]:
        d = d.assign(race_grp=d.race.map(race_group), age_grp=np.where(d.age >= 65, ">=65", "<65"), sex=np.where(d.male == 1, "Male", "Female"))
        for var in ["sex", "age_grp", "race_grp"]:
            for lvl, g in d.groupby(var):
                idx = g.index.values; yy = d.shock.values[idx].astype(int)
                if yy.sum() < 10 or (1 - yy).sum() < 10: continue
                p = P["RF"][idx]
                rows.append({"dataset": ds, "subgroup": f"{var}={lvl}", "n": len(idx), "events": int(yy.sum()),
                             "AUROC": cluster_ci(yy, p, CL[idx], roc_auc_score), "O:E (uncalibrated RF)": yy.mean() / p.mean()})
    pd.DataFrame(rows).to_csv(f"{RES}/subgroup_performance.csv", index=False)
    say(pd.DataFrame(rows).to_string(index=False))


# ------------------------------------------------------------------ 11. ablations
@section("ablations")
def ablations():
    sets = {"Primary (all features)": (FEATS, True),
            "No temperature": ([c for c in FEATS if c not in TEMP_COLS], True),
            "No missing-value indicators": (FEATS, False),
            "No temperature, no missing indicators": ([c for c in FEATS if c not in TEMP_COLS], False),
            "No lactate": ([c for c in FEATS if c not in LAC_COLS], True),
            "No SOFA/organ-score columns": ([c for c in FEATS if c not in SOFA_COLS], True),
            "No lactate and no SOFA columns": ([c for c in FEATS if c not in LAC_COLS + SOFA_COLS], True),
            "No time-since-admission": ([c for c in FEATS if c != "hours_icu_to_pred"], True)}
    rows = []
    for nm, (cols, ind) in sets.items():
        m, (a, b) = fit_predict("RF", cols, tr, [te, ev], indicator=ind)
        r = {"feature set": nm, "n_features": len(cols)}
        for ds, y, p, key in [("temporal", y_te, a, "te"), ("eICU", y_ev, b, "ev")]:
            for mn, fn in [("AUROC", roc_auc_score), ("AUPRC", average_precision_score)]:
                v = boot_metric(y, p, BOOT[key + "_cluster"][:min(B, 500)], fn); r[f"{ds} {mn}"] = fmt(fn(y, p), *ci(v))
        rows.append(r); say(nm, "|", r["temporal AUROC"], "|", r["eICU AUROC"])
    pd.DataFrame(rows).to_csv(f"{RES}/ablation_feature_sets.csv", index=False)


# ------------------------------------------------------------------ 12. alternative outcome (new vasopressor, no lactate criterion)
@section("alt_outcome")
def alt_outcome():
    import duckdb
    pm, pe, pc = f"{OUT}/sofa_hourly.parquet", f"{OUT}/sofa_hourly_eicu.parquet", f"{OUT}/cohort_final_mimic.parquet"
    if not all(os.path.exists(p) for p in (pm, pe, pc)):
        say("sofa_hourly*.parquet / cohort_final_mimic.parquet not found: alternative-outcome analysis skipped"); STATUS["alt_outcome"] = "skipped (files missing)"; return
    con = duckdb.connect()
    mm = con.execute(f"""
      SELECT c.stay_id,
             MIN(CASE WHEN s.hr > c.index_time AND s.any_pressor = 1 THEN s.hr END) AS first_after,
             MAX(CASE WHEN s.hr <= c.index_time AND s.any_pressor = 1 THEN 1 ELSE 0 END) AS pressor_before
      FROM read_parquet('{pc}') c JOIN read_parquet('{pm}') s USING (stay_id) GROUP BY c.stay_id""").df()
    ee = con.execute(f"SELECT stay_id, hr, any_pressor FROM read_parquet('{pe}')").df()
    idx_e = ev[["stay_id", "hours_icu_to_pred"]].assign(index_hr=ev.hours_icu_to_pred - 1)     # notebook: hours_icu_to_pred = index_hr + 1
    ee = ee.merge(idx_e, on="stay_id")
    after = ee[(ee.hr > ee.index_hr) & (ee.any_pressor == 1)].groupby("stay_id").hr.min().rename("first_after")
    before = ee[(ee.hr <= ee.index_hr) & (ee.any_pressor == 1)].groupby("stay_id").size().rename("n_before")
    fe = pd.DataFrame({"stay_id": ee.stay_id.unique()}).merge(after, on="stay_id", how="left").merge(before, on="stay_id", how="left")
    fe["pressor_before"] = (fe.n_before.fillna(0) > 0).astype(int)
    tr2 = tr.merge(mm, on="stay_id", how="inner"); te2 = te.merge(mm, on="stay_id", how="inner"); ev2 = ev.merge(fe, on="stay_id", how="inner")
    for d in (tr2, te2, ev2):
        d["alt"] = d.first_after.notna().astype(int)
    tr2, te2, ev2 = [d[d.pressor_before == 0].reset_index(drop=True) for d in (tr2, te2, ev2)]
    say(f"alt outcome (new vasopressor after the index hour, stays without a vasopressor at/before the index hour): train {len(tr2):,} ({tr2.alt.mean():.1%}), "
        f"test {len(te2):,} ({te2.alt.mean():.1%}), eICU {len(ev2):,} ({ev2.alt.mean():.1%})")
    cols = [c for c in FEATS if c != "on_pressor"]
    m, (a, b) = fit_predict("RF", cols, tr2, [te2, ev2], label="alt")
    rows = []
    for ds, d, p, CL in [("MIMIC-IV temporal test", te2, a, te2.subject_id.values), ("eICU-CRD external", ev2, b, ev2.hospitalid.values)]:
        y = d.alt.values
        so = d.sofa_total.fillna(0).values
        lac = d.lactate_last.fillna(med.lactate_last).values
        rows.append({"dataset": ds, "n": len(d), "events": int(y.sum()), "prevalence": y.mean(),
                     "AUROC_RF": cluster_ci(y, p, CL, roc_auc_score), "AUPRC_RF": cluster_ci(y, p, CL, average_precision_score),
                     "AUROC_SOFA_only": cluster_ci(y, so, CL, roc_auc_score), "AUROC_lactate_only": cluster_ci(y, lac, CL, roc_auc_score)})
    pd.DataFrame(rows).to_csv(f"{RES}/alternative_outcome_vasopressor_initiation.csv", index=False)
    say(pd.DataFrame(rows).to_string(index=False))


# ------------------------------------------------------------------ 13. SHAP (eICU and no-indicator model)
@section("shap")
def shap_section():
    import shap
    def run(model, cols, data, name):
        sample = data[cols].sample(min(args.shap_n, len(data)), random_state=42)
        imp, rf = model[0], model[-1]
        X = pd.DataFrame(imp.transform(sample), columns=list(imp.get_feature_names_out(cols)))
        sv = shap.TreeExplainer(rf).shap_values(X)
        sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
        plt.figure(); shap.summary_plot(sv, X, max_display=15, show=False); plt.tight_layout(); plt.savefig(f"{RES}/fig_shap_{name}.png", dpi=300); plt.close()
        pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False).head(25).to_csv(f"{RES}/shap_top25_{name}.csv")
    run(MODELS["RF"], FEATS, ev, "RF_eICU")
    cols2 = [c for c in FEATS if c not in TEMP_COLS]
    m2, _ = fit_predict("RF", cols2, tr, [te], indicator=False)
    run(m2, cols2, te, "RF_noTemp_noIndicators_MIMICtest")
    say("saved SHAP figures/tables")


# ------------------------------------------------------------------ 14. NEW: outcome composition (predictor/outcome overlap)
@section("outcome_composition")
def outcome_composition():
    rows = []
    for ds, d in [("MIMIC-IV development", tr), ("MIMIC-IV temporal test", te), ("eICU-CRD external", ev)]:
        e, n = d[d.shock == 1], d[d.shock == 0]
        lac_e, lac_n = e.lactate_last.dropna(), n.lactate_last.dropna()
        rows.append({"dataset": ds, "events": len(e), "non_events": len(n),
                     "events_with_vasopressor_running_at_index_hour_%": 100 * e.on_pressor.mean(),
                     "non_events_with_vasopressor_at_index_hour_%": 100 * n.on_pressor.mean(),
                     "events_with_last_lactate_>2_before_prediction_%": 100 * (lac_e > 2).sum() / max(len(e), 1),
                     "events_with_no_lactate_in_prior_24h_%": 100 * e.lactate_last.isna().mean(),
                     "non_events_with_last_lactate_>2_%": 100 * (lac_n > 2).sum() / max(len(n), 1),
                     "events_onset_in_first_hour_after_index_%": 100 * (e.hours_to_shock == 1).mean(),
                     "events_onset_<=3h_%": 100 * (e.hours_to_shock <= 3).mean(),
                     "events_onset_<=6h_%": 100 * (e.hours_to_shock <= 6).mean(),
                     "median_hours_to_onset": e.hours_to_shock.median(), "IQR_hours_to_onset": f"{e.hours_to_shock.quantile(.25):.0f}-{e.hours_to_shock.quantile(.75):.0f}"})
    d = pd.DataFrame(rows); d.to_csv(f"{RES}/outcome_composition.csv", index=False)
    say(d.T.to_string())


# ------------------------------------------------------------------ 15. NEW: gap sensitivity (drop events with onset <= g h after index)
@section("gap_sensitivity")
def gap_sensitivity():
    rows = []
    gaps = [0] + [int(x) for x in args.gap_hours.split(",") if x.strip()]
    for ds, d, P, CL in [("MIMIC-IV temporal test", te, PRED_TE, CL_TE), ("eICU-CRD external", ev, PRED_EV, CL_EV)]:
        for g in gaps:
            keep = ((d.shock == 0) | (d.hours_to_shock > g)).values
            yy = d.shock.values[keep].astype(int)
            if yy.sum() < 10: continue
            for k in ["RF"] + COMPARATORS:
                rows.append({"dataset": ds, "events_with_onset_<=_g_hours_after_index_removed": g, "model": k, "n": int(keep.sum()), "events": int(yy.sum()),
                             "prevalence": yy.mean(), "AUROC": cluster_ci(yy, P[k][keep], CL[keep], roc_auc_score), "AUPRC": cluster_ci(yy, P[k][keep], CL[keep], average_precision_score)})
    d = pd.DataFrame(rows); d.to_csv(f"{RES}/gap_sensitivity.csv", index=False)
    say(d[["dataset", "events_with_onset_<=_g_hours_after_index_removed", "model", "events", "AUROC", "AUPRC"]].to_string(index=False))
    say("NOTE: evaluation-only (the model is not retrained). Non-events are kept; prevalence changes, so compare AUPRC with prevalence.")


# ------------------------------------------------------------------ 16. NEW: stays without a vasopressor at the index hour
@section("pressor_excluded")
def pressor_excluded():
    rows = []
    mk = {"te": (te.on_pressor.fillna(0) == 0).values, "ev": (ev.on_pressor.fillna(0) == 0).values}
    trs = tr[tr.on_pressor.fillna(0) == 0].reset_index(drop=True)
    m, (a2, b2) = fit_predict("RF", FEATS, trs, [te[mk["te"]], ev[mk["ev"]]])
    for ds, d, P, CL, key, p_re in [("MIMIC-IV temporal test", te, PRED_TE, CL_TE, "te", a2), ("eICU-CRD external", ev, PRED_EV, CL_EV, "ev", b2)]:
        msk = mk[key]; yy = d.shock.values[msk].astype(int)
        for lab, p in [("primary RF, evaluated in stays without vasopressor at index", P["RF"][msk]), ("RF retrained without vasopressor-at-index stays", p_re)] + \
                      [(f"{k} (comparator)", P[k][msk]) for k in ["Lactate only", "SOFA only"] if k in P]:
            rows.append({"dataset": ds, "analysis": lab, "n": int(msk.sum()), "excluded_n": int((~msk).sum()), "events": int(yy.sum()), "prevalence": yy.mean(),
                         "AUROC": cluster_ci(yy, p, CL[msk], roc_auc_score), "AUPRC": cluster_ci(yy, p, CL[msk], average_precision_score)})
    d = pd.DataFrame(rows); d.to_csv(f"{RES}/pressor_excluded.csv", index=False); say(d.to_string(index=False))


# ------------------------------------------------------------------ 17. NEW: performance by prediction landmark (hours since ICU admission)
@section("landmark_strata")
def landmark_strata():
    bins = [("<=1 h", 0, 1), ("2-3 h", 2, 3), ("4-6 h", 4, 6), (">6 h", 7, 10 ** 6)]
    rows = []
    for ds, d, P, CL in [("MIMIC-IV temporal test", te, PRED_TE, CL_TE), ("eICU-CRD external", ev, PRED_EV, CL_EV)]:
        for lab, lo, hi in bins:
            msk = ((d.hours_icu_to_pred >= lo) & (d.hours_icu_to_pred <= hi)).values
            yy = d.shock.values[msk].astype(int)
            if yy.sum() < 10 or (1 - yy).sum() < 10:
                rows.append({"dataset": ds, "hours_ICU_admission_to_prediction": lab, "n": int(msk.sum()), "events": int(yy.sum()), "AUROC": "too few events"}); continue
            for k in ["RF", "Lactate only", "SOFA only"]:
                if k in P:
                    rows.append({"dataset": ds, "hours_ICU_admission_to_prediction": lab, "model": k, "n": int(msk.sum()), "events": int(yy.sum()), "prevalence": yy.mean(),
                                 "AUROC": cluster_ci(yy, P[k][msk], CL[msk], roc_auc_score), "AUPRC": cluster_ci(yy, P[k][msk], CL[msk], average_precision_score)})
    d = pd.DataFrame(rows); d.to_csv(f"{RES}/performance_by_landmark.csv", index=False); say(d.to_string(index=False))


# ------------------------------------------------------------------ 18. NEW: SOFA audit (same scoring rules, different input density)
@section("sofa_audit")
def sofa_audit():
    rows = []
    for c in SOFA_COLS:
        for ds, d in [("MIMIC-IV development", tr), ("MIMIC-IV temporal test", te), ("eICU-CRD", ev)]:
            s = d[c]
            rows.append({"variable": c, "dataset": ds, "missing_%": 100 * s.isna().mean(), "mean": s.mean(), "median": s.median(),
                         "share_nonzero_%": 100 * (s.fillna(0) > 0).mean(), **({"share_max_level(4)_%": 100 * (s == 4).mean()} if c.startswith("s_") else {})})
    a = pd.DataFrame(rows); a.to_csv(f"{RES}/sofa_component_audit.csv", index=False)
    say(a[a.variable.isin(["sofa_total", "s_resp", "s_cv", "s_cns", "s_renal", "pf_min", "gcs_min"])].round(2).to_string(index=False))
    # level distribution of s_cv among stays with a vasopressor running at the index hour
    lv = []
    for ds, d in [("MIMIC-IV (all)", mi), ("eICU-CRD", ev)]:
        sub = d[d.on_pressor.fillna(0) == 1]
        lv.append({"dataset": ds, "stays_with_vasopressor_at_index": len(sub), **{f"s_cv=={i}_%": 100 * (sub.s_cv == i).mean() for i in range(5)}})
    pd.DataFrame(lv).to_csv(f"{RES}/sofa_cv_levels_on_pressor.csv", index=False); say(pd.DataFrame(lv).round(1).to_string(index=False))
    # imputed-default dose share (eICU notebook: dose NULL when the unit is ml/hr -> default 0.05 mcg/kg/min)
    try:
        import duckdb
        con = duckdb.connect(); dl = []
        for ds, f in [("MIMIC-IV", "sofa_hourly.parquet"), ("eICU-CRD", "sofa_hourly_eicu.parquet")]:
            if os.path.exists(f"{OUT}/{f}"):
                r = con.execute(f"SELECT COUNT(*) n, AVG((ne_epi=0.05)::INT) share_default FROM read_parquet('{OUT}/{f}') WHERE any_pressor=1 AND ne_epi>0").df().iloc[0]
                dl.append({"dataset": ds, "pressor_hours_with_ne_epi>0": int(r.n), "share_with_24h_max_dose_exactly_0.05_(default_or_true)": float(r.share_default)})
        if dl:
            pd.DataFrame(dl).to_csv(f"{RES}/sofa_default_dose_share.csv", index=False); say(pd.DataFrame(dl).to_string(index=False))
    except Exception as e:
        say(f"default-dose audit skipped: {e}")


# ------------------------------------------------------------------ run
for fn in (discrimination, delong, calibration, decision_curve, operating_points, horizon, hospital_heterogeneity,
           covariate_shift, table1, subgroups, ablations, alt_outcome, outcome_composition, gap_sensitivity,
           pressor_excluded, landmark_strata, sofa_audit, shap_section):
    fn()

bad = {k: v for k, v in STATUS.items() if v not in ("ok", "skipped")}
say("\n=== STATUS ===")
for k, v in STATUS.items():
    say(f"  {k:28s} {v}")
say("\nSECTIONS THAT NEED ATTENTION: " + (", ".join(bad) if bad else "none"))
say("Done. Results in", RES)
json.dump(STATUS, open(os.path.join(RES, "results_status.json"), "w"), indent=2)
LOG.close()
