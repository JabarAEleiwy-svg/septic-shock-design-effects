#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sequence_analysis.py
====================
Controlled experiment of ladder_analysis.py repeated with a SEQUENCE model (GRU) that reads the hourly time course of the
24 h before the prediction time, instead of the summary features (mean / min / max / last) used by the tree models.

Why: the manuscript shows that the apparent AUROC depends on where the predictor window ends (design B vs C) and on the inputs
that overlap with the outcome definition (P = vasopressor-derived inputs, L = lactate).  Whether the same holds for a model that
sees the whole time course is untested; this script tests it.  Nothing here is tuned on the temporal test set or on eICU-CRD.

Same as ladder_analysis.py (so that the numbers are comparable stay by stay):
  * cohort, outcome (shock with onset in (index, index + H], H in --horizons), temporal split (MIMIC-IV development 2008-2016,
    temporal test 2017-2022, eICU-CRD external without retraining), analysis set (ICU stay lasts until index + H + 1 h);
  * designs: B (predictors end 1 h after the index), C (predictors end at index + H + 1 h), and the input ablations
    (suffix P = without vasopressor-derived inputs on_pressor / s_cv / sofa_total; L = without lactate; PL = without both);
  * paired cluster bootstrap (patients in MIMIC-IV, hospitals in eICU-CRD), same stays and same resamples in both designs.

Sequence representation (per stay): 24 hourly bins ending at the prediction time, bin 23 = [pred - 1 h, pred).  Channels:
7 vital signs (mean of the bin), 12 laboratory variables (last value in the bin), 10 SOFA columns taken at hour pred-24 ... pred-1
(the hourly SOFA table is itself a rolling 24-h worst value).  Per channel the model receives: value (winsorised at the 0.5/99.5
percentiles of the development set, standardised, last observation carried forward inside the window, 0 if never observed),
an observed-in-this-bin mask, and the time since the last observation (in units of 24 h; 1.0 = not observed in the window).  Static
inputs: age, sex, hours from ICU admission to the prediction time.  Only data recorded BEFORE the prediction time are read
(tests/test_sequence.py checks this by perturbing everything at or after that time).

Model (fixed, not tuned): GRU(64) -> concat(static) -> Linear(64)-ReLU-Dropout(0.2)-Linear(1); AdamW lr 1e-3, weight decay 1e-4,
batch 256, unweighted binary cross-entropy (probabilities are not class-weighted), early stopping on the AUROC of a validation
set made of 10 % of the development PATIENTS (patience --patience, max --epochs); the prediction is the mean of --seeds
independently initialised networks.  Hyper-parameters are the same in every design.

Self-check (k = 1): the last-bin laboratory values and SOFA columns of the sequence tensor must reproduce `*_last` and the SOFA
columns of features_mimic / features_eicu (the table used by the primary analysis); the run stops otherwise.

Inputs (in --out):  features_mimic, features_eicu, cohort_final_mimic, cohort_eicu, chart_sub, lab_sub, sofa_hourly,
                    eicu_vitalp_sub, eicu_vitala_sub, eicu_lab_sub, sofa_hourly_eicu   (all .parquet, as for ladder_analysis.py)
Outputs (in --out/sequence): seq_{mimic,eicu}_k{K}.npz (PATIENT-LEVEL: keep private), sequence_results.csv,
                    sequence_paired_differences.csv, sequence_design_ns.csv, sequence_training.csv, sequence_log.txt
"""
import argparse, os, sys, warnings, platform, time
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

T = 24                                                   # hourly bins
VARS = ["hr", "rr", "spo2", "sbp", "dbp", "map", "temp"]
M_LAB = {"creat": 50912, "bili": 50885, "plt": 51265, "lactate": 50813, "wbc": 51301, "hgb": 51222,
         "k": 50971, "na": 50983, "bicarb": 50882, "ph": 50820, "pao2": 50821, "base_excess": 50802}
E_LAB = {"creat": "creatinine", "bili": "total bilirubin", "plt": "platelets x 1000", "lactate": "lactate", "wbc": "wbc x 1000",
         "hgb": "hgb", "k": "potassium", "na": "sodium", "bicarb": "bicarbonate", "ph": "ph", "pao2": "pao2", "base_excess": "base excess"}
SOFA_COLS = ["sofa_total", "s_resp", "s_coag", "s_liver", "s_cv", "s_cns", "s_renal", "gcs_min", "pf_min", "on_pressor"]
SOFA_SEL = ("s.sofa AS sofa_total, s.s_resp, s.s_coag, s.s_liver, s.s_cv, s.s_cns, s.s_renal, s.gcs_min, s.pf_min, "
            "s.any_pressor AS on_pressor")
CHANNELS = VARS + list(M_LAB) + SOFA_COLS
CIDX = {c: i for i, c in enumerate(CHANNELS)}
PRESS = ["on_pressor", "s_cv", "sofa_total"]             # vasopressor-derived inputs (as in ladder_analysis.py)
LAC = ["lactate"]
LIM = {"hr": (0, 300), "rr": (0, 100), "spo2": (0, 100), "sbp": (0, 350), "dbp": (0, 250), "map": (0, 300), "temp": (25, 45)}
TRAIN_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
MIMIC_ITEMS = [220045, 220210, 220277, 220179, 220050, 220180, 220051, 220181, 220052, 225312, 223762, 223761]
STAT_M = ["stay_id", "subject_id", "anchor_year_group", "shock", "hours_to_shock", "age", "male"]
STAT_E = ["stay_id", "hospitalid", "shock", "hours_to_shock", "age", "male"]


# ------------------------------------------------------------------ data cleaning (same limits as ladder_analysis.clean)
def clean_tensor(X):
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    for v, (lo, hi) in LIM.items():
        a = X[:, :, CIDX[v]]
        a[(a < lo) | (a > hi)] = np.nan
    a = X[:, :, CIDX["pf_min"]]
    a[a > 800] = np.nan
    return X


def newcon():
    import duckdb
    con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=2"); return con


def _fill(X, ids, df, cols, jcol="j"):
    """df: stay_id, j (0..T-1), one column per channel in `cols`; writes into X[row, j, channel]"""
    if len(df) == 0: return
    rows = pd.Index(ids).get_indexer(df["stay_id"].to_numpy())
    j = df[jcol].to_numpy().astype(int)
    ok = (rows >= 0) & (j >= 0) & (j < T)
    assert ok.all(), "bin index outside 0..23 or unknown stay: the time arithmetic is wrong"
    for c in cols:
        v = df[c].to_numpy(dtype="float64")
        X[rows, j, CIDX[c]] = v


# ------------------------------------------------------------------ builders (SQL logic follows ladder_analysis.build_*)
def build_mimic_seq(out, k, static):
    """static: DataFrame with stay_id.  Returns ids (sorted), X [N, 24, C] (NaN = not observed), hours_icu_to_pred."""
    static = static[["stay_id"]].drop_duplicates().sort_values("stay_id").reset_index(drop=True)
    ids = static.stay_id.to_numpy()
    con = newcon(); R = lambda f: f"read_parquet('{out}/{f}.parquet')"
    con.register("st", static)
    con.execute(f"""CREATE TABLE cf AS
        SELECT c.stay_id, c.subject_id, c.intime, c.index_time + INTERVAL '{k} hours' AS pred_time
        FROM {R('cohort_final_mimic')} c JOIN st USING (stay_id)""")
    X = np.full((len(ids), T, len(CHANNELS)), np.nan, dtype=np.float64)
    # vital signs: mean per hourly bin; bin 23 = [pred-1h, pred)  (b = ceil(delta/3600) - 1, delta = pred - charttime in (0, 24h])
    vdf = con.execute(f"""
        SELECT stay_id, 23 - b AS j, var, AVG(v) AS v FROM (
          SELECT cf.stay_id, CAST(ceil(date_diff('second', ch.charttime, cf.pred_time) / 3600.0) AS INTEGER) - 1 AS b,
            CASE WHEN ch.itemid=220045 THEN 'hr' WHEN ch.itemid=220210 THEN 'rr' WHEN ch.itemid=220277 THEN 'spo2'
                 WHEN ch.itemid IN (220179,220050) THEN 'sbp' WHEN ch.itemid IN (220180,220051) THEN 'dbp'
                 WHEN ch.itemid IN (220181,220052,225312) THEN 'map' WHEN ch.itemid IN (223762,223761) THEN 'temp' END AS var,
            CASE WHEN ch.itemid=223761 THEN (ch.valuenum-32)/1.8 ELSE ch.valuenum END AS v
          FROM {R('chart_sub')} ch JOIN cf USING (stay_id)
          WHERE ch.itemid IN ({','.join(map(str, MIMIC_ITEMS))})
            AND ch.charttime < cf.pred_time AND ch.charttime >= cf.pred_time - INTERVAL '24 hours')
        GROUP BY stay_id, j, var""").df()
    for var, g in vdf.groupby("var"):
        g = g.rename(columns={"v": var}); _fill(X, ids, g, [var])
    # laboratory: last value in the bin (labs are joined by subject, as in the primary analysis)
    case = "CASE " + " ".join(f"WHEN lb.itemid={i} THEN '{n}'" for n, i in M_LAB.items()) + " END"
    ldf = con.execute(f"""
        SELECT stay_id, 23 - b AS j, var, arg_max(v, charttime) AS v FROM (
          SELECT cf.stay_id, lb.charttime, CAST(ceil(date_diff('second', lb.charttime, cf.pred_time) / 3600.0) AS INTEGER) - 1 AS b,
                 {case} AS var, lb.valuenum AS v
          FROM {R('lab_sub')} lb JOIN cf ON cf.subject_id = lb.subject_id
          WHERE lb.itemid IN ({','.join(map(str, M_LAB.values()))})
            AND lb.charttime < cf.pred_time AND lb.charttime >= cf.pred_time - INTERVAL '24 hours')
        GROUP BY stay_id, j, var""").df()
    for var, g in ldf.groupby("var"):
        g = g.rename(columns={"v": var}); _fill(X, ids, g, [var])
    # hourly SOFA rows at hr = pred-24h ... pred-1h  ->  bin j = 24 - (pred - hr in hours)
    sdf = con.execute(f"""
        SELECT cf.stay_id, 24 - date_diff('hour', s.hr, cf.pred_time) AS j, {SOFA_SEL}
        FROM cf JOIN {R('sofa_hourly')} s ON s.stay_id = cf.stay_id
        WHERE s.hr >= cf.pred_time - INTERVAL '24 hours' AND s.hr <= cf.pred_time - INTERVAL '1 hour'""").df()
    _fill(X, ids, sdf, SOFA_COLS)
    hrs = con.execute("SELECT stay_id, date_diff('hour', intime, pred_time) AS h FROM cf").df().set_index("stay_id").h
    return ids, clean_tensor(X), hrs.reindex(ids).to_numpy(dtype="float64")


def build_eicu_seq(out, k, static):
    """static: DataFrame with stay_id, hours_icu_to_pred (primary design: index_hr + 1)."""
    static = static[["stay_id", "hours_icu_to_pred"]].drop_duplicates("stay_id").sort_values("stay_id").reset_index(drop=True)
    ids = static.stay_id.to_numpy()
    con = newcon(); R = lambda f: f"read_parquet('{out}/{f}.parquet')"
    cf = static.assign(index_hr=lambda d: d.hours_icu_to_pred - 1)
    cf["pred_hr"] = cf.index_hr + k; cf["pred_min"] = cf.pred_hr * 60
    con.register("cf", cf[["stay_id", "index_hr", "pred_hr", "pred_min"]])
    X = np.full((len(ids), T, len(CHANNELS)), np.nan, dtype=np.float64)
    con.execute(f"""CREATE TABLE vit AS
        SELECT stay_id, t, heartrate AS hrt, respiration AS rrt, sao2 AS spo2t, temperature AS tempt,
               systemicsystolic AS sbpt, systemicdiastolic AS dbpt, systemicmean AS mapt FROM {R('eicu_vitalp_sub')}
        UNION ALL
        SELECT stay_id, t, NULL, NULL, NULL, NULL, noninvasivesystolic, noninvasivediastolic, noninvasivemean FROM {R('eicu_vitala_sub')}""")
    vmap = {"hr": "hrt", "rr": "rrt", "spo2": "spo2t", "sbp": "sbpt", "dbp": "dbpt", "map": "mapt", "temp": "tempt"}
    aggs = ", ".join(f"AVG({c}) AS {x}" for x, c in vmap.items())
    vdf = con.execute(f"""
        SELECT cf.stay_id, 23 - (CAST(ceil((cf.pred_min - vit.t) / 60.0) AS INTEGER) - 1) AS j, {aggs}
        FROM vit JOIN cf ON cf.stay_id = vit.stay_id AND vit.t < cf.pred_min AND vit.t >= cf.pred_min - 1440
        GROUP BY cf.stay_id, j""").df()
    _fill(X, ids, vdf, VARS)
    laggs = ", ".join(f"arg_max(lab.v, lab.t) FILTER (WHERE lab.labname='{nm}') AS {x}" for x, nm in E_LAB.items())
    ldf = con.execute(f"""
        SELECT cf.stay_id, 23 - (CAST(ceil((cf.pred_min - lab.t) / 60.0) AS INTEGER) - 1) AS j, {laggs}
        FROM {R('eicu_lab_sub')} lab JOIN cf ON cf.stay_id = lab.stay_id AND lab.t < cf.pred_min AND lab.t >= cf.pred_min - 1440
        GROUP BY cf.stay_id, j""").df()
    _fill(X, ids, ldf, list(M_LAB))
    sdf = con.execute(f"""
        SELECT cf.stay_id, 24 - (cf.pred_hr - s.hr) AS j, {SOFA_SEL}
        FROM cf JOIN {R('sofa_hourly_eicu')} s ON s.stay_id = cf.stay_id
        WHERE s.hr >= cf.pred_hr - 24 AND s.hr <= cf.pred_hr - 1""").df()
    _fill(X, ids, sdf, SOFA_COLS)
    return ids, clean_tensor(X), cf.pred_hr.to_numpy(dtype="float64")


def selfcheck(tag, ids, X, feat, say):
    """k = 1 tensor must reproduce the lab `*_last` and SOFA columns of the table used by the primary analysis"""
    cols = [f"{n}_last" for n in M_LAB] + SOFA_COLS
    cols = [c for c in cols if c in feat.columns]
    f = feat.set_index("stay_id").reindex(ids)
    f = f[cols].astype("float64")
    # same cleaning as ladder_analysis.clean (pf_min)
    if "pf_min" in f: f.loc[f["pf_min"] > 800, "pf_min"] = np.nan
    bad = []
    for c in cols:
        if c.endswith("_last"):
            ch = c[:-5]
            seq = X[:, :, CIDX[ch]]
            has = ~np.isnan(seq)
            last = np.where(has.any(1), seq[np.arange(len(seq)), T - 1 - np.argmax(has[:, ::-1], axis=1)], np.nan)   # last observed in window
            tol = 0.02
        else:
            last = X[:, -1, CIDX[c]]; tol = 5e-4
        a, b = f[c].to_numpy(), last
        eq = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-6 + 1e-6 * np.abs(np.nan_to_num(a)))
        frac = 1 - eq.mean()
        if frac > tol: bad.append((c, round(float(frac), 4)))
    say(f"[{tag}] self-check on {len(ids):,} stays, {len(cols)} columns")
    if bad:
        raise RuntimeError(f"[{tag}] the sequence tensor does NOT reproduce the primary feature table (column, mismatch fraction): {bad}. "
                           "Do not trust any sequence result until this is explained.")
    say(f"[{tag}] self-check PASSED")


# ------------------------------------------------------------------ inputs for the network
def fit_stats(Xtr, keep):
    """winsorisation limits and standardisation from the DEVELOPMENT set only, per channel"""
    n = len(keep); lo = np.zeros(n); hi = np.zeros(n); mu = np.zeros(n); sd = np.ones(n)
    for i, c in enumerate(keep):
        v = Xtr[:, :, c].ravel(); v = v[~np.isnan(v)]
        if len(v) < 2: continue
        lo[i], hi[i] = np.percentile(v, [0.5, 99.5]); v = np.clip(v, lo[i], hi[i]); mu[i], sd[i] = v.mean(), max(v.std(), 1e-6)
    return lo, hi, mu, sd


def make_inputs(X, keep, stats):
    lo, hi, mu, sd = stats
    Xk = X[:, :, keep]; obs = ~np.isnan(Xk)
    Z = (np.clip(Xk, lo, hi) - mu) / sd
    N, _, C = Z.shape
    filled = np.zeros((N, T, C), np.float32); delta = np.zeros((N, T, C), np.float32)
    last_v = np.zeros((N, C)); last_t = np.full((N, C), -1.0)
    for t in range(T):
        o = obs[:, t, :]
        last_v = np.where(o, np.nan_to_num(Z[:, t, :]), last_v); last_t = np.where(o, t, last_t)
        filled[:, t, :] = last_v
        delta[:, t, :] = np.where(o, 0.0, np.where(last_t < 0, T, t - last_t))
    return np.concatenate([filled, obs.astype(np.float32), delta / T], axis=2).astype(np.float32)


def static_raw(age, male, hrs):
    return np.stack([np.asarray(age, dtype=float), np.asarray(male, dtype=float), np.asarray(hrs, dtype=float)], 1)


# ------------------------------------------------------------------ model
def get_torch():
    import torch, torch.nn as nn
    return torch, nn


def fit_predict_gru(Xtr, Str, ytr, Xva, Sva, yva, tests, args, seed, device):
    torch, nn = get_torch()
    from sklearn.metrics import roc_auc_score

    class Net(nn.Module):
        def __init__(s, d_in, d_st, hidden, drop):
            super().__init__()
            s.gru = nn.GRU(d_in, hidden, batch_first=True)
            s.head = nn.Sequential(nn.Linear(hidden + d_st, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 1))
        def forward(s, x, st):
            _, h = s.gru(x)
            return s.head(torch.cat([h[-1], st], 1)).squeeze(1)

    torch.manual_seed(seed); np.random.seed(seed)
    model = Net(Xtr.shape[2], Str.shape[1], args.hidden, 0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    tt = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    xtr, st_tr, yt = tt(Xtr), tt(Str), tt(ytr)
    gen = torch.Generator().manual_seed(seed)

    def predict(X, S):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                out.append(torch.sigmoid(model(tt(X[i:i + 2048]), tt(S[i:i + 2048]))).cpu().numpy())
        return np.concatenate(out)

    best, best_state, bad, best_ep = -1.0, None, 0, 0
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(xtr), generator=gen)
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch].to(device)
            opt.zero_grad(); loss = lossf(model(xtr[idx], st_tr[idx]), yt[idx]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        auc = roc_auc_score(yva, predict(Xva, Sva))
        if auc > best + 1e-4: best, bad, best_ep = auc, 0, ep + 1; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience: break
    model.load_state_dict(best_state)
    return [predict(X, S) for X, S in tests], best, best_ep


# ------------------------------------------------------------------ statistics (same as ladder_analysis.py)
def make_boot(clusters, nb, seed):
    rng = np.random.default_rng(seed); codes = pd.factorize(pd.Series(clusters))[0]
    groups = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    return [np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))]) for _ in range(nb)]


def boot_vals(y, p, idxs, fn):
    out = []
    for i in idxs:
        yy = y[i]; out.append(fn(yy, p[i]) if 0 < yy.sum() < len(yy) else np.nan)
    return np.array(out)


def pct(v): v = v[~np.isnan(v)]; return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def fmt(pt, lo, hi): return f"{pt:.3f} ({lo:.3f}-{hi:.3f})"


def design_channels(code):
    """channel indices kept in a design; code = B / C / B-P / B-L / B-PL / C-P / C-L / C-PL"""
    drop = set()
    suffix = code.split("-")[1] if "-" in code else ""
    if "P" in suffix: drop |= {CIDX[c] for c in PRESS}
    if "L" in suffix: drop |= {CIDX[c] for c in LAC}
    return [i for i in range(len(CHANNELS)) if i not in drop]


# ------------------------------------------------------------------ main
def main(argv=None):
    from sklearn.metrics import roc_auc_score, average_precision_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/drive/MyDrive/sepsis_project")
    ap.add_argument("--horizons", default="6,24")
    ap.add_argument("--designs", default="B,C,B-PL,C-PL")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=5, help="networks averaged per design")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--only_build", action="store_true", help="build the tensors + self-check, then stop")
    args = ap.parse_args(argv)
    OUT = args.out; SD = os.path.join(OUT, "sequence"); os.makedirs(SD, exist_ok=True)
    HORIZONS = [int(x) for x in args.horizons.split(",") if x.strip()]; DESIGNS = [d.strip() for d in args.designs.split(",") if d.strip()]
    LOG = open(os.path.join(SD, "sequence_log.txt"), "w")

    def say(*a):
        s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()
    device = args.device
    if device == "auto":
        try:
            import torch; device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception: device = "cpu"
    import duckdb
    say("python", platform.python_version(), "| duckdb", duckdb.__version__, "| pandas", pd.__version__, "| numpy", np.__version__, "| device", device)
    say(f"horizons={HORIZONS} designs={DESIGNS} nboot={args.nboot} seeds={args.seeds} epochs<={args.epochs} patience={args.patience} hidden={args.hidden}")

    fm_raw = pd.read_parquet(f"{OUT}/features_mimic.parquet"); fe_raw = pd.read_parquet(f"{OUT}/features_eicu.parquet")
    cm = pd.read_parquet(f"{OUT}/cohort_final_mimic.parquet")[["stay_id", "index_time", "outtime"]].set_index("stay_id")
    ce = pd.read_parquet(f"{OUT}/cohort_eicu.parquet")[["stay_id", "outoffset"]].set_index("stay_id")
    fm_raw = fm_raw.sort_values("stay_id").reset_index(drop=True); fe_raw = fe_raw.sort_values("stay_id").reset_index(drop=True)
    fe_idx = fe_raw.set_index("stay_id").hours_icu_to_pred - 1
    say(f"MIMIC stays {len(fm_raw):,} | eICU stays {len(fe_raw):,}")

    KS = sorted({1} | {H + 1 for H in HORIZONS})
    TM, TE = {}, {}
    for k in KS:
        for tag, builder, raw, store in (("mimic", build_mimic_seq, fm_raw, TM), ("eicu", build_eicu_seq, fe_raw, TE)):
            path = f"{SD}/seq_{tag}_k{k}.npz"
            if args.rebuild or not os.path.exists(path):
                say(f"[{tag} k={k}] building ..."); t0 = time.time()
                ids, X, hrs = builder(OUT, k, raw[["stay_id"] + (["hours_icu_to_pred"] if tag == "eicu" else [])])
                if k == 1: selfcheck(f"{tag} k=1", ids, X, raw, say)
                np.savez_compressed(path, ids=ids, X=X, hrs=hrs); say(f"[{tag} k={k}] built in {time.time() - t0:.0f}s")
            z = np.load(path)
            assert (z["ids"] == raw.stay_id.to_numpy()).all(), "stay order differs from the primary feature table"
            store[k] = (z["X"], z["hrs"])
            say(f"[{tag} k={k}] {len(z['ids']):,} stays; median hours_icu_to_pred {np.median(z['hrs']):.0f}; observed fraction {np.mean(~np.isnan(z['X'])):.3f}")
    if args.only_build:
        say("--only_build: tensors built and self-check passed."); return

    rows, diffs, ns, trainlog = [], [], [], []
    for H in HORIZONS:
        kC = H + 1
        okm = cm.loc[fm_raw.stay_id].pipe(lambda d: d.outtime.values >= (d.index_time + pd.Timedelta(hours=kC)).values)
        oke = ce.loc[fe_raw.stay_id].outoffset.values >= (fe_idx.loc[fe_raw.stay_id].values + kC) * 60
        mm, ee = np.flatnonzero(okm), np.flatnonzero(oke)                       # rows kept (identical for every design)
        sm, se = fm_raw.iloc[mm].reset_index(drop=True), fe_raw.iloc[ee].reset_index(drop=True)
        is_tr = sm.anchor_year_group.isin(TRAIN_GROUPS).to_numpy()
        tr_i, te_i = np.flatnonzero(is_tr), np.flatnonzero(~is_tr)
        Y = {"tr": ((sm.shock == 1) & (sm.hours_to_shock <= H)).astype(int).to_numpy()[tr_i],
             "te": ((sm.shock == 1) & (sm.hours_to_shock <= H)).astype(int).to_numpy()[te_i],
             "ev": ((se.shock == 1) & (se.hours_to_shock <= H)).astype(int).to_numpy()}
        CL = {"te": sm.subject_id.to_numpy()[te_i], "ev": se.hospitalid.to_numpy()}
        for part, nm in (("tr", "MIMIC development"), ("te", "MIMIC temporal test"), ("ev", "eICU external")):
            ns.append({"H": H, "dataset": nm, "n": len(Y[part]), "events": int(Y[part].sum()), "prevalence": Y[part].mean()})
            say(f"H={H}h {nm:22s} n={len(Y[part]):6,} events={int(Y[part].sum()):5,} ({Y[part].mean():.1%})")
        # patient-grouped validation split of the development set (identical for every design)
        subj = sm.subject_id.to_numpy()[tr_i]; us = np.unique(subj)
        rng = np.random.default_rng(args.seed); val_s = set(rng.choice(us, size=max(1, int(round(0.1 * len(us)))), replace=False))
        isva = np.array([s in val_s for s in subj]); fit_i, va_i = np.flatnonzero(~isva), np.flatnonzero(isva)
        assert set(subj[fit_i]).isdisjoint(subj[va_i])
        if Y["tr"][va_i].sum() == 0 or Y["tr"][fit_i].sum() == 0: raise RuntimeError("no events in the fitting or validation part of the development set")
        P = {"te": {}, "ev": {}}
        for code in DESIGNS:
            k = 1 if code.startswith("B") else kC; keep = design_channels(code)
            Xm, hm = TM[k]; Xe, he = TE[k]
            Xm_s, hm_s = Xm[mm], hm[mm]; Xe_s, he_s = Xe[ee], he[ee]
            stats = fit_stats(Xm_s[tr_i][fit_i], keep)                            # development (fitting part) only
            I_tr = make_inputs(Xm_s[tr_i], keep, stats); I_te = make_inputs(Xm_s[te_i], keep, stats); I_ev = make_inputs(Xe_s, keep, stats)
            raw_fit = static_raw(sm.age.to_numpy()[tr_i][fit_i], sm.male.to_numpy()[tr_i][fit_i], hm_s[tr_i][fit_i])
            med = np.nanmedian(raw_fit, axis=0); ref = np.where(np.isnan(raw_fit), med, raw_fit)     # median imputation, development set only
            smu, ssd = ref.mean(0), np.maximum(ref.std(0), 1e-6)
            def stat(age, male, hrs):
                S = static_raw(age, male, hrs); return ((np.where(np.isnan(S), med, S) - smu) / ssd).astype(np.float32)
            S_tr = stat(sm.age.to_numpy()[tr_i], sm.male.to_numpy()[tr_i], hm_s[tr_i]); S_te = stat(sm.age.to_numpy()[te_i], sm.male.to_numpy()[te_i], hm_s[te_i])
            S_ev = stat(se.age.to_numpy(), se.male.to_numpy(), he_s)
            ps_te, ps_ev = [], []
            for s in range(args.seeds):
                (p_te, p_ev), vauc, bep = fit_predict_gru(I_tr[fit_i], S_tr[fit_i], Y["tr"][fit_i], I_tr[va_i], S_tr[va_i], Y["tr"][va_i],
                                                          [(I_te, S_te), (I_ev, S_ev)], args, args.seed + 1000 * s, device)
                ps_te.append(p_te); ps_ev.append(p_ev)
                trainlog.append({"H": H, "design": code, "seed": args.seed + 1000 * s, "best_epoch": bep, "val_auroc": vauc})
                say(f"  H={H} {code:5s} net {s + 1}/{args.seeds}: best epoch {bep}, validation AUROC {vauc:.3f}")
            P["te"][code], P["ev"][code] = np.mean(ps_te, 0), np.mean(ps_ev, 0)
            say(f"  H={H} design {code:5s} GRU AUROC temporal {roc_auc_score(Y['te'], P['te'][code]):.3f} | eICU {roc_auc_score(Y['ev'], P['ev'][code]):.3f}")
        for part, nm in (("te", "MIMIC temporal test"), ("ev", "eICU external")):
            y = Y[part]; idxs = make_boot(CL[part], args.nboot, args.seed + H * 7 + (0 if part == "te" else 1)); met = {}
            for code in DESIGNS:
                met[code] = {}
                for mn, fn in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
                    v = boot_vals(y, P[part][code], idxs, fn); met[code][mn] = v; pt = fn(y, P[part][code]); lo, hi = pct(v)
                    rows.append({"model": "GRU", "H": H, "dataset": nm, "code": code, "metric": mn, "value": pt, "lo": lo, "hi": hi,
                                 "formatted": fmt(pt, lo, hi), "n": len(y), "events": int(y.sum()), "prevalence": y.mean()})
            pairs = [(a, b) for a, b in (("C", "B"), ("C-PL", "B-PL"), ("B-PL", "B"), ("C-PL", "B"), ("C-P", "B-P"), ("C-L", "B-L"),
                                         ("B-P", "B"), ("B-L", "B"), ("C-P", "B"), ("C-L", "B")) if a in DESIGNS and b in DESIGNS]
            for a, b in pairs:
                for mn, fn in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
                    d = met[a][mn] - met[b][mn]; pt = fn(y, P[part][a]) - fn(y, P[part][b]); lo, hi = pct(d)
                    diffs.append({"model": "GRU", "H": H, "dataset": nm, "comparison": f"{a} minus {b}", "metric": mn, "difference": pt, "lo": lo, "hi": hi, "formatted": fmt(pt, lo, hi)})
    res = pd.DataFrame(rows); res.to_csv(f"{SD}/sequence_results.csv", index=False)
    pd.DataFrame(diffs).to_csv(f"{SD}/sequence_paired_differences.csv", index=False)
    pd.DataFrame(ns).to_csv(f"{SD}/sequence_design_ns.csv", index=False); pd.DataFrame(trainlog).to_csv(f"{SD}/sequence_training.csv", index=False)
    say("\n=== GRU: AUROC by design ==="); say(res[res.metric == "AUROC"][["H", "dataset", "code", "formatted", "events", "prevalence"]].to_string(index=False))
    say("\n=== GRU: paired differences (cluster bootstrap) ==="); say(pd.DataFrame(diffs)[["H", "dataset", "comparison", "metric", "formatted"]].to_string(index=False))
    say(f"\nDone. Results in {SD}"); LOG.close()


if __name__ == "__main__":
    main()
