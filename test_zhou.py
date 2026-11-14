"""Tests of scripts/zhou_design_analysis.py on synthetic tables (no patient data).
Run:  python tests/test_zhou.py        (or: pytest tests/test_zhou.py)
  1. builders vs oracle  - the cohort (first stay, age 18-89, ICU stay 24 h-30 d), shock onset and EVERY feature column built with duckdb equal an
                           independent pandas implementation (MIMIC-IV and eICU-CRD, "agg" and "first" columns, SOFA read at hour 23)
  2. self-checks         - S1/S2/S3 pass on tables that follow the primary pipeline, and the run STOPS when a primary table is corrupted
  3. no future leakage   - overwriting every record at or after t0 + 24 h leaves the features unchanged; overwriting records inside the window
                           changes them (so the test can fail)
  4. pipeline + direction - the whole script runs; with the built-in structure, counting early events (E1) gives a higher AUROC than the
                           late-event prediction (E2), and removing the vasopressor/lactate inputs lowers E1
"""
import os, sys, shutil, tempfile, importlib.util
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_zhou as syn
spec = importlib.util.spec_from_file_location("zhou", os.path.join(HERE, "..", "scripts", "zhou_design_analysis.py"))
zhou = importlib.util.module_from_spec(spec); spec.loader.exec_module(zhou)


def frames_equal(a, b, cols, what, tol_cols=()):
    m = a[["stay_id"] + cols].merge(b[["stay_id"] + cols], on="stay_id", suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b), f"{what}: row sets differ ({len(a)}, {len(b)}, {len(m)})"
    bad = []
    for c in cols:
        x, y = m[c + "_a"].astype(float), m[c + "_b"].astype(float)
        eq = (x.isna() & y.isna()) | ((x - y).abs() <= 1e-6 + 1e-6 * x.abs())
        frac = 1 - eq.mean()
        if frac > (0.03 if (c.endswith("_last") or c.endswith("_first")) else 0): bad.append((c, round(float(frac), 4)))
    assert not bad, f"{what}: columns differ from the oracle: {bad[:10]}"


def test_builders(tmp, T):
    zm, flow_m, shock_m = zhou.build_mimic_z(tmp)
    # --- expected cohort, from the generator's own tables
    cm, s3 = T["cohort_mimic"], T["s3"]
    first = cm.sort_values(["subject_id", "intime", "stay_id"]).groupby("subject_id").head(1).stay_id
    exp = s3[s3.stay_id.isin(first) & s3.age.between(18, 89)]
    mins = (exp.outtime - exp.intime) / pd.Timedelta(minutes=1)
    exp = exp[(mins >= 1440) & (mins <= 30 * 1440)]
    assert set(zm.stay_id) == set(exp.stay_id), (len(zm), len(exp))
    assert flow_m["ICU stay 24 h to 30 d (analysis set)"] == len(exp)
    for name, sub in (("age", s3[~s3.age.between(18, 89)]), ("not first", s3[~s3.stay_id.isin(first)])):
        assert len(sub) > 0, f"the generator has no stays to test the {name} rule"
        assert not set(sub.stay_id) & set(zm.stay_id), f"{name} rule not applied"
    # --- shock onset vs oracle (hours from the hour of admission)
    t0 = exp.set_index("stay_id").intime.dt.floor("h")
    sh = T["shock_m"].reindex(zm.stay_id)
    exp_h = ((sh.values - t0.reindex(zm.stay_id).values) / np.timedelta64(1, "h"))
    assert np.array_equal(np.isnan(exp_h), zm.shock_h.isna().values) and np.allclose(exp_h[~np.isnan(exp_h)], zm.shock_h.dropna().values)
    s = zm.shock_h
    assert (s <= 23).sum() > 10 and (s >= 24).sum() > 10 and (s > 168).sum() > 0, "generator lacks early / late / very late events"
    # --- features vs independent pandas
    zz = exp.set_index("stay_id").loc[zm.stay_id].reset_index()[["stay_id", "subject_id", "intime"]]
    orc = syn.oracle_features_mimic(zz, T["chart"], T["lab"], T["sofa"])
    cols = zhou.AGG_COLS + [c for c in zhou.FIRST_COLS if c.endswith("_first")]
    frames_equal(zm.sort_values("stay_id").reset_index(drop=True), orc.sort_values("stay_id").reset_index(drop=True), cols, "MIMIC features")
    # explicit window edges (stay 1: records at t0-1s, t0, pred-1s, pred): first = value at t0, last = value at pred-1s
    r = zm[zm.stay_id == 1].iloc[0]
    assert r.hr_first == 112.0 and r.hr_last == 113.0 and r.lactate_first == 122.0 and r.lactate_last == 123.0, (r.hr_first, r.hr_last, r.lactate_first, r.lactate_last)
    print("PASS builders vs oracle (MIMIC-IV): cohort rules, shock onset, all feature columns, window edges")
    ze, flow_e, s3e, shock_e = zhou.build_eicu_z(tmp)
    ce = T["cohort_e"]; ok = ce[ce.age_num.between(18, 89) & (ce.outoffset >= 1440) & (ce.outoffset <= 30 * 1440)]
    ok = ok[ok.stay_id.isin(s3e.stay_id)]
    assert set(ze.stay_id) == set(ok.stay_id), (len(ze), len(ok))
    assert len(set(T["c3"].stay_id) - set(s3e.stay_id)) == 0 and len(s3e) > len(T["c3"]), "eICU sepsis-3 re-derivation differs"
    e_h = T["shock_e"].reindex(ze.stay_id).values
    assert np.array_equal(np.isnan(e_h), ze.shock_h.isna().values) and np.allclose(e_h[~np.isnan(e_h)], ze.shock_h.dropna().values)
    orc = syn.oracle_features_eicu(ze[["stay_id"]], T["vp"], T["va"], T["el"], T["es"])
    frames_equal(ze.sort_values("stay_id").reset_index(drop=True), orc.sort_values("stay_id").reset_index(drop=True), cols, "eICU features")
    r = ze[ze.stay_id == 1]
    if len(r):
        r = r.iloc[0]; assert r.hr_first == 112.0, r.hr_first
    print("PASS builders vs oracle (eICU-CRD): cohort rules, shock onset, all feature columns")
    return zm, ze


def test_selfchecks(tmp, T, zm, ze):
    msgs = []; say = msgs.append
    cf = pd.read_parquet(f"{tmp}/cohort_final_mimic.parquet"); fm = pd.read_parquet(f"{tmp}/features_mimic.parquet"); fe = pd.read_parquet(f"{tmp}/features_eicu.parquet")
    zhou.check_shock_mimic(build_shock_all(tmp), cf, say)
    zhou.check_features("MIMIC", zm, fm, say)
    _, _, s3e, shock_e = zhou.build_eicu_z(tmp)
    zhou.check_eicu(s3e, shock_e, fe, say); zhou.check_features("eICU", ze, fe, say)
    assert sum("PASSED" in m for m in msgs) == 4, msgs
    n24 = int((fm.hours_icu_to_pred == 24).sum()); assert n24 > 10, n24
    # corrupted primary tables must stop the run
    bad = fm.copy(); bad.loc[bad.hours_icu_to_pred == 24, "hr_mean"] += 5
    for fn, args in ((zhou.check_features, ("MIMIC", zm, bad, say)),):
        try: fn(*args); raise AssertionError("corrupted features were accepted")
        except RuntimeError: pass
    bad = cf.copy(); idx = bad.index[bad.shock == 1][0]; bad.loc[idx, "hours_to_shock"] += 1
    try: zhou.check_shock_mimic(build_shock_all(tmp), bad, say); raise AssertionError("corrupted shock onset was accepted")
    except RuntimeError: pass
    bad = fe.copy(); bad.loc[bad.index[0], "hours_icu_to_pred"] += 1
    try: zhou.check_eicu(s3e, shock_e, bad, say); raise AssertionError("corrupted eICU index was accepted")
    except RuntimeError: pass
    print("PASS self-checks: accept the primary tables, stop on corrupted ones")


def build_shock_all(tmp):
    return zhou.build_mimic_z(tmp)[2]


def test_no_leakage(tmp, T):
    zm0, _, _ = zhou.build_mimic_z(tmp)
    ch, lb, sf = pd.read_parquet(f"{tmp}/chart_sub.parquet"), pd.read_parquet(f"{tmp}/lab_sub.parquet"), pd.read_parquet(f"{tmp}/sofa_hourly.parquet")
    z = zm0[["stay_id", "subject_id", "t0"]].copy(); z["pred"] = z.t0 + pd.Timedelta(hours=24)
    chp = ch.merge(z[["stay_id", "pred"]], on="stay_id", how="left"); lbp = lb.merge(z[["subject_id", "pred"]].drop_duplicates("subject_id"), on="subject_id", how="left")
    sfp = sf.merge(z[["stay_id", "pred"]], on="stay_id", how="left")
    assert len(chp) == len(ch) and len(lbp) == len(lb) and len(sfp) == len(sf) and (chp.index == ch.index).all()
    ch2 = ch.copy(); ch2.loc[chp.index[chp.charttime >= chp.pred], "valuenum"] = 999.0
    lb2 = lb.copy(); lb2.loc[lbp.index[lbp.charttime >= lbp.pred], "valuenum"] = 999.0
    sf2 = sf.copy()
    for c in ["lactate", "any_pressor", "sofa", "s_cv", "s_resp"]: sf2.loc[sfp.index[sfp.hr >= sfp.pred], c] = 99
    assert (chp.charttime >= chp.pred).sum() > 1000 and (lbp.charttime >= lbp.pred).sum() > 1000 and (sfp.hr >= sfp.pred).sum() > 1000
    d2 = tempfile.mkdtemp()
    for f in os.listdir(tmp): shutil.copy(f"{tmp}/{f}", d2)
    ch2.to_parquet(f"{d2}/chart_sub.parquet"); lb2.to_parquet(f"{d2}/lab_sub.parquet"); sf2.to_parquet(f"{d2}/sofa_hourly.parquet")
    zm1, _, _ = zhou.build_mimic_z(d2)
    cols = zhou.AGG_COLS + [c for c in zhou.FIRST_COLS if c.endswith("_first")]
    a, b = zm0.sort_values("stay_id").reset_index(drop=True), zm1.sort_values("stay_id").reset_index(drop=True)
    for c in cols:
        x, y = a[c].astype(float), b[c].astype(float)
        assert ((x.isna() & y.isna()) | (x == y)).all(), f"future records changed {c}"
    # sensitivity of the test: records INSIDE the window must change the features
    ch3 = ch.copy(); ch3.loc[chp.index[(chp.charttime < chp.pred) & (chp.charttime >= chp.pred - pd.Timedelta(hours=24))], "valuenum"] = 999.0
    ch3.to_parquet(f"{d2}/chart_sub.parquet"); lb.to_parquet(f"{d2}/lab_sub.parquet"); sf.to_parquet(f"{d2}/sofa_hourly.parquet")
    zm2, _, _ = zhou.build_mimic_z(d2)
    assert (zm2.sort_values("stay_id").reset_index(drop=True).hr_mean.dropna() > 900).mean() > 0.9, "the leakage test cannot fail"
    shutil.rmtree(d2)
    print("PASS no future leakage (records at or after t0 + 24 h are never read; the test is sensitive)")


def test_pipeline(tmp):
    out = tempfile.mkdtemp(); [shutil.copy(f"{tmp}/{f}", out) for f in os.listdir(tmp)]
    zhou.main(["--out", out, "--nboot", "40", "--rf_trees", "60", "--folds", "3", "--seed", "1"])
    ZD = f"{out}/zhou_emulation"
    # second run: the L variant (without lactate only) must be written to separate files and must reproduce the shared rows exactly
    zhou.main(["--out", out, "--nboot", "40", "--rf_trees", "60", "--folds", "3", "--seed", "1", "--variants", "full,L", "--tag", "L"])
    for f in ("zhou_results_L.csv", "zhou_paired_differences_L.csv", "zhou_design_ns_L.csv", "zhou_overlap_exposure_L.csv", "zhou_log_L.txt"):
        assert os.path.exists(f"{ZD}/{f}"), f
    logL = open(f"{ZD}/zhou_log_L.txt").read()
    assert "reproducibility:" in logL and "WARNING" not in logL, [l for l in logL.split("\n") if "reproducib" in l]
    rL = pd.read_csv(f"{ZD}/zhou_results_L.csv"); assert set(rL.variant) == {"full", "L"}
    r1 = pd.read_csv(f"{ZD}/zhou_results.csv"); assert set(r1.variant) == {"full", "PL"}, "the second run overwrote the first"
    auL = rL[(rL.metric == "AUROC") & (rL.analysis == "E1") & (rL.dataset.str.startswith("MIMIC")) & (rL.predictors == "agg")].set_index("variant").value
    auP = r1[(r1.metric == "AUROC") & (r1.analysis == "E1") & (r1.dataset.str.startswith("MIMIC")) & (r1.predictors == "agg")].set_index("variant").value
    assert auL["L"] > auP["PL"] + 0.02 and auL["L"] <= auL["full"] + 0.02, (auL.to_dict(), auP.to_dict())   # removing lactate only loses less than removing pressor inputs as well
    for f in ("zhou_results.csv", "zhou_paired_differences.csv", "zhou_design_ns.csv", "zhou_overlap_exposure.csv", "zhou_flow.csv", "zhou_log.txt"):
        assert os.path.exists(f"{ZD}/{f}"), f
    r = pd.read_csv(f"{ZD}/zhou_results.csv"); au = r[r.metric == "AUROC"]
    val = lambda ds, ps, var, an: float(au[(au.dataset.str.startswith(ds)) & (au.predictors == ps) & (au.variant == var) & (au.analysis == an)].value.iloc[0])
    out_ = {}
    for ds in ("MIMIC", "eICU"):
        for ps in ("first", "agg"):
            e1, e2, pl = val(ds, ps, "full", "E1"), val(ds, ps, "full", "E2"), val(ds, ps, "PL", "E1")
            early, late = val(ds, ps, "full", "E1-on-early"), val(ds, ps, "full", "E1-on-late")
            out_[(ds, ps)] = (round(e1, 3), round(e2, 3), round(pl, 3), round(early, 3), round(late, 3))
            assert e1 > e2 + 0.05, (ds, ps, "E1 should exceed E2", e1, e2)
            assert pl < e1 - 0.05, (ds, ps, "removing P and L should lower E1", pl, e1)
            assert early > late + 0.05, (ds, ps, "early events should be easier than late events", early, late)
    ns = pd.read_csv(f"{ZD}/zhou_design_ns.csv"); assert (ns.events_early_onset_in_window > 0).all() and (ns.events_late > 0).all()
    ex = pd.read_csv(f"{ZD}/zhou_overlap_exposure.csv"); e = ex[ex.dataset == "MIMIC-IV"].set_index("group")
    assert e.loc["early events", "vasopressor_running_at_last_window_hour_%"] > 80 and e.loc["no event", "vasopressor_running_at_last_window_hour_%"] < 30, e
    shutil.rmtree(out)
    print("PASS pipeline: outputs written; E1 > E2, PL < full, early > late;", out_)


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(); T = syn.make(tmp)
    try:
        zm, ze = test_builders(tmp, T)
        test_selfchecks(tmp, T, zm, ze)
        test_no_leakage(tmp, T)
        test_pipeline(tmp)
        print("ALL TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
