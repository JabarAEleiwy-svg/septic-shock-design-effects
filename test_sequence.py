"""Tests of scripts/sequence_analysis.py on synthetic tables (no patient data).
Run:  python tests/test_sequence.py        (or: pytest tests/test_sequence.py)
  1. bin alignment      - hand-built events at the exact bin edges land in the intended bin (MIMIC-IV and eICU-CRD builders)
  2. self-check         - the k = 1 tensor reproduces the `*_last` and SOFA columns of an independently computed feature table
  3. no future leakage  - overwriting every record at or after the prediction time changes nothing; overwriting records inside
                          the window does change the tensor (so that the test can fail)
  4. pipeline + direction - the whole script runs; with a built-in signal, design C (window includes the onset) beats design B
"""
import os, sys, shutil, tempfile, importlib.util
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_tables as syn
spec = importlib.util.spec_from_file_location("seq", os.path.join(HERE, "..", "scripts", "sequence_analysis.py"))
seq = importlib.util.module_from_spec(spec); spec.loader.exec_module(seq)
CI = seq.CIDX
S = pd.Timedelta(seconds=1)


def nan_equal(a, b): return np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])


def test_bin_alignment_mimic(tmp):
    index = pd.Timestamp("2150-03-05 10:00:00"); pred = index + pd.Timedelta(hours=1)
    pd.DataFrame([dict(stay_id=1, subject_id=7, intime=pd.Timestamp("2150-03-05 08:20:00"), index_time=index, outtime=index + pd.Timedelta(days=3))]
                 ).to_parquet(f"{tmp}/cohort_final_mimic.parquet")
    labs = [(pred - pd.Timedelta(minutes=30), 1), (pred - pd.Timedelta(hours=1), 2), (pred - pd.Timedelta(hours=1) - S, 3),
            (pred - pd.Timedelta(hours=24), 4), (pred - pd.Timedelta(hours=24) - S, 5), (pred, 6), (pred + pd.Timedelta(minutes=10), 7)]
    pd.DataFrame([dict(subject_id=7, itemid=50813, charttime=t, valuenum=float(v)) for t, v in labs]).to_parquet(f"{tmp}/lab_sub.parquet")
    ch = [(pred - pd.Timedelta(minutes=10), 100), (pred - pd.Timedelta(minutes=59), 110), (pred - pd.Timedelta(hours=1) - S, 90), (pred, 150)]
    pd.DataFrame([dict(stay_id=1, itemid=220045, charttime=t, valuenum=float(v)) for t, v in ch]).to_parquet(f"{tmp}/chart_sub.parquet")
    rows = [(pred - pd.Timedelta(hours=24), 1), (pred - pd.Timedelta(hours=1), 2), (pred, 9), (pred - pd.Timedelta(hours=25), 8)]
    sf = pd.DataFrame([dict(stay_id=1, hr=h, sofa=v, s_resp=0, s_coag=0, s_liver=0, s_cv=0, s_cns=0, s_renal=0, gcs_min=15.0, pf_min=300.0, any_pressor=0) for h, v in rows])
    sf.to_parquet(f"{tmp}/sofa_hourly.parquet")
    ids, X, hrs = seq.build_mimic_seq(tmp, 1, pd.DataFrame({"stay_id": [1]}))
    lac = X[0, :, CI["lactate"]]
    assert lac[23] == 1 and lac[22] == 3 and lac[0] == 4, lac           # bin 23 = [pred-1h, pred): last value = 30 min before pred
    assert np.isnan(lac[1:22]).all() and np.sum(~np.isnan(lac)) == 3, lac   # pred-24h-1s, pred and later are excluded
    hr = X[0, :, CI["hr"]]
    assert hr[23] == 105 and hr[22] == 90 and np.sum(~np.isnan(hr)) == 2, hr   # mean in the bin; the record at pred is excluded
    so = X[0, :, CI["sofa_total"]]
    assert so[0] == 1 and so[23] == 2 and np.sum(~np.isnan(so)) == 2, so
    assert hrs[0] == 3, hrs                                                # duckdb date_diff('hour', 08:20, 11:00) = 3, as in ladder_analysis.py
    print("PASS bin alignment (MIMIC-IV)")


def test_bin_alignment_eicu(tmp):
    pd.DataFrame([dict(stay_id=1, outoffset=100000)]).to_parquet(f"{tmp}/cohort_eicu.parquet")
    pm = 3 * 60                                                          # hours_icu_to_pred = 3 -> index_hr 2, k = 1 -> pred_hr 3
    ev = [(pm - 1, 1), (pm - 60, 2), (pm - 61, 3), (pm - 1440, 4), (pm - 1441, 5), (pm, 6), (pm + 10, 7)]
    pd.DataFrame([dict(stay_id=1, t=t, labname="lactate", v=float(v)) for t, v in ev]).to_parquet(f"{tmp}/eicu_lab_sub.parquet")
    pd.DataFrame([dict(stay_id=1, t=t, heartrate=float(v), respiration=np.nan, sao2=np.nan, temperature=np.nan, systemicsystolic=np.nan,
                       systemicdiastolic=np.nan, systemicmean=np.nan) for t, v in [(pm - 10, 100), (pm - 59, 110), (pm - 61, 90), (pm, 150)]]
                 ).to_parquet(f"{tmp}/eicu_vitalp_sub.parquet")
    pd.DataFrame(columns=["stay_id", "t", "noninvasivesystolic", "noninvasivediastolic", "noninvasivemean"]).astype(
        {"stay_id": "int64", "t": "int64", "noninvasivesystolic": "float64", "noninvasivediastolic": "float64", "noninvasivemean": "float64"}
    ).to_parquet(f"{tmp}/eicu_vitala_sub.parquet")
    rows = [(3 - 24, 1), (3 - 1, 2), (3, 9), (3 - 25, 8)]
    pd.DataFrame([dict(stay_id=1, hr=h, sofa=v, s_resp=0, s_coag=0, s_liver=0, s_cv=0, s_cns=0, s_renal=0, gcs_min=15.0, pf_min=300.0, any_pressor=0)
                  for h, v in rows]).to_parquet(f"{tmp}/sofa_hourly_eicu.parquet")
    ids, X, hrs = seq.build_eicu_seq(tmp, 1, pd.DataFrame({"stay_id": [1], "hours_icu_to_pred": [3]}))
    lac = X[0, :, CI["lactate"]]
    assert lac[23] == 1 and lac[22] == 3 and lac[0] == 4 and np.sum(~np.isnan(lac)) == 3, lac
    hr = X[0, :, CI["hr"]]
    assert hr[23] == 105 and hr[22] == 90 and np.sum(~np.isnan(hr)) == 2, hr
    so = X[0, :, CI["sofa_total"]]
    assert so[0] == 1 and so[23] == 2 and np.sum(~np.isnan(so)) == 2, so
    assert hrs[0] == 3
    print("PASS bin alignment (eICU-CRD)")


def test_selfcheck_and_leakage(tmp):
    syn.make(tmp, n_m=250, n_e=120, seed=1)
    fm = pd.read_parquet(f"{tmp}/features_mimic.parquet").sort_values("stay_id").reset_index(drop=True)
    fe = pd.read_parquet(f"{tmp}/features_eicu.parquet").sort_values("stay_id").reset_index(drop=True)
    say = lambda *a: None
    ids, X, _ = seq.build_mimic_seq(tmp, 1, fm[["stay_id"]]); seq.selfcheck("mimic", ids, X, fm, say)
    ide, Xe, _ = seq.build_eicu_seq(tmp, 1, fe[["stay_id", "hours_icu_to_pred"]]); seq.selfcheck("eicu", ide, Xe, fe, say)
    print("PASS self-check against the independently computed feature tables")
    # a deliberately wrong table must be caught
    bad = fm.copy(); bad["lactate_last"] = bad["lactate_last"] + 1.0
    try:
        seq.selfcheck("bad", ids, X, bad, say); raise SystemExit("self-check failed to detect a wrong feature table")
    except RuntimeError: pass
    # leakage: overwrite everything at/after the prediction time (k = 1 and k = 7)
    coh = pd.read_parquet(f"{tmp}/cohort_final_mimic.parquet"); ec = fe.set_index("stay_id")
    for k in (1, 7):
        tmp2 = tempfile.mkdtemp(); [shutil.copy(f"{tmp}/{f}", tmp2) for f in os.listdir(tmp) if f.endswith(".parquet")]
        pred = coh.set_index("stay_id").index_time + pd.Timedelta(hours=k)
        ch = pd.read_parquet(f"{tmp}/chart_sub.parquet"); m = ch.charttime >= ch.stay_id.map(pred); ch.loc[m, "valuenum"] = 999.0; ch.to_parquet(f"{tmp2}/chart_sub.parquet")
        sub_pred = coh.set_index("subject_id").index_time + pd.Timedelta(hours=k)
        lb = pd.read_parquet(f"{tmp}/lab_sub.parquet"); m = lb.charttime >= lb.subject_id.map(sub_pred); lb.loc[m, "valuenum"] = 999.0; lb.to_parquet(f"{tmp2}/lab_sub.parquet")
        sf = pd.read_parquet(f"{tmp}/sofa_hourly.parquet"); m = sf.hr >= sf.stay_id.map(pred); sf.loc[m, ["sofa", "s_cv", "any_pressor"]] = 99; sf.to_parquet(f"{tmp2}/sofa_hourly.parquet")
        a = seq.build_mimic_seq(tmp, k, fm[["stay_id"]])[1]; b = seq.build_mimic_seq(tmp2, k, fm[["stay_id"]])[1]
        assert nan_equal(a, b), f"MIMIC-IV tensor depends on records at or after the prediction time (k={k})"
        # eICU
        pm = (ec.hours_icu_to_pred - 1 + k) * 60; ph = ec.hours_icu_to_pred - 1 + k
        for f, tcol, mapper, cols in (("eicu_vitalp_sub", "t", pm, ["heartrate", "systemicmean"]), ("eicu_vitala_sub", "t", pm, ["noninvasivemean"]),
                                      ("eicu_lab_sub", "t", pm, ["v"]), ("sofa_hourly_eicu", "hr", ph, ["sofa", "s_cv", "any_pressor"])):
            d = pd.read_parquet(f"{tmp}/{f}.parquet"); m = d[tcol] >= d.stay_id.map(mapper); d.loc[m, cols] = 999.0; d.to_parquet(f"{tmp2}/{f}.parquet")
        a = seq.build_eicu_seq(tmp, k, fe[["stay_id", "hours_icu_to_pred"]])[1]; b = seq.build_eicu_seq(tmp2, k, fe[["stay_id", "hours_icu_to_pred"]])[1]
        assert nan_equal(a, b), f"eICU-CRD tensor depends on records at or after the prediction time (k={k})"
        # sensitivity of the test: overwriting records INSIDE the window must change the tensor
        tmp3 = tempfile.mkdtemp(); [shutil.copy(f"{tmp}/{f}", tmp3) for f in os.listdir(tmp) if f.endswith(".parquet")]
        lb = pd.read_parquet(f"{tmp}/lab_sub.parquet"); m = (lb.charttime < lb.subject_id.map(sub_pred)) & (lb.charttime >= lb.subject_id.map(sub_pred) - pd.Timedelta(hours=24))
        lb.loc[m, "valuenum"] = 999.0; lb.to_parquet(f"{tmp3}/lab_sub.parquet")
        c = seq.build_mimic_seq(tmp3, k, fm[["stay_id"]])[1]; assert not nan_equal(seq.build_mimic_seq(tmp, k, fm[["stay_id"]])[1], c)
        shutil.rmtree(tmp2); shutil.rmtree(tmp3)
    print("PASS no future leakage (k = 1 and 7), and the test is sensitive to changes inside the window")


def test_pipeline_direction(tmp):
    syn.make(tmp, n_m=1500, n_e=500, seed=2)
    seq.main(["--out", tmp, "--horizons", "6", "--designs", "B,C,B-PL,C-PL", "--nboot", "60", "--seeds", "2", "--epochs", "12",
              "--patience", "4", "--device", "cpu"])
    sd = f"{tmp}/sequence"
    for f in ("sequence_results.csv", "sequence_paired_differences.csv", "sequence_design_ns.csv", "sequence_training.csv", "sequence_log.txt"):
        assert os.path.exists(f"{sd}/{f}"), f
    r = pd.read_csv(f"{sd}/sequence_results.csv"); r = r[r.metric == "AUROC"].set_index(["dataset", "code"]).value
    for ds in ("MIMIC temporal test", "eICU external"):
        assert r[(ds, "C")] > r[(ds, "B")] + 0.05, (ds, r[(ds, "C")], r[(ds, "B")])   # the window that includes the onset must discriminate better
        assert r[(ds, "C-PL")] < r[(ds, "C")] - 0.05, (ds,)                              # removing the inputs that encode the outcome lowers it
    d = pd.read_csv(f"{sd}/sequence_paired_differences.csv"); assert {"C minus B", "C-PL minus B-PL", "B-PL minus B"} <= set(d.comparison)
    print("PASS pipeline: outputs written; C > B and C-PL < C on both test sets;", {k: round(v, 3) for k, v in r.items()})


if __name__ == "__main__":
    for fn in (test_bin_alignment_mimic, test_bin_alignment_eicu, test_selfcheck_and_leakage, test_pipeline_direction):
        d = tempfile.mkdtemp()
        try: fn(d)
        finally: shutil.rmtree(d, ignore_errors=True)
    print("ALL TESTS PASSED")
