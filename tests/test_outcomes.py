"""Tests of the outcome sensitivity analysis (scripts/outcome_definitions.py and `ladder_analysis.py --outcome ...`) on synthetic tables.
No patient data.  Run:  python tests/test_outcomes.py
  1. onset rules vs an independent pandas implementation (timestamp and integer hour grids; random pressor/lactate series)
  2. relabel(): stays whose outcome is present at the index are removed, hours_to_shock is exact, shock = onset exists
  3. grid_gaps() detects a missing hour
  4. end-to-end: primary ladder run, then --outcome vaso_any and vaso_2h on a table with built-in brief vasopressor exposures
       - the re-derived primary outcome reproduces the labels of the primary features (the script stops otherwise)
       - vaso_any: stays with a pre-index exposure are removed and stays with a later 1-h exposure become events
       - vaso_2h: 1-h exposures are ignored, 2-h exposures are events
       - the landmark tables of the primary run are not modified; outputs are written to ladder_outcome_<name>/
       - summarize_outcome_sensitivity.py builds the comparison table with the pre-specified rules
"""
import os, sys, shutil, tempfile, subprocess, hashlib
import numpy as np, pandas as pd, duckdb

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import outcome_definitions as od
import synthetic_tables as syn


def oracle(df, rule):
    out = {}
    for sid, g in df.sort_values(["stay_id", "hr"]).groupby("stay_id"):
        p = g.any_pressor.values.astype(int); lac = g.lactate.values.astype(float); hr = g.hr.values; n = len(g)
        for i in range(n):
            if rule == "primary":
                lo, hi = max(0, i - 6), min(n, i + 7)
                w = lac[lo:hi]; ok = p[i] == 1 and np.nanmax(np.where(np.isnan(w), -np.inf, w)) > 2
            elif rule == "vaso_any": ok = p[i] == 1
            else: ok = p[i] == 1 and i + 1 < n and p[i + 1] == 1
            if ok: out[sid] = hr[i]; break
    return pd.Series(out, name="onset")


def test_rules():
    rng = np.random.default_rng(1)
    for is_ts in (True, False):
        rows = []
        for sid in range(1, 80):
            n = int(rng.integers(10, 40)); base = pd.Timestamp("2150-01-01") if is_ts else 0
            hr = [base + pd.Timedelta(hours=i) if is_ts else i for i in range(n)]
            p = (rng.random(n) < 0.12).astype(int); lac = np.where(rng.random(n) < 0.3, np.nan, rng.normal(2, 1.3, n))
            rows.append(pd.DataFrame({"stay_id": sid, "hr": hr, "lactate": lac, "any_pressor": p}))
        df = pd.concat(rows, ignore_index=True); con = duckdb.connect(); con.register("h", df)
        assert od.grid_gaps(con, "h", is_ts) == 0
        for rule in od.RULES:
            got = od.onset_frame(con, "h", rule, is_ts).set_index("stay_id")["onset"].sort_index()
            exp = oracle(df, rule).sort_index()
            assert list(got.index) == list(exp.index), (rule, is_ts, len(got), len(exp))
            assert (got.values == exp.values).all(), (rule, is_ts)
        # monotonic: onset(vaso_any) <= onset(primary)
        a = od.onset_frame(con, "h", "vaso_any", is_ts).set_index("stay_id").onset; b = od.onset_frame(con, "h", "primary", is_ts).set_index("stay_id").onset
        assert (a.reindex(b.index) <= b).all()
    print("1 rules vs oracle: OK")


def test_relabel_and_gaps():
    raw = pd.DataFrame({"stay_id": [1, 2, 3, 4], "shock": [1, 0, 0, 0], "hours_to_shock": [5.0, np.nan, np.nan, np.nan], "x": [1, 2, 3, 4]})
    idx = pd.Series([10, 10, 10, 10], index=[1, 2, 3, 4])
    onset = pd.DataFrame({"stay_id": [1, 2, 3], "onset": [14, 10, 17]})        # stay 2: onset == index -> removed; stay 1: 4 h; stay 3: 7 h
    out = od.relabel(raw, idx, onset, "t", log=lambda *a: None).set_index("stay_id")
    assert list(out.index) == [1, 3, 4] and out.loc[1, "hours_to_shock"] == 4 and out.loc[3, "shock"] == 1 and out.loc[4, "shock"] == 0 and np.isnan(out.loc[4, "hours_to_shock"])
    ts = pd.Timestamp("2150-01-01 00:00")
    idx = pd.Series([ts, ts], index=[1, 2]); raw = pd.DataFrame({"stay_id": [1, 2], "shock": [0, 0], "hours_to_shock": [np.nan, np.nan]})
    onset = pd.DataFrame({"stay_id": [1, 2], "onset": [ts + pd.Timedelta(hours=3), ts - pd.Timedelta(hours=1)]})
    out = od.relabel(raw, idx, onset, "t", log=lambda *a: None)
    assert list(out.stay_id) == [1] and out.hours_to_shock.iloc[0] == 3
    gap = pd.DataFrame({"stay_id": [1, 1, 1], "hr": [0, 1, 3], "lactate": 1.0, "any_pressor": 0}); con = duckdb.connect(); con.register("g", gap)
    assert od.grid_gaps(con, "g", False) == 1
    print("2-3 relabel and grid gaps: OK")


def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()


def run(args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ladder_analysis.py")] + args, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_end_to_end():
    tmp = tempfile.mkdtemp(prefix="outc_")
    try:
        syn.make(tmp, n_m=600, n_e=260, seed=3)
        rng = np.random.default_rng(5); marks = {}
        for ds, sofa_f, idx in (("mimic", "sofa_hourly", pd.read_parquet(f"{tmp}/cohort_final_mimic.parquet").set_index("stay_id").index_time),
                                ("eicu", "sofa_hourly_eicu", None)):
            s = pd.read_parquet(f"{tmp}/{sofa_f}.parquet"); feat = pd.read_parquet(f"{tmp}/features_{ds}.parquet")
            if idx is None: idx = (feat.set_index("stay_id").hours_icu_to_pred - 1)
            step = (lambda k: pd.Timedelta(hours=k)) if ds == "mimic" else (lambda k: k)
            shock_ids = set(feat.stay_id[feat.shock == 1])
            s["lactate"] = np.where(s.stay_id.isin(shock_ids) & (s.any_pressor == 1), 4.0, 1.2)      # primary rule: pressor and lactate > 2
            ns = [i for i in feat.stay_id if i not in shock_ids]; rng.shuffle(ns)
            pre, post1, post2 = ns[:40], ns[40:90], ns[90:120]
            keyed = s.set_index(["stay_id", "hr"])
            for sid in pre: keyed.loc[(sid, idx[sid] - step(2)), "any_pressor"] = 1                  # before the index -> removed under vaso_any
            for sid in post1: keyed.loc[(sid, idx[sid] + step(3)), "any_pressor"] = 1                # 1-h exposure at index + 3
            for sid in post2:
                keyed.loc[(sid, idx[sid] + step(5)), "any_pressor"] = 1; keyed.loc[(sid, idx[sid] + step(6)), "any_pressor"] = 1
            keyed.reset_index().to_parquet(f"{tmp}/{sofa_f}.parquet"); marks[ds] = (len(pre), len(post1), len(post2))
        common = ["--out", tmp, "--horizons", "6", "--nboot", "20", "--rf_trees", "30"]
        code, log = run(common); assert code == 0, log[-3000:]
        lm = {f: md5(os.path.join(tmp, "ladder", f)) for f in os.listdir(os.path.join(tmp, "ladder")) if f.endswith(".parquet")}
        prim = pd.read_csv(f"{tmp}/ladder/ladder_design_ns.csv")
        for rule in ("vaso_any", "vaso_2h"):
            code, log = run(common + ["--outcome", rule]); assert code == 0, log[-3000:]
            assert "reproduces shock and hours_to_shock of the primary features: OK" in log, log[-2000:]
            d = os.path.join(tmp, "ladder_outcome_" + rule); assert os.path.exists(f"{d}/ladder_results.csv") and os.path.exists(f"{d}/outcome_cohort_summary.csv")
            sm = pd.read_csv(f"{d}/outcome_cohort_summary.csv").set_index("dataset")
            for ds, nm in (("mimic", "MIMIC-IV"), ("eicu", "eICU-CRD")):
                npre, np1, np2 = marks[ds]
                if rule == "vaso_any":
                    assert sm.loc[nm, "stays_removed"] == npre, (rule, nm, sm.loc[nm].to_dict(), npre)
                    assert sm.loc[nm, "events_alt"] - sm.loc[nm, "events_primary_in_kept"] == np1 + np2, (rule, nm, sm.loc[nm].to_dict())
                else:
                    assert sm.loc[nm, "stays_removed"] == 0
                    assert sm.loc[nm, "events_alt"] - sm.loc[nm, "events_primary_in_kept"] == np2, (rule, nm, sm.loc[nm].to_dict())
            n_alt = pd.read_csv(f"{d}/ladder_design_ns.csv")
            assert (n_alt.events > prim.events).all() or rule == "vaso_2h"
        code, log = run(common + ["--outcome", "primary_kept_vaso_any"]); assert code == 0, log[-3000:]
        sm = pd.read_csv(f"{tmp}/ladder_outcome_primary_kept_vaso_any/outcome_cohort_summary.csv").set_index("dataset")
        for ds, nm in (("mimic", "MIMIC-IV"), ("eicu", "eICU-CRD")):       # control: same stays as vaso_any, primary labels
            assert sm.loc[nm, "stays_removed"] == marks[ds][0] and sm.loc[nm, "events_alt"] == sm.loc[nm, "events_primary_in_kept"], sm.loc[nm].to_dict()
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "summarize_outcome_sensitivity.py"), "--out", tmp], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        sm = pd.read_csv(f"{tmp}/outcome_sensitivity_summary.csv"); assert set(sm.outcome) == {"primary", "vaso_any", "vaso_2h", "primary_kept_vaso_any"} and len(sm) == 8
        assert {"R1 C-B CI>0", "R2 C-PL minus B-PL CI>0", "R3 |B - B(primary)|<0.05"} <= set(sm.columns)
        for f, h in lm.items(): assert md5(os.path.join(tmp, "ladder", f)) == h, f"landmark table modified: {f}"
        print("4 end-to-end (primary, vaso_any, vaso_2h; landmark tables untouched): OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_rules(); test_relabel_and_gaps(); test_end_to_end(); print("ALL OUTCOME TESTS PASSED")
