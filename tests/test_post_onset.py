"""Test of scripts/post_onset_exposure.py on synthetic tables (no patient data), against an independent brute-force pandas implementation.
Run:  python tests/test_post_onset.py
The synthetic sofa tables lack the `lactate` column that the real sofa_hourly tables have (needed by the onset cross-check), so it is added here
from the known onset: lactate 3.5 from the onset hour on, 1.5 before."""
import os, sys, tempfile
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import synthetic_tables as syn, post_onset_exposure as po

TRAIN = po.TRAIN_GROUPS


def add_lactate(tmp):
    fm = pd.read_parquet(f"{tmp}/features_mimic.parquet"); cm = pd.read_parquet(f"{tmp}/cohort_final_mimic.parquet")
    d = fm.merge(cm[["stay_id", "index_time"]], on="stay_id"); d["onset"] = d.index_time + pd.to_timedelta(d.hours_to_shock, unit="h")
    s = pd.read_parquet(f"{tmp}/sofa_hourly.parquet").merge(d[["stay_id", "onset"]], on="stay_id")
    s["lactate"] = np.where(s.onset.notna() & (s.hr >= s.onset), 3.5, 1.5); s.drop(columns="onset").to_parquet(f"{tmp}/sofa_hourly.parquet")
    fe = pd.read_parquet(f"{tmp}/features_eicu.parquet"); fe["onset"] = (fe.hours_icu_to_pred - 1) + fe.hours_to_shock
    s = pd.read_parquet(f"{tmp}/sofa_hourly_eicu.parquet").merge(fe[["stay_id", "onset"]], on="stay_id")
    s["lactate"] = np.where(s.onset.notna() & (s.hr >= s.onset), 3.5, 1.5); s.drop(columns="onset").to_parquet(f"{tmp}/sofa_hourly_eicu.parquet")


def oracle_mimic(tmp, H, k):
    fm = pd.read_parquet(f"{tmp}/features_mimic.parquet"); cm = pd.read_parquet(f"{tmp}/cohort_final_mimic.parquet")
    d = fm.merge(cm[["stay_id", "index_time", "outtime"]], on="stay_id"); d = d[~d.anchor_year_group.isin(TRAIN)]
    d = d[d.outtime >= d.index_time + pd.Timedelta(hours=H + 1)]; d = d[(d.shock == 1) & (d.hours_to_shock <= H)]
    ch = pd.read_parquet(f"{tmp}/chart_sub.parquet"); ch = ch[ch.itemid.isin(po.M_ITEMS)]
    lb = pd.read_parquet(f"{tmp}/lab_sub.parquet"); lb = lb[lb.itemid.isin(po.M_LAB.values())]; so = pd.read_parquet(f"{tmp}/sofa_hourly.parquet")
    out = []
    for r in d.itertuples():
        on = r.index_time + pd.Timedelta(hours=float(r.hours_to_shock)); on_e = on + pd.Timedelta(hours=1)
        we = r.index_time + pd.Timedelta(hours=k); ws = we - pd.Timedelta(hours=24)
        c = ch[(ch.stay_id == r.stay_id) & (ch.charttime < we) & (ch.charttime >= ws)]
        l = lb[(lb.subject_id == r.subject_id) & (lb.charttime < we) & (lb.charttime >= ws)]
        pr = so[(so.stay_id == r.stay_id) & (so.hr == we - pd.Timedelta(hours=1))].any_pressor
        out.append(dict(span_from=(we - on) / pd.Timedelta(hours=1), v0=(c.charttime >= on).any(), v1=(c.charttime >= on_e).any(),
                        l0=(l.charttime >= on).any(), c0=((l.charttime >= on) & (l.itemid == 50813)).any(), c1=((l.charttime >= on_e) & (l.itemid == 50813)).any(),
                        pressor=bool(len(pr) and pr.iloc[0] > 0)))
    return pd.DataFrame(out)


def oracle_eicu(tmp, H, k):
    fe = pd.read_parquet(f"{tmp}/features_eicu.parquet"); ce = pd.read_parquet(f"{tmp}/cohort_eicu.parquet")
    d = fe.merge(ce, on="stay_id"); d["index_hr"] = d.hours_icu_to_pred - 1
    d = d[d.outoffset >= (d.index_hr + H + 1) * 60]; d = d[(d.shock == 1) & (d.hours_to_shock <= H)]
    vp = pd.read_parquet(f"{tmp}/eicu_vitalp_sub.parquet"); va = pd.read_parquet(f"{tmp}/eicu_vitala_sub.parquet"); lb = pd.read_parquet(f"{tmp}/eicu_lab_sub.parquet")
    vp = vp[vp[["heartrate", "respiration", "sao2", "temperature", "systemicsystolic", "systemicdiastolic", "systemicmean"]].notna().any(axis=1)][["stay_id", "t"]]
    va = va[va[["noninvasivesystolic", "noninvasivediastolic", "noninvasivemean"]].notna().any(axis=1)][["stay_id", "t"]]
    v = pd.concat([vp, va]); out = []
    for r in d.itertuples():
        on = (r.index_hr + r.hours_to_shock) * 60; we = (r.index_hr + k) * 60; ws = we - 1440
        c = v[(v.stay_id == r.stay_id) & (v.t < we) & (v.t >= ws)]; l = lb[(lb.stay_id == r.stay_id) & (lb.t < we) & (lb.t >= ws) & lb.v.notna()]
        out.append(dict(v0=(c.t >= on).any(), v1=(c.t >= on + 60).any(), l0=(l.t >= on).any(), c0=((l.t >= on) & (l.labname == "lactate")).any(),
                        c1=((l.t >= on + 60) & (l.labname == "lactate")).any()))
    return pd.DataFrame(out)


def main():
    tmp = tempfile.mkdtemp(); syn.make(tmp, n_m=500, n_e=250, seed=3); add_lactate(tmp)
    res = po.main(["--out", tmp]); g = lambda ds, H, des: res[(res.dataset.str.startswith(ds)) & (res.H == H) & (res.design == des)].iloc[0]
    for H in (6, 24):
        o = oracle_mimic(tmp, H, H + 1); r = g("MIMIC", H, "C")
        assert r.events == len(o) > 10, (r.events, len(o))
        for col, key in (("%any_vital_from_onset_hour", "v0"), ("%any_vital_strictly_after_onset_hour", "v1"), ("%any_lab_from_onset_hour", "l0"),
                         ("%lactate_from_onset_hour", "c0"), ("%lactate_strictly_after_onset_hour", "c1"), ("%vasopressor_running_at_last_predictor_hour", "pressor")):
            assert abs(r[col] - 100 * o[key].mean()) < 1e-9, (H, col, r[col], 100 * o[key].mean())
        assert abs(r.span_from_onset_h_median - o.span_from.median()) < 1e-9
        assert r["%sofa_input_hour_at_or_after_onset_hour"] == 100.0                       # by construction in design C
        b = g("MIMIC", H, "B")
        for col in b.index:
            if col.startswith("%") and ("vital" in col or "lab" in col or "lactate" in col or "sofa" in col):
                assert b[col] == 0.0, (H, col, b[col])                                    # design B: nothing from the onset hour onward
        e = g("eICU", H, "C"); oe = oracle_eicu(tmp, H, H + 1); assert e.events == len(oe) > 10
        for col, key in (("%any_vital_from_onset_hour", "v0"), ("%any_vital_strictly_after_onset_hour", "v1"), ("%any_lab_from_onset_hour", "l0"),
                         ("%lactate_from_onset_hour", "c0"), ("%lactate_strictly_after_onset_hour", "c1")):
            assert abs(e[col] - 100 * oe[key].mean()) < 1e-9, ("eICU", H, col, e[col], 100 * oe[key].mean())
        assert e["%sofa_input_hour_at_or_after_onset_hour"] == 100.0 and e.events > 10
    print("PASS post_onset_exposure: matches the brute-force oracle (MIMIC-IV and eICU-CRD, H = 6 and 24); design B has no post-onset information; design C always has the onset hour in its SOFA input")


if __name__ == "__main__":
    main()
