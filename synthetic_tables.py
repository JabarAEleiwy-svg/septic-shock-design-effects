"""Synthetic tables with the SAME schema as the parquet files of the project (no patient data).  Used only by tests/test_sequence.py.
The shock signal is built in on purpose: vasopressor use and lactate rise at shock onset, heart rate rises a few hours before it,
so that a design whose predictor window includes the onset (C) must discriminate better than one that ends before it (B)."""
import os
import numpy as np, pandas as pd

LABS_M = {"creat": 50912, "bili": 50885, "plt": 51265, "lactate": 50813, "wbc": 51301, "hgb": 51222,
          "k": 50971, "na": 50983, "bicarb": 50882, "ph": 50820, "pao2": 50821, "base_excess": 50802}
LABS_E = {"creat": "creatinine", "bili": "total bilirubin", "plt": "platelets x 1000", "lactate": "lactate", "wbc": "wbc x 1000",
          "hgb": "hgb", "k": "potassium", "na": "sodium", "bicarb": "bicarbonate", "ph": "ph", "pao2": "pao2", "base_excess": "base excess"}
BASE = {"creat": 1.1, "bili": 0.8, "plt": 220, "lactate": 1.5, "wbc": 11, "hgb": 11, "k": 4.1, "na": 138, "bicarb": 23, "ph": 7.38, "pao2": 95, "base_excess": -1}
SOFA = ["sofa", "s_resp", "s_coag", "s_liver", "s_cv", "s_cns", "s_renal", "gcs_min", "pf_min", "any_pressor"]


def _sofa_rows(rng, hrs, onset_hr, shock):
    """hrs: array of hour keys; onset_hr: hour key at which shock starts (or None)"""
    n = len(hrs); on = (hrs >= onset_hr) if (shock and onset_hr is not None) else np.zeros(n, bool)
    s_cv = np.where(on, 4, rng.choice([0, 0, 1], n)); s_resp = rng.integers(0, 3, n); s_renal = rng.integers(0, 3, n)
    df = pd.DataFrame({"s_resp": s_resp, "s_coag": rng.integers(0, 2, n), "s_liver": rng.integers(0, 2, n), "s_cv": s_cv,
                       "s_cns": rng.integers(0, 2, n), "s_renal": s_renal})
    df["sofa"] = df[["s_resp", "s_coag", "s_liver", "s_cv", "s_cns", "s_renal"]].sum(1)
    df["gcs_min"] = np.where(rng.random(n) < 0.25, np.nan, rng.integers(8, 16, n).astype(float))
    df["pf_min"] = rng.normal(250, 60, n); df["any_pressor"] = on.astype(int)
    return df


def make(out, n_m=1500, n_e=500, seed=0):
    rng = np.random.default_rng(seed); os.makedirs(out, exist_ok=True)
    # ------------------------------------------------------------ MIMIC-like
    base = pd.Timestamp("2150-01-01 00:00:00")
    stays = []
    for sid in range(1, n_m + 1):
        intime = base + pd.Timedelta(days=int(rng.integers(0, 3000)), hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))
        index = intime.floor("h") + pd.Timedelta(hours=int(rng.integers(1, 5)))
        shock = rng.random() < 0.3; hts = int(rng.integers(1, 13)) if shock else np.nan
        los = int(rng.integers(40, 80)) + (0 if not shock else int(hts))
        stays.append(dict(stay_id=sid, subject_id=sid, intime=intime, index_time=index, outtime=index + pd.Timedelta(hours=los),
                          shock=int(shock), hours_to_shock=hts, age=float(rng.normal(65, 15)), male=int(rng.integers(0, 2)),
                          anchor_year_group=rng.choice(["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"], p=[.25, .25, .25, .25])))
    cohort = pd.DataFrame(stays)
    chart, lab, sofa = [], [], []
    for r in cohort.itertuples():
        onset = r.index_time + pd.Timedelta(hours=float(r.hours_to_shock)) if r.shock else None
        t = pd.date_range(r.intime.floor("h") - pd.Timedelta(hours=2), r.outtime, freq="37min") + pd.to_timedelta(rng.integers(0, 600, 1)[0], unit="s")
        n = len(t); pre = np.zeros(n, bool) if onset is None else ((t >= onset - pd.Timedelta(hours=3)) & (t < onset))
        # items: hr, rr, spo2, nibp sys / dia / map, art sys, art dia, art map, temp F
        for itemid, mu, sd, pre_shift in [(220045, 85, 10, 6), (220210, 19, 4, 1), (220277, 96, 2, -1), (220179, 118, 14, -5),
                                          (220180, 65, 9, -3), (220181, 80, 10, -4), (220050, 120, 15, -5), (223761, 98.6, 1.2, 0.3)]:
            keep = rng.random(n) < 0.8
            v = rng.normal(mu, sd, n) + pre_shift * pre
            if itemid == 220045: v[rng.random(n) < 0.002] = 500                      # implausible value (must be cleaned)
            chart.append(pd.DataFrame({"stay_id": r.stay_id, "itemid": itemid, "charttime": t[keep], "valuenum": v[keep]}))
        tl = pd.date_range(r.intime.floor("h"), r.outtime, freq="173min") + pd.to_timedelta(int(rng.integers(0, 3000)), unit="s")
        for name, itemid in LABS_M.items():
            m = len(tl); keep = rng.random(m) < 0.7; v = rng.normal(BASE[name], 0.15 * abs(BASE[name]) + 0.1, m)
            if name == "lactate" and onset is not None: v = v + 3.0 * (tl >= onset - pd.Timedelta(minutes=30))
            lab.append(pd.DataFrame({"subject_id": r.subject_id, "itemid": itemid, "charttime": tl[keep], "valuenum": v[keep]}))
        hrs = pd.date_range(r.index_time - pd.Timedelta(hours=30), r.outtime.floor("h"), freq="h")
        s = _sofa_rows(rng, hrs.to_numpy(), onset.floor("h").to_datetime64() if onset is not None else None, bool(r.shock))
        s.insert(0, "hr", hrs); s.insert(0, "stay_id", r.stay_id); sofa.append(s)
    chart, lab, sofa = pd.concat(chart, ignore_index=True), pd.concat(lab, ignore_index=True), pd.concat(sofa, ignore_index=True)
    # oracle for the primary feature table at k = 1 (independent pandas implementation)
    feat = cohort[["stay_id", "subject_id", "anchor_year_group", "shock", "hours_to_shock", "age", "male"]].copy()
    pred = cohort.index_time + pd.Timedelta(hours=1)
    feat["hours_icu_to_pred"] = ((pred - cohort.intime.dt.floor("h")) / pd.Timedelta(hours=1)).astype(int)
    for name, itemid in LABS_M.items():
        d = lab[lab.itemid == itemid].merge(cohort[["subject_id"]].assign(pred=pred.values), on="subject_id")
        d = d[(d.charttime < d.pred) & (d.charttime >= d.pred - pd.Timedelta(hours=24))].sort_values("charttime")
        feat[f"{name}_last"] = feat.subject_id.map(d.groupby("subject_id").valuenum.last())
    sk = cohort[["stay_id"]].assign(hr=(pred - pd.Timedelta(hours=1)).values)
    feat = feat.merge(sofa.merge(sk, on=["stay_id", "hr"]).rename(columns={"sofa": "sofa_total", "any_pressor": "on_pressor"})
                      [["stay_id", "sofa_total", "s_resp", "s_coag", "s_liver", "s_cv", "s_cns", "s_renal", "gcs_min", "pf_min", "on_pressor"]],
                      on="stay_id", how="left")
    cohort[["stay_id", "subject_id", "intime", "index_time", "outtime"]].to_parquet(f"{out}/cohort_final_mimic.parquet")
    feat.to_parquet(f"{out}/features_mimic.parquet"); chart.to_parquet(f"{out}/chart_sub.parquet")
    lab.to_parquet(f"{out}/lab_sub.parquet"); sofa.to_parquet(f"{out}/sofa_hourly.parquet")
    # ------------------------------------------------------------ eICU-like (time in minutes / hours since unit admission)
    es = []
    for sid in range(1, n_e + 1):
        ih = int(rng.integers(1, 6)); shock = rng.random() < 0.25; hts = int(rng.integers(1, 13)) if shock else np.nan
        los_h = ih + int(rng.integers(40, 80)) + (0 if not shock else int(hts))
        es.append(dict(stay_id=sid, hospitalid=int(rng.integers(1, 13)), shock=int(shock), hours_to_shock=hts, age=float(rng.normal(65, 15)),
                       male=int(rng.integers(0, 2)), hours_icu_to_pred=ih + 1, outoffset=los_h * 60, index_hr=ih))
    ec = pd.DataFrame(es); vp, va, el, esofa = [], [], [], []
    for r in ec.itertuples():
        on_min = (r.index_hr + r.hours_to_shock) * 60 if r.shock else None
        t = np.arange(0, r.outoffset, 7) + rng.integers(0, 6)
        pre = np.zeros(len(t), bool) if on_min is None else ((t >= on_min - 180) & (t < on_min))
        keep = rng.random(len(t)) < 0.8
        vp.append(pd.DataFrame({"stay_id": r.stay_id, "t": t[keep], "heartrate": rng.normal(85, 10, len(t))[keep] + 6 * pre[keep],
                                "respiration": rng.normal(19, 4, len(t))[keep], "sao2": rng.normal(96, 2, len(t))[keep],
                                "temperature": np.where(rng.random(len(t)) < 0.9, np.nan, rng.normal(37, .5, len(t)))[keep],
                                "systemicsystolic": rng.normal(120, 15, len(t))[keep] - 5 * pre[keep], "systemicdiastolic": rng.normal(65, 9, len(t))[keep],
                                "systemicmean": rng.normal(80, 10, len(t))[keep] - 4 * pre[keep]}))
        t2 = np.arange(0, r.outoffset, 41) + rng.integers(0, 6)
        va.append(pd.DataFrame({"stay_id": r.stay_id, "t": t2, "noninvasivesystolic": rng.normal(118, 14, len(t2)), "noninvasivediastolic": rng.normal(65, 9, len(t2)),
                                "noninvasivemean": rng.normal(80, 10, len(t2))}))
        tl = np.arange(0, r.outoffset, 190) + rng.integers(0, 30)
        for name, nm in LABS_E.items():
            keep = rng.random(len(tl)) < 0.7; v = rng.normal(BASE[name], 0.15 * abs(BASE[name]) + 0.1, len(tl))
            if name == "lactate" and on_min is not None: v = v + 3.0 * (tl >= on_min - 30)
            el.append(pd.DataFrame({"stay_id": r.stay_id, "t": tl[keep], "labname": nm, "v": v[keep]}))
        hrs = np.arange(max(r.index_hr - 24, -24), r.outoffset // 60)
        s = _sofa_rows(rng, hrs, (r.index_hr + r.hours_to_shock) if r.shock else None, bool(r.shock))
        s.insert(0, "hr", hrs); s.insert(0, "stay_id", r.stay_id); esofa.append(s)
    vp, va, el, esofa = [pd.concat(x, ignore_index=True) for x in (vp, va, el, esofa)]
    fe = ec[["stay_id", "hospitalid", "shock", "hours_to_shock", "age", "male", "hours_icu_to_pred"]].copy()
    for name, nm in LABS_E.items():
        pm = (ec.set_index("stay_id").index_hr + 1) * 60
        d = el[el.labname == nm].assign(pm=lambda x: x.stay_id.map(pm)); d = d[(d.t < d.pm) & (d.t >= d.pm - 1440)].sort_values("t")
        fe[f"{name}_last"] = fe.stay_id.map(d.groupby("stay_id").v.last())
    sk = ec[["stay_id", "index_hr"]].rename(columns={"index_hr": "hr"})          # SOFA at hr = pred_hr - 1 = index_hr
    fe = fe.merge(esofa.merge(sk, on=["stay_id", "hr"]).rename(columns={"sofa": "sofa_total", "any_pressor": "on_pressor"})
                  [["stay_id", "sofa_total", "s_resp", "s_coag", "s_liver", "s_cv", "s_cns", "s_renal", "gcs_min", "pf_min", "on_pressor"]], on="stay_id", how="left")
    ec[["stay_id", "outoffset"]].to_parquet(f"{out}/cohort_eicu.parquet"); fe.to_parquet(f"{out}/features_eicu.parquet")
    vp.to_parquet(f"{out}/eicu_vitalp_sub.parquet"); va.to_parquet(f"{out}/eicu_vitala_sub.parquet")
    el.to_parquet(f"{out}/eicu_lab_sub.parquet"); esofa.to_parquet(f"{out}/sofa_hourly_eicu.parquet")
    return out
