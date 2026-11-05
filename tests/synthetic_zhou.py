"""Synthetic tables with the SAME schema as the parquet files read by scripts/zhou_design_analysis.py (no patient data), plus
independent pandas implementations ("oracles") of the shock definition and of the 24-h window features.  Used only by tests/test_zhou.py.

Built-in structure (so that the pipeline has something to find and every rule of the script is exercised):
  * shock types: early (onset inside the first 24 h), late (after 24 h), very late (> 168 h), vasopressor without high lactate (no shock),
    high lactate without vasopressor (no shock), vasopressor first and lactate rising 10 h later (shock starts BEFORE the lactate peak: the +/-6-row window);
  * ICU stays shorter than 24 h / longer than 30 d, ages < 18 / > 89, second stays of the same subject, earlier non-sepsis stays, non-sepsis-3 stays;
  * records before the ICU admission and exactly at the window edges.
"""
import os
import numpy as np, pandas as pd

M_LAB = {"creat": 50912, "bili": 50885, "plt": 51265, "lactate": 50813, "wbc": 51301, "hgb": 51222,
         "k": 50971, "na": 50983, "bicarb": 50882, "ph": 50820, "pao2": 50821, "base_excess": 50802}
E_LAB = {"creat": "creatinine", "bili": "total bilirubin", "plt": "platelets x 1000", "lactate": "lactate", "wbc": "wbc x 1000",
         "hgb": "hgb", "k": "potassium", "na": "sodium", "bicarb": "bicarbonate", "ph": "ph", "pao2": "pao2", "base_excess": "base excess"}
BASE = {"creat": 1.1, "bili": 0.8, "plt": 220, "lactate": 1.3, "wbc": 11, "hgb": 11, "k": 4.1, "na": 138, "bicarb": 23, "ph": 7.38, "pao2": 95, "base_excess": -1}
MCHART = [(220045, "hr", 85, 10), (220210, "rr", 19, 4), (220277, "spo2", 96, 2), (220179, "sbp", 118, 14), (220180, "dbp", 65, 9),
          (220181, "map", 80, 10), (223761, "temp_f", 98.6, 1.2)]


# ------------------------------------------------------------------ oracles
def oracle_shock(h, hr="hr"):
    """first hour with a vasopressor and (max lactate over the 6 rows before .. 6 rows after) > 2; h: stay_id, hr, lactate, any_pressor"""
    out = {}
    for sid, g in h.sort_values(["stay_id", hr]).groupby("stay_id"):
        lw = g.lactate.rolling(13, center=True, min_periods=1).max()
        ok = (g.any_pressor == 1) & (lw > 2)
        if ok.any(): out[sid] = g[hr][ok].iloc[0]
    return pd.Series(out, name="shock_hr")


def oracle_features_mimic(z, chart, lab, sofa, win=24):
    """z: stay_id, subject_id, intime.  returns one row per stay with the agg + first columns, independently of duckdb"""
    rows = []
    cm = chart.groupby("stay_id"); lm = lab.groupby("subject_id"); sm = sofa.set_index(["stay_id", "hr"])
    for r in z.itertuples():
        t0 = r.intime.floor("h"); pred = t0 + pd.Timedelta(hours=win); o = {"stay_id": r.stay_id}
        c = cm.get_group(r.stay_id) if r.stay_id in cm.groups else chart.iloc[0:0]
        c = c[(c.charttime >= t0) & (c.charttime < pred)].sort_values("charttime")
        for item, name, _, _ in MCHART:
            nm = "temp" if name == "temp_f" else name
            v = c[c.itemid == item]
            vals = ((v.valuenum - 32) / 1.8) if name == "temp_f" else v.valuenum
            o.update({f"{nm}_mean": vals.mean() if len(v) else np.nan, f"{nm}_min": vals.min() if len(v) else np.nan,
                      f"{nm}_max": vals.max() if len(v) else np.nan, f"{nm}_last": vals.iloc[-1] if len(v) else np.nan,
                      f"{nm}_first": vals.iloc[0] if len(v) else np.nan})
        l = lm.get_group(r.subject_id) if r.subject_id in lm.groups else lab.iloc[0:0]
        l = l[(l.charttime >= t0) & (l.charttime < pred)].sort_values("charttime")
        for name, item in M_LAB.items():
            v = l[l.itemid == item].valuenum
            o.update({f"{name}_last": v.iloc[-1] if len(v) else np.nan, f"{name}_min": v.min() if len(v) else np.nan,
                      f"{name}_max": v.max() if len(v) else np.nan, f"{name}_first": v.iloc[0] if len(v) else np.nan})
        k = (r.stay_id, pred - pd.Timedelta(hours=1))
        if k in sm.index:
            s = sm.loc[k]
            o.update({"sofa_total": s.sofa, "s_resp": s.s_resp, "s_coag": s.s_coag, "s_liver": s.s_liver, "s_cv": s.s_cv, "s_cns": s.s_cns,
                      "s_renal": s.s_renal, "gcs_min": s.gcs_min, "pf_min": s.pf_min, "on_pressor": s.any_pressor})
        rows.append(o)
    return pd.DataFrame(rows)


def oracle_features_eicu(z, vp, va, lab, sofa, win=24):
    rows = []
    vv = pd.concat([vp.rename(columns={"heartrate": "hr", "respiration": "rr", "sao2": "spo2", "temperature": "temp", "systemicsystolic": "sbp",
                                       "systemicdiastolic": "dbp", "systemicmean": "map"}),
                    va.rename(columns={"noninvasivesystolic": "sbp", "noninvasivediastolic": "dbp", "noninvasivemean": "map"})], ignore_index=True)
    vg = vv.groupby("stay_id"); lg = lab.groupby("stay_id"); sm = sofa.set_index(["stay_id", "hr"])
    for r in z.itertuples():
        o = {"stay_id": r.stay_id}
        v = vg.get_group(r.stay_id) if r.stay_id in vg.groups else vv.iloc[0:0]
        v = v[(v.t >= 0) & (v.t < win * 60)].sort_values("t", kind="stable")
        for x in ["hr", "rr", "spo2", "sbp", "dbp", "map", "temp"]:
            s = v[x].dropna() if x in v else pd.Series(dtype=float)
            o.update({f"{x}_mean": s.mean() if len(s) else np.nan, f"{x}_min": s.min() if len(s) else np.nan, f"{x}_max": s.max() if len(s) else np.nan,
                      f"{x}_last": s.iloc[-1] if len(s) else np.nan, f"{x}_first": s.iloc[0] if len(s) else np.nan})
        l = lg.get_group(r.stay_id) if r.stay_id in lg.groups else lab.iloc[0:0]
        l = l[(l.t >= 0) & (l.t < win * 60)].sort_values("t", kind="stable")
        for name, nm in E_LAB.items():
            s = l[l.labname == nm].v
            o.update({f"{name}_last": s.iloc[-1] if len(s) else np.nan, f"{name}_min": s.min() if len(s) else np.nan,
                      f"{name}_max": s.max() if len(s) else np.nan, f"{name}_first": s.iloc[0] if len(s) else np.nan})
        k = (r.stay_id, win - 1)
        if k in sm.index:
            s = sm.loc[k]
            o.update({"sofa_total": s.sofa, "s_resp": s.s_resp, "s_coag": s.s_coag, "s_liver": s.s_liver, "s_cv": s.s_cv, "s_cns": s.s_cns,
                      "s_renal": s.s_renal, "gcs_min": s.gcs_min, "pf_min": s.pf_min, "on_pressor": s.any_pressor})
        rows.append(o)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ generator
def _hourly(rng, n, s, kind, L):
    """pressor / lactate / SOFA columns for rows h = 0 .. n-1 (hours from t0).  s = onset hour (shock types), L = stay length in hours"""
    pres = np.zeros(n, int); lac = np.where(rng.random(n) < 0.5, rng.normal(1.3, 0.2, n), np.nan)
    h = np.arange(n)
    if kind in ("early", "late", "vlate"):
        pres[h >= s] = 1; lac[(h >= s - 2) & (h <= s + 6)] = 3.5
    elif kind == "pressor_only":
        a = int(rng.integers(2, max(3, min(60, n - 2)))); pres[(h >= a) & (h < a + 30)] = 1
    elif kind == "lactate_only":
        a = int(rng.integers(2, max(3, min(60, n - 2)))); lac[(h >= a) & (h < a + 10)] = 3.0
    elif kind == "delayed":
        p0 = int(rng.integers(10, 40)); pres[h >= p0] = 1; lac[(h >= p0 + 10) & (h <= p0 + 14)] = 3.0
    return pres, lac


def make(out, n_m=700, n_e=300, seed=0):
    rng = np.random.default_rng(seed); os.makedirs(out, exist_ok=True)
    base = pd.Timestamp("2150-01-01")
    # ================================================================= MIMIC-IV like
    stays, extra, sofa, chart, lab = [], [], [], [], []
    subj = 0; prev = None; sid = 0
    kinds = ["early", "late", "vlate", "pressor_only", "lactate_only", "delayed", "none"]
    pk = [0.18, 0.18, 0.03, 0.10, 0.08, 0.06, 0.37]
    for i in range(n_m):
        sid += 1
        if prev is not None and rng.random() < 0.08:            # second stay of the same subject (excluded by the first-stay rule)
            sj = prev["subject_id"]; intime = prev["outtime"] + pd.Timedelta(days=5)
        else:
            subj += 1; sj = subj; intime = base + pd.Timedelta(days=int(rng.integers(0, 3000)), hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60)))
        u = rng.random(); L = int(rng.integers(30, 220)) if u < 0.94 else (int(rng.integers(6, 23)) if u < 0.97 else int(rng.integers(31 * 24, 33 * 24)))
        if i < 3: L = 120                                      # keep some stays regular
        outtime = intime + pd.Timedelta(hours=L, minutes=int(rng.integers(0, 50)))
        age = int(np.clip(rng.normal(65, 14), 18, 88)); a = rng.random()
        age = int(rng.integers(90, 92)) if a < 0.03 else (int(rng.integers(15, 18)) if a < 0.05 else age)
        kind = str(rng.choice(kinds, p=pk))
        if kind == "early": s = int(rng.integers(0, 24))
        elif kind == "late": s = int(rng.integers(24, 100))
        elif kind == "vlate": s = int(rng.integers(170, 190))
        else: s = -1
        n = int(np.floor((outtime - intime.floor("h")) / pd.Timedelta(hours=1))) + 1
        if s >= n - 3: kind, s = "none", -1
        t0 = intime.floor("h")
        pres, lac = _hourly(rng, n, s, kind, L)
        hrs = t0 + pd.to_timedelta(np.arange(n), unit="h")
        s_cv = np.where(pres == 1, 4, rng.choice([0, 0, 1], n)); comp = {c: rng.integers(0, 3, n) for c in ["s_resp", "s_renal"]}
        comp.update({c: rng.integers(0, 2, n) for c in ["s_coag", "s_liver", "s_cns"]})
        df = pd.DataFrame({"stay_id": sid, "hr": hrs, "lactate": lac, "any_pressor": pres, "pf_min": rng.normal(250, 60, n),
                           "gcs_min": np.where(rng.random(n) < 0.25, np.nan, rng.integers(8, 16, n).astype(float)), "s_cv": s_cv, **comp})
        df["sofa"] = df[["s_resp", "s_coag", "s_liver", "s_cv", "s_cns", "s_renal"]].sum(1)
        sofa.append(df)
        # chart: ICU records from 90 min before the hour of admission (the first ones lie before t0 and must be ignored) to discharge
        tt = pd.date_range(t0 - pd.Timedelta(minutes=90), outtime, freq="55min") + pd.to_timedelta(rng.integers(0, 600, 1)[0], unit="s")
        pre_late = np.zeros(len(tt), bool)
        if kind == "late": pre_late = (tt >= t0 + pd.Timedelta(hours=s - 6)) & (tt < t0 + pd.Timedelta(hours=s))
        for item, name, mu, sd in MCHART:
            keep = rng.random(len(tt)) < 0.8
            v = rng.normal(mu, sd, len(tt)) + (6 * pre_late if name == "hr" else 0)
            chart.append(pd.DataFrame({"stay_id": sid, "itemid": item, "charttime": tt[keep], "valuenum": v[keep]}))
        tl = pd.date_range(t0 - pd.Timedelta(hours=5), outtime, freq="173min") + pd.to_timedelta(int(rng.integers(0, 3000)), unit="s")
        for name, item in M_LAB.items():
            keep = rng.random(len(tl)) < 0.7; v = rng.normal(BASE[name], 0.15 * abs(BASE[name]) + 0.1, len(tl))
            if name == "lactate" and kind in ("early", "late", "vlate"): v = v + 2.2 * ((tl >= t0 + pd.Timedelta(hours=s - 2)) & (tl <= t0 + pd.Timedelta(hours=s + 6)))
            lab.append(pd.DataFrame({"subject_id": sj, "hadm_id": 10 * sj, "itemid": item, "charttime": tl[keep], "valuenum": v[keep]}))
        # sepsis-3 index: ~25 % exactly at hour 23 of the stay (so that the primary prediction time = t0 + 24 h)
        si = t0 + pd.Timedelta(hours=23, minutes=int(rng.integers(0, 60))) if rng.random() < 0.25 else t0 + pd.Timedelta(hours=int(rng.integers(0, 60)), minutes=int(rng.integers(0, 60)))
        in_s3 = rng.random() < 0.9
        rec = dict(subject_id=sj, hadm_id=10 * sj, stay_id=sid, intime=intime, outtime=outtime, los=(outtime - intime) / pd.Timedelta(days=1), si_time=si,
                   gender=str(rng.choice(["M", "F"])), race="X", hospital_expire_flag=int(rng.integers(0, 2)), age=age,
                   sofa2_time=si - pd.Timedelta(minutes=int(rng.integers(0, 300))), in_s3=in_s3)
        stays.append(rec); prev = rec
        if rng.random() < 0.04:                                 # an earlier non-sepsis ICU stay of the same subject (in cohort_mimic only)
            sid += 1; extra.append(dict(rec, stay_id=sid, intime=intime - pd.Timedelta(days=20), outtime=intime - pd.Timedelta(days=18), in_s3=False))
    allst = pd.DataFrame(stays + extra)
    cohort_mimic = allst.drop(columns=["in_s3", "sofa2_time"])
    s3 = allst[allst.in_s3].drop(columns=["in_s3"])
    sofa = pd.concat(sofa, ignore_index=True); chart = pd.concat(chart, ignore_index=True); lab = pd.concat(lab, ignore_index=True)
    # explicit window-edge records for stay 1
    z1 = stays[0]; t0 = z1["intime"].floor("h"); pred = t0 + pd.Timedelta(hours=24)
    edge = [(t0 - pd.Timedelta(seconds=1), 111.0), (t0, 112.0), (pred - pd.Timedelta(seconds=1), 113.0), (pred, 114.0)]
    chart = pd.concat([chart, pd.DataFrame([dict(stay_id=1, itemid=220045, charttime=t, valuenum=v) for t, v in edge])], ignore_index=True)
    lab = pd.concat([lab, pd.DataFrame([dict(subject_id=z1["subject_id"], hadm_id=10 * z1["subject_id"], itemid=50813, charttime=t, valuenum=v + 10) for t, v in edge])], ignore_index=True)
    shock = oracle_shock(sofa)
    # cohort_final_mimic: index = hour of max(si_time, sofa2_time, intime); shock must start after the index (the primary cohort)
    f = s3.copy(); f["index_time"] = f[["si_time", "sofa2_time", "intime"]].max(axis=1).dt.floor("h")
    f["shock_time"] = f.stay_id.map(shock)
    f = f[(f.index_time < f.outtime) & (f.shock_time.isna() | (f.shock_time > f.index_time))].copy()
    f["shock"] = f.shock_time.notna().astype(int); f["hours_to_shock"] = ((f.shock_time - f.index_time) / pd.Timedelta(hours=1)).round()
    cohort_final = f[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los", "si_time", "gender", "race", "hospital_expire_flag", "age", "sofa2_time",
                      "shock_time", "index_time", "shock", "hours_to_shock"]].copy()
    # primary feature table oracle (all stays of the primary cohort; prediction time = index + 1 h)
    fx = cohort_final[["stay_id", "subject_id", "intime", "index_time"]].copy(); fx["pred"] = fx.index_time + pd.Timedelta(hours=1)
    # the oracle is the same function with a per-stay window end -> reuse it by shifting: window end = pred
    feats = []
    for r in fx.itertuples():
        zz = pd.DataFrame([dict(stay_id=r.stay_id, subject_id=r.subject_id, intime=r.pred - pd.Timedelta(hours=24))])
        feats.append(oracle_features_mimic(zz, chart[chart.stay_id == r.stay_id], lab[lab.subject_id == r.subject_id], sofa[sofa.stay_id == r.stay_id]))
    fm = pd.concat(feats, ignore_index=True).merge(cohort_final[["stay_id", "shock", "hours_to_shock"]], on="stay_id")
    fm["hours_icu_to_pred"] = ((fx.set_index("stay_id").pred.reindex(fm.stay_id).values - fx.set_index("stay_id").intime.reindex(fm.stay_id).dt.floor("h").values) / pd.Timedelta(hours=1)).astype(int)
    fm.to_parquet(f"{out}/features_mimic.parquet")
    cohort_final.to_parquet(f"{out}/cohort_final_mimic.parquet"); s3.to_parquet(f"{out}/sepsis3_mimic.parquet"); cohort_mimic.to_parquet(f"{out}/cohort_mimic.parquet")
    chart.to_parquet(f"{out}/chart_sub.parquet"); lab.to_parquet(f"{out}/lab_sub.parquet"); sofa.to_parquet(f"{out}/sofa_hourly.parquet")

    # ================================================================= eICU-CRD like (minutes since unit admission)
    cohort, vp, va, el, es = [], [], [], [], []
    for sid in range(1, n_e + 1):
        u = rng.random(); Lh = int(rng.integers(30, 220)) if u < 0.93 else (int(rng.integers(6, 23)) if u < 0.97 else int(rng.integers(31 * 24, 33 * 24)))
        out_min = Lh * 60 + int(rng.integers(0, 59)); a = rng.random()
        age = float(np.clip(rng.normal(65, 14), 18, 88)); age = 90.0 if a < 0.04 else age
        kind = str(rng.choice(kinds, p=pk))
        s = int(rng.integers(0, 24)) if kind == "early" else int(rng.integers(24, 100)) if kind == "late" else int(rng.integers(170, 190)) if kind == "vlate" else -1
        nrow = min(out_min // 60, 168) + 25                       # hr = -24 .. min(outhr, 168)  -> number of rows
        hrs = np.arange(-24, -24 + nrow)
        if s >= hrs.max() - 3: kind, s = "none", -1
        pres, lac = _hourly(rng, len(hrs) + 24, s + 24 if s >= 0 else s, kind, Lh)       # shift so that index 24 = hour 0
        pres, lac = pres[:len(hrs)], lac[:len(hrs)]
        s_cv = np.where(pres == 1, 4, rng.choice([0, 0, 1], len(hrs)))
        zero_sofa = rng.random() < 0.10                          # never reaches SOFA >= 2 -> not a sepsis-3 stay
        comp = {c: (np.zeros(len(hrs), int) if zero_sofa else rng.integers(0, 3, len(hrs))) for c in ["s_resp", "s_renal"]}
        comp.update({c: (np.zeros(len(hrs), int) if zero_sofa else rng.integers(0, 2, len(hrs))) for c in ["s_coag", "s_liver", "s_cns"]})
        d = pd.DataFrame({"stay_id": sid, "hr": hrs, "lactate": lac, "any_pressor": pres, "pf_min": rng.normal(250, 60, len(hrs)),
                          "gcs_min": np.where(rng.random(len(hrs)) < 0.25, np.nan, rng.integers(8, 16, len(hrs)).astype(float)),
                          "s_cv": np.where(zero_sofa, 0, s_cv), **comp})
        d["any_pressor"] = np.where(zero_sofa, 0, d.any_pressor)
        d["sofa"] = d[["s_resp", "s_coag", "s_liver", "s_cv", "s_cns", "s_renal"]].sum(1)
        es.append(d)
        si_t = (23 * 60 + int(rng.integers(0, 60))) if rng.random() < 0.25 else int(rng.integers(-48 * 60, 24 * 60))
        cohort.append(dict(stay_id=sid, hadm_id=sid, hospitalid=int(rng.integers(1, 9)), gender=str(rng.choice(["Male", "Female"])), race="X", dis_year=2014,
                           outoffset=out_min, hospitaldischargestatus="Alive", admissionweight=80.0, age_num=age, unitvisitnumber=1, si_t=si_t))
        t = np.arange(-300, out_min, 7) + int(rng.integers(0, 6)); pre = np.zeros(len(t), bool)
        if kind == "late": pre = (t >= (s - 6) * 60) & (t < s * 60)
        keep = rng.random(len(t)) < 0.8
        vp.append(pd.DataFrame({"stay_id": sid, "t": t[keep], "heartrate": rng.normal(85, 10, len(t))[keep] + 6 * pre[keep], "respiration": rng.normal(19, 4, len(t))[keep],
                                "sao2": rng.normal(96, 2, len(t))[keep], "temperature": np.where(rng.random(len(t)) < 0.9, np.nan, rng.normal(37, .5, len(t)))[keep],
                                "systemicsystolic": rng.normal(120, 15, len(t))[keep], "systemicdiastolic": rng.normal(65, 9, len(t))[keep], "systemicmean": rng.normal(80, 10, len(t))[keep]}))
        t2 = np.arange(-200, out_min, 41) + int(rng.integers(0, 6))
        va.append(pd.DataFrame({"stay_id": sid, "t": t2, "noninvasivesystolic": rng.normal(118, 14, len(t2)), "noninvasivediastolic": rng.normal(65, 9, len(t2)),
                                "noninvasivemean": rng.normal(80, 10, len(t2))}))
        tl = np.arange(-240, out_min, 190) + int(rng.integers(0, 30))
        for name, nm in E_LAB.items():
            keep = rng.random(len(tl)) < 0.7; v = rng.normal(BASE[name], 0.15 * abs(BASE[name]) + 0.1, len(tl))
            if name == "lactate" and kind in ("early", "late", "vlate"): v = v + 2.2 * ((tl >= (s - 2) * 60) & (tl <= (s + 6) * 60))
            el.append(pd.DataFrame({"stay_id": sid, "t": tl[keep], "labname": nm, "v": v[keep]}))
    cohort = pd.DataFrame(cohort); vp, va, el, es = [pd.concat(x, ignore_index=True) for x in (vp, va, el, es)]
    # explicit window-edge records for stay 1 (minutes): -1, 0, 1439, 1440
    vp = pd.concat([vp, pd.DataFrame([dict(stay_id=1, t=t, heartrate=v, respiration=np.nan, sao2=np.nan, temperature=np.nan, systemicsystolic=np.nan,
                                           systemicdiastolic=np.nan, systemicmean=np.nan) for t, v in [(-1, 111.0), (0, 112.0), (1439, 113.0), (1440, 114.0)]])], ignore_index=True)
    shock_e = oracle_shock(es)
    # primary eICU cohort / features oracle
    c2 = cohort.copy(); c2["si_hr"] = np.floor(c2.si_t / 60.0).astype(int)
    sofa2 = {}
    for sid_, g in es[es.sofa >= 2].groupby("stay_id"):
        si_hr = int(c2.loc[c2.stay_id == sid_, "si_hr"].iloc[0]); gg = g[(g.hr >= si_hr - 48) & (g.hr <= si_hr + 24)]
        if len(gg): sofa2[sid_] = int(gg.hr.min())
    c2["sofa2_hr"] = c2.stay_id.map(sofa2); c3 = c2[c2.sofa2_hr.notna()].copy()
    c3["index_hr"] = np.maximum.reduce([c3.si_hr.values, c3.sofa2_hr.values.astype(int), np.zeros(len(c3), int)])
    c3["shock_hr"] = c3.stay_id.map(shock_e); c3["outhr"] = np.floor(c3.outoffset / 60.0).astype(int)
    c3 = c3[(c3.index_hr < c3.outhr) & (c3.shock_hr.isna() | (c3.shock_hr > c3.index_hr))].copy()
    c3["shock"] = c3.shock_hr.notna().astype(int); c3["hours_to_shock"] = c3.shock_hr - c3.index_hr
    feats = []
    # window of the primary eICU table = [pred_min - 1440, pred_min) with pred_min = (index_hr + 1) * 60: shift the time axis of the records
    for r in c3.itertuples():
        sh = (r.index_hr + 1) * 60 - 1440
        feats.append(oracle_features_eicu(pd.DataFrame([dict(stay_id=r.stay_id)]),
                                          vp[vp.stay_id == r.stay_id].assign(t=lambda x: x.t - sh), va[va.stay_id == r.stay_id].assign(t=lambda x: x.t - sh),
                                          el[el.stay_id == r.stay_id].assign(t=lambda x: x.t - sh),
                                          es[es.stay_id == r.stay_id].assign(hr=lambda x: x.hr - (r.index_hr + 1 - 24)), win=24))
    fe = pd.concat(feats, ignore_index=True).merge(c3[["stay_id", "hospitalid", "shock", "hours_to_shock", "index_hr"]], on="stay_id")
    fe["hours_icu_to_pred"] = fe.index_hr + 1; fe = fe.drop(columns=["index_hr"])
    fe.to_parquet(f"{out}/features_eicu.parquet"); cohort.to_parquet(f"{out}/cohort_eicu.parquet")
    vp.to_parquet(f"{out}/eicu_vitalp_sub.parquet"); va.to_parquet(f"{out}/eicu_vitala_sub.parquet"); el.to_parquet(f"{out}/eicu_lab_sub.parquet"); es.to_parquet(f"{out}/sofa_hourly_eicu.parquet")
    return dict(chart=chart, lab=lab, sofa=sofa, shock_m=shock, s3=s3, cohort_mimic=cohort_mimic, vp=vp, va=va, el=el, es=es, shock_e=shock_e, cohort_e=cohort, c3=c3)
