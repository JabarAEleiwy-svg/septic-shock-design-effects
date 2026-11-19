#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
post_onset_exposure.py
======================
How much of the predictor window of design C lies AFTER the onset of septic shock?  (design B: none, by construction; reported as a check)

For every event stay of the evaluation sets of the controlled experiment (MIMIC-IV temporal test, eICU-CRD external; stays that remain in the
ICU until index + H + 1 h; events = shock with onset in (index, index + H], exactly as in ladder_analysis.py) it counts, for the predictor window
[end - 24 h, end) of each design (design B: end = index + 1 h; design C: end = index + H + 1 h):

  span_from_onset_h     hours from the START of the onset hour to the end of the window (= H + 1 - hours_to_shock for design C; 0 when the window ends before the onset)
  span_after_onset_h    hours from the END of the onset hour to the end of the window (= H - hours_to_shock; 0 if the onset hour is the last hour)
  vitals / labs / lactate   at least one charted vital sign / laboratory value / lactate at or after the START of the onset hour, inside the window
  vitals / labs / lactate (strictly after)   the same, at or after the END of the onset hour (certainly after the vasopressor had started)
  sofa_hour_at_or_after_onset   the SOFA / vasopressor inputs of the design are taken at hour end - 1 h; is that hour >= the onset hour?
  vasopressor_running_at_last_predictor_hour   reproduces Table A2 of the manuscript at H = 6 h (design C 85.7% / 74.4%, design B 4.3% / 2.5%): a built-in check

Inputs (in --out, parquet): features_mimic, features_eicu, cohort_final_mimic, cohort_eicu, chart_sub, lab_sub, sofa_hourly,
                            eicu_vitalp_sub, eicu_vitala_sub, eicu_lab_sub, sofa_hourly_eicu      (the files of ladder_analysis.py)
Outputs (in --out/post_onset): post_onset_exposure.csv (aggregated only), post_onset_log.txt.  No patient-level file is written.
Onset times are cross-checked against outcome_definitions.onset_frame (the primary rule re-derived from the hourly tables); the run stops on a mismatch.
"""
import argparse, os, sys
import numpy as np, pandas as pd
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import outcome_definitions as od

TRAIN_GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016"]
M_ITEMS = [220045, 220210, 220277, 220179, 220050, 220180, 220051, 220181, 220052, 225312, 223762, 223761]
M_LAB = {"creat": 50912, "bili": 50885, "plt": 51265, "lactate": 50813, "wbc": 51301, "hgb": 51222, "k": 50971, "na": 50983, "bicarb": 50882,
         "ph": 50820, "pao2": 50821, "base_excess": 50802}
E_LAB = {"creat": "creatinine", "bili": "total bilirubin", "plt": "platelets x 1000", "lactate": "lactate", "wbc": "wbc x 1000", "hgb": "hgb",
         "k": "potassium", "na": "sodium", "bicarb": "bicarbonate", "ph": "ph", "pao2": "pao2", "base_excess": "base excess"}


def pct(x): return float(100 * np.mean(x)) if len(x) else np.nan


def summarise(ds, H, design, ev):
    """ev: one row per event stay with columns span_from, span_after, v0, v1, l0, l1, c0, c1, sofa_ok, sofa_after, pressor"""
    q = lambda s, p: float(np.percentile(s, p)) if len(s) else np.nan
    return {"dataset": ds, "H": H, "design": design, "events": len(ev),
            "span_from_onset_h_median": q(ev.span_from, 50), "span_from_onset_h_q1": q(ev.span_from, 25), "span_from_onset_h_q3": q(ev.span_from, 75),
            "span_after_onset_h_median": q(ev.span_after, 50), "span_after_onset_h_q1": q(ev.span_after, 25), "span_after_onset_h_q3": q(ev.span_after, 75),
            "%onset_hour_is_last_hour": pct(ev.span_after == 0),
            "%any_vital_from_onset_hour": pct(ev.v0), "%any_lab_from_onset_hour": pct(ev.l0), "%lactate_from_onset_hour": pct(ev.c0),
            "%any_vital_strictly_after_onset_hour": pct(ev.v1), "%any_lab_strictly_after_onset_hour": pct(ev.l1), "%lactate_strictly_after_onset_hour": pct(ev.c1),
            "%vital_or_lab_from_onset_hour": pct(ev.v0 | ev.l0), "%vital_or_lab_strictly_after_onset_hour": pct(ev.v1 | ev.l1),
            "%sofa_input_hour_at_or_after_onset_hour": pct(ev.sofa_ok), "%sofa_input_hour_strictly_after_onset_hour": pct(ev.sofa_after),
            "%vasopressor_running_at_last_predictor_hour": pct(ev.pressor)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--horizons", default="6,24")
    a = ap.parse_args(argv); OUT = a.out; HZ = [int(x) for x in a.horizons.split(",") if x.strip()]
    ODIR = os.path.join(OUT, "post_onset"); os.makedirs(ODIR, exist_ok=True); LOG = open(os.path.join(ODIR, "post_onset_log.txt"), "w")
    def say(*x):
        s = " ".join(str(i) for i in x); print(s); LOG.write(s + "\n"); LOG.flush()
    R = lambda f: f"read_parquet('{OUT}/{f}.parquet')"
    con = duckdb.connect(); con.execute("SET memory_limit='8GB'"); con.execute("SET threads=2")
    rows = []

    # ============================================================ MIMIC-IV
    fm = pd.read_parquet(f"{OUT}/features_mimic.parquet")[["stay_id", "subject_id", "anchor_year_group", "shock", "hours_to_shock"]]
    cm = pd.read_parquet(f"{OUT}/cohort_final_mimic.parquet")[["stay_id", "index_time", "outtime"]]
    d = fm.merge(cm, on="stay_id")
    assert (d.index_time == d.index_time.dt.floor("h")).all(), "index_time is not on the hourly grid"
    on = od.onset_frame(con, R("sofa_hourly"), "primary", True).set_index("stay_id").onset
    ev_all = d[d.shock == 1]
    chk = (ev_all.index_time + pd.to_timedelta(ev_all.hours_to_shock, unit="h")).values == on.reindex(ev_all.stay_id).values
    if not chk.all(): sys.exit(f"MIMIC-IV: onset = index_time + hours_to_shock does not match the onset re-derived from sofa_hourly in {(~chk).sum()} stays; stop.")
    say(f"MIMIC-IV: onset cross-check passed ({len(ev_all):,} event stays)")
    test = d[~d.anchor_year_group.isin(TRAIN_GROUPS)]
    for H in HZ:
        kC = H + 1
        keep = test[test.outtime >= test.index_time + pd.Timedelta(hours=kC)]
        ev = keep[(keep.shock == 1) & (keep.hours_to_shock <= H)].copy()
        ev["onset"] = ev.index_time + pd.to_timedelta(ev.hours_to_shock, unit="h"); ev["onset_end"] = ev.onset + pd.Timedelta(hours=1)
        say(f"MIMIC-IV test, H={H}: {len(keep):,} stays in the analysis set, {len(ev):,} events")
        for design, k in (("C", kC), ("B", 1)):
            e = ev.copy(); e["wend"] = e.index_time + pd.Timedelta(hours=k); e["wstart"] = e.wend - pd.Timedelta(hours=24)
            con.register("e", e[["stay_id", "subject_id", "onset", "onset_end", "wend", "wstart"]])
            def cnt(tbl_sql):
                return con.execute(tbl_sql).df().set_index("stay_id")
            ch = cnt(f"""SELECT e.stay_id,
                           COUNT(*) FILTER (WHERE c.charttime >= e.onset) AS n0, COUNT(*) FILTER (WHERE c.charttime >= e.onset_end) AS n1
                         FROM e JOIN {R('chart_sub')} c ON c.stay_id = e.stay_id
                         WHERE c.itemid IN ({','.join(map(str, M_ITEMS))}) AND c.valuenum IS NOT NULL AND c.charttime < e.wend AND c.charttime >= e.wstart GROUP BY e.stay_id""")
            lb = cnt(f"""SELECT e.stay_id,
                           COUNT(*) FILTER (WHERE l.charttime >= e.onset) AS n0, COUNT(*) FILTER (WHERE l.charttime >= e.onset_end) AS n1,
                           COUNT(*) FILTER (WHERE l.charttime >= e.onset AND l.itemid = {M_LAB['lactate']}) AS c0,
                           COUNT(*) FILTER (WHERE l.charttime >= e.onset_end AND l.itemid = {M_LAB['lactate']}) AS c1
                         FROM e JOIN {R('lab_sub')} l ON l.subject_id = e.subject_id
                         WHERE l.itemid IN ({','.join(map(str, M_LAB.values()))}) AND l.valuenum IS NOT NULL AND l.charttime < e.wend AND l.charttime >= e.wstart GROUP BY e.stay_id""")
            so = cnt(f"""SELECT e.stay_id, s.any_pressor FROM e JOIN {R('sofa_hourly')} s ON s.stay_id = e.stay_id AND s.hr = e.wend - INTERVAL '1 hour'""")
            r = pd.DataFrame({"stay_id": e.stay_id.values}).set_index("stay_id")
            r["span_from"] = np.maximum(0, ((e.wend - e.onset) / pd.Timedelta(hours=1)).values); r["span_after"] = np.maximum(0, ((e.wend - e.onset_end) / pd.Timedelta(hours=1)).values)
            for nm, tab, col in (("v0", ch, "n0"), ("v1", ch, "n1"), ("l0", lb, "n0"), ("l1", lb, "n1"), ("c0", lb, "c0"), ("c1", lb, "c1")):
                r[nm] = (tab[col].reindex(r.index).fillna(0) > 0).values
            r["sofa_ok"] = ((e.wend - pd.Timedelta(hours=1)) >= e.onset).values; r["sofa_after"] = ((e.wend - pd.Timedelta(hours=1)) >= e.onset_end).values
            r["pressor"] = (so.any_pressor.reindex(r.index).fillna(0) > 0).values
            rows.append(summarise("MIMIC-IV temporal test", H, design, r))
        con.unregister("e")

    # ============================================================ eICU-CRD  (time in minutes / hours since unit admission)
    fe = pd.read_parquet(f"{OUT}/features_eicu.parquet")[["stay_id", "shock", "hours_to_shock", "hours_icu_to_pred"]]
    ce = pd.read_parquet(f"{OUT}/cohort_eicu.parquet")[["stay_id", "outoffset"]]
    d = fe.merge(ce, on="stay_id"); d["index_hr"] = d.hours_icu_to_pred - 1
    on = od.onset_frame(con, R("sofa_hourly_eicu"), "primary", False).set_index("stay_id").onset
    ev_all = d[d.shock == 1]
    if not ((ev_all.index_hr + ev_all.hours_to_shock).values == on.reindex(ev_all.stay_id).values).all():
        sys.exit("eICU-CRD: onset = index_hr + hours_to_shock does not match the onset re-derived from sofa_hourly_eicu; stop.")
    say(f"eICU-CRD: onset cross-check passed ({len(ev_all):,} event stays)")
    for H in HZ:
        kC = H + 1
        keep = d[d.outoffset >= (d.index_hr + kC) * 60]
        ev = keep[(keep.shock == 1) & (keep.hours_to_shock <= H)].copy()
        ev["onset_h"] = ev.index_hr + ev.hours_to_shock
        say(f"eICU-CRD external, H={H}: {len(keep):,} stays in the analysis set, {len(ev):,} events")
        for design, k in (("C", kC), ("B", 1)):
            e = ev.copy(); e["wend_h"] = e.index_hr + k
            e["onset"] = e.onset_h * 60; e["onset_end"] = (e.onset_h + 1) * 60; e["wend"] = e.wend_h * 60; e["wstart"] = e.wend - 1440
            con.register("e", e[["stay_id", "onset", "onset_end", "wend", "wstart", "wend_h"]])
            ch = con.execute(f"""WITH v AS (
                      SELECT stay_id, t FROM {R('eicu_vitalp_sub')} WHERE COALESCE(heartrate, respiration, sao2, temperature, systemicsystolic, systemicdiastolic, systemicmean) IS NOT NULL
                      UNION ALL
                      SELECT stay_id, t FROM {R('eicu_vitala_sub')} WHERE COALESCE(noninvasivesystolic, noninvasivediastolic, noninvasivemean) IS NOT NULL)
                    SELECT e.stay_id, COUNT(*) FILTER (WHERE v.t >= e.onset) AS n0, COUNT(*) FILTER (WHERE v.t >= e.onset_end) AS n1
                    FROM e JOIN v ON v.stay_id = e.stay_id WHERE v.t < e.wend AND v.t >= e.wstart GROUP BY e.stay_id""").df().set_index("stay_id")
            lb = con.execute(f"""SELECT e.stay_id, COUNT(*) FILTER (WHERE l.t >= e.onset) AS n0, COUNT(*) FILTER (WHERE l.t >= e.onset_end) AS n1,
                           COUNT(*) FILTER (WHERE l.t >= e.onset AND l.labname = '{E_LAB['lactate']}') AS c0,
                           COUNT(*) FILTER (WHERE l.t >= e.onset_end AND l.labname = '{E_LAB['lactate']}') AS c1
                         FROM e JOIN {R('eicu_lab_sub')} l ON l.stay_id = e.stay_id
                         WHERE l.labname IN ({','.join("'" + v + "'" for v in E_LAB.values())}) AND l.v IS NOT NULL AND l.t < e.wend AND l.t >= e.wstart GROUP BY e.stay_id""").df().set_index("stay_id")
            so = con.execute(f"SELECT e.stay_id, s.any_pressor FROM e JOIN {R('sofa_hourly_eicu')} s ON s.stay_id = e.stay_id AND s.hr = e.wend_h - 1").df().set_index("stay_id")
            r = pd.DataFrame({"stay_id": e.stay_id.values}).set_index("stay_id")
            r["span_from"] = np.maximum(0, ((e.wend - e.onset) / 60).values); r["span_after"] = np.maximum(0, ((e.wend - e.onset_end) / 60).values)
            for nm, tab, col in (("v0", ch, "n0"), ("v1", ch, "n1"), ("l0", lb, "n0"), ("l1", lb, "n1"), ("c0", lb, "c0"), ("c1", lb, "c1")):
                r[nm] = (tab[col].reindex(r.index).fillna(0) > 0).values
            r["sofa_ok"] = ((e.wend_h - 1) >= e.onset_h).values; r["sofa_after"] = ((e.wend_h - 1) >= e.onset_h + 1).values
            r["pressor"] = (so.any_pressor.reindex(r.index).fillna(0) > 0).values
            rows.append(summarise("eICU-CRD external", H, design, r))
        con.unregister("e")

    res = pd.DataFrame(rows); res.to_csv(os.path.join(ODIR, "post_onset_exposure.csv"), index=False)
    say("\n" + res.round(1).T.to_string())
    say("\nBuilt-in check against Table A2 of the manuscript (events with a vasopressor running at the last predictor hour, H = 6 h): "
        "design C 85.7% (MIMIC-IV) and 74.4% (eICU-CRD); design B 4.3% and 2.5%. If these do not reproduce, the windows are misaligned: do not use the other numbers.")
    say("Design B must show 0% for every vital / lab / lactate / SOFA-hour quantity from the onset hour onward (its window ends before the onset hour).")
    return res


if __name__ == "__main__":
    main()
