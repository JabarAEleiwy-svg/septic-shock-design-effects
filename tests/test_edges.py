"""Edge test of the `win` parameter: records placed exactly at the window edges must be in/out as intended (MIMIC-IV and eICU builders)."""
import os, sys, tempfile, importlib.util
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp()
S = pd.Timedelta(seconds=1)
index = pd.Timestamp("2150-03-05 10:00:00")
# --- MIMIC-like raw tables, one stay
pd.DataFrame([dict(stay_id=1, subject_id=7, intime=pd.Timestamp("2150-03-04 00:00:00"), index_time=index, outtime=index + pd.Timedelta(days=9))]).to_parquet(f"{tmp}/cohort_final_mimic.parquet")
def edge(k, win):   # prediction time for landmark k and the two edge instants of the window
    pred = index + pd.Timedelta(hours=k); return pred, pred - pd.Timedelta(hours=win)
pred1, _ = edge(1, 24)
labs = []   # (time, value): values encode position
for k_, win_ in ((1, 24), (1, 30), (7, 24), (7, 30), (1, 6), (7, 6)):
    pass
# lactate at a few absolute instants relative to the k=1 prediction time (11:00): -6h-1s, -6h, -24h-1s, -24h, -30h-1s, -30h, 0, +10min
t = {"m6h_1s": pred1 - pd.Timedelta(hours=6) - S, "m6h": pred1 - pd.Timedelta(hours=6), "m24h_1s": pred1 - pd.Timedelta(hours=24) - S, "m24h": pred1 - pd.Timedelta(hours=24),
     "m30h_1s": pred1 - pd.Timedelta(hours=30) - S, "m30h": pred1 - pd.Timedelta(hours=30), "p0": pred1, "p10m": pred1 + pd.Timedelta(minutes=10)}
vals = {"m6h_1s": 1., "m6h": 2., "m24h_1s": 3., "m24h": 4., "m30h_1s": 5., "m30h": 6., "p0": 7., "p10m": 8.}
pd.DataFrame([dict(subject_id=7, itemid=50813, charttime=t[n], valuenum=vals[n]) for n in t]).to_parquet(f"{tmp}/lab_sub.parquet")
pd.DataFrame([dict(stay_id=1, itemid=220045, charttime=pred1 - pd.Timedelta(hours=7), valuenum=100.)]).to_parquet(f"{tmp}/chart_sub.parquet")
hrs = pd.date_range(index - pd.Timedelta(hours=40), index + pd.Timedelta(hours=40), freq="h")
pd.DataFrame(dict(stay_id=1, hr=hrs, sofa=1, s_resp=0, s_coag=0, s_liver=0, s_cv=0, s_cns=0, s_renal=0, gcs_min=15.0, pf_min=300.0, any_pressor=0)).to_parquet(f"{tmp}/sofa_hourly.parquet")
# load the builders without running the script: exec the source up to the data section
src = open(os.path.join(HERE, "..", "scripts", "window_cumulative.py")).read().split("# ------------------------------------------------------------------ data")[0]
sys.argv = ["x", "--out", tmp]
ns = {"__name__": "w"}; exec(compile(src, "w", "exec"), ns)
bm = ns["build_mimic"]; st = pd.DataFrame({"stay_id": [1]})
def lac(k, win):
    r = bm(k, st, win).iloc[0]; return (r["lactate_min"], r["lactate_max"])
# k=1 (pred 11:00): win 24 -> [pred-24h, pred) contains m24h(4)... not m24h_1s(3): min=2? values inside: 2 (m6h),1? m6h_1s=pred-6h-1s is inside too (1)
assert lac(1, 24) == (1.0, 4.0), lac(1, 24)       # values 1,2,4 inside; 3 (24h+1s earlier), 5,6, 7 (=pred) and 8 outside
assert lac(1, 30) == (1.0, 6.0), lac(1, 30)       # adds 3 and 6 (m30h inclusive), still excludes 5
assert lac(1, 6) == (2.0, 2.0), lac(1, 6)         # [pred-6h, pred): only 2 (m6h inclusive); 1 is 1 s too early
# k=7: pred = 17:00, window [pred-win, pred): everything shifts; value 7 (11:00) and 8 (11:10) are now inside the window
a, b = lac(7, 24); assert (a, b) == (1.0, 8.0) or (a, b) == (1.0, 8.0), (a, b)
assert lac(7, 6) == (7.0, 8.0), lac(7, 6)         # [11:00, 17:00): 7 (inclusive) and 8
# eICU builder: times in minutes from unit admission; index_hr = 5, k=1 -> pred_min = 360
be = ns["build_eicu"]
E = pd.DataFrame({"stay_id": [1], "hours_icu_to_pred": [6]}); pm = (5 + 1) * 60
pd.DataFrame([dict(stay_id=1, t=pm - 24 * 60 - 1, labname="lactate", v=3.), dict(stay_id=1, t=pm - 24 * 60, labname="lactate", v=4.),
              dict(stay_id=1, t=pm - 30 * 60, labname="lactate", v=6.), dict(stay_id=1, t=pm - 30 * 60 - 1, labname="lactate", v=5.),
              dict(stay_id=1, t=pm - 360, labname="lactate", v=2.), dict(stay_id=1, t=pm, labname="lactate", v=7.)]).to_parquet(f"{tmp}/eicu_lab_sub.parquet")
pd.DataFrame(dict(stay_id=[1], t=[pm - 10], heartrate=[80.], respiration=[18.], sao2=[97.], temperature=[37.], systemicsystolic=[120.], systemicdiastolic=[70.], systemicmean=[85.])).to_parquet(f"{tmp}/eicu_vitalp_sub.parquet")
pd.DataFrame(dict(stay_id=[1], t=[pm - 10], noninvasivesystolic=[120.], noninvasivediastolic=[70.], noninvasivemean=[85.])).to_parquet(f"{tmp}/eicu_vitala_sub.parquet")
pd.DataFrame(dict(stay_id=[1] * 80, hr=np.arange(-30, 50), sofa=1, s_resp=0, s_coag=0, s_liver=0, s_cv=0, s_cns=0, s_renal=0, gcs_min=15.0, pf_min=300.0, any_pressor=0)).to_parquet(f"{tmp}/sofa_hourly_eicu.parquet")
def lace(k, win):
    r = be(k, E, win).iloc[0]; return (r["lactate_min"], r["lactate_max"])
assert lace(1, 24) == (2.0, 4.0), lace(1, 24)
assert lace(1, 30) == (2.0, 6.0), lace(1, 30)
assert lace(1, 6) == (2.0, 2.0), lace(1, 6)
assert lace(7, 6) == (7.0, 7.0), lace(7, 6)
print("window edge tests passed (MIMIC-IV and eICU-CRD builders, win = 6 / 24 / 30, k = 1 / 7)")
