#!/usr/bin/env python3
"""
summarize_outcome_sensitivity.py
================================
Collects the results of `ladder_analysis.py` (primary outcome, folder ladder/) and of `ladder_analysis.py --outcome vaso_any|vaso_2h`
(folders ladder_outcome_*/) into ONE table (outcome_sensitivity_summary.csv) and applies the decision rules below.
The rules were written BEFORE the sensitivity runs were done and must not be changed after seeing the results.

Pre-specified rules (per outcome definition, dataset and horizon; AUROC, random forest, paired cluster bootstrap 95% CI)
  R1  window effect:           C minus B has a 95% CI above 0.
  R2  not explained by lactate-free overlap removal: C-PL minus B-PL (matched inputs) has a 95% CI above 0
      (the window effect remains when the vasopressor-derived and lactate inputs are removed from both designs).
  R3  predictive level:        the AUROC of design B differs from that of the primary outcome by less than 0.05.
primary_kept_<rule> rows are the stay-selection CONTROL (primary outcome on the stays kept under <rule>); they are reported but not judged.
The design effect is called ROBUST to the outcome definition if R1 and R2 hold in every dataset and horizon;
R3 is reported as information about how far the level of performance depends on the definition.  Any failure is reported as such.
Usage:  python scripts/summarize_outcome_sensitivity.py --out <project folder> [--outcomes vaso_any,vaso_2h] [--model RF]
"""
import argparse, os, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--outcomes", default="vaso_any,vaso_2h,primary_kept_vaso_any,primary_kept_vaso_2h"); ap.add_argument("--model", default="RF", choices=["RF", "LightGBM"])
a = ap.parse_args()
suffix = "" if a.model == "RF" else "_" + a.model.lower()
dirs = {"primary": os.path.join(a.out, "ladder" + suffix)}
for o in [x for x in a.outcomes.split(",") if x]:
    if os.path.isdir(os.path.join(a.out, f"ladder_outcome_{o}{suffix}")) or not o.startswith("primary_kept_"): dirs[o] = os.path.join(a.out, f"ladder_outcome_{o}{suffix}")   # controls are optional
rows, flags = [], []
for name, d in dirs.items():
    for f in ("ladder_results.csv", "ladder_paired_differences.csv", "ladder_design_ns.csv"):
        if not os.path.exists(os.path.join(d, f)): sys.exit(f"missing {os.path.join(d, f)}")
    res = pd.read_csv(os.path.join(d, "ladder_results.csv")); dif = pd.read_csv(os.path.join(d, "ladder_paired_differences.csv"))
    ns = pd.read_csv(os.path.join(d, "ladder_design_ns.csv"))
    res = res[res.metric == "AUROC"]; dif = dif[dif.metric == "AUROC"]
    for (H, ds), g in res.groupby(["H", "dataset"]):
        n_ = ns[(ns.H == H) & (ns.dataset == ds)].iloc[0]
        r = {"outcome": name, "H": H, "dataset": ds, "n": int(n_.n), "events": int(n_.events), "prevalence": round(float(n_.prevalence), 4)}
        for c in ("B", "C", "B-PL", "C-PL"):
            x = g[g.code == c].iloc[0]; r[f"AUROC {c}"] = f"{x.value:.3f} ({x.lo:.3f}-{x.hi:.3f})"; r[f"_{c}"] = x.value
        dd = dif[(dif.H == H) & (dif.dataset == ds)]
        for lab, key in (("C minus B", "C minus B"), ("C-PL minus B-PL", "C-PL minus B-PL (matched inputs)")):
            x = dd[dd.comparison == key].iloc[0]; r[lab] = f"{x.difference:+.3f} ({x.lo:+.3f} to {x.hi:+.3f})"; r["_lo " + lab] = x.lo
        rows.append(r)
df = pd.DataFrame(rows)
prim = df[df.outcome == "primary"].set_index(["H", "dataset"])["_B"]
df["R1 C-B CI>0"] = df["_lo C minus B"] > 0
df["R2 C-PL minus B-PL CI>0"] = df["_lo C-PL minus B-PL"] > 0
df["R3 |B - B(primary)|<0.05"] = [abs(r["_B"] - prim.loc[(r.H, r.dataset)]) < 0.05 for _, r in df.iterrows()]
out = df.drop(columns=[c for c in df.columns if c.startswith("_")])
path = os.path.join(a.out, f"outcome_sensitivity_summary{suffix}.csv"); out.to_csv(path, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(out.to_string(index=False))
for o in [k for k in dirs if k != "primary" and not k.startswith("primary_kept_")]:
    s = out[out.outcome == o]; ok = bool(s["R1 C-B CI>0"].all() and s["R2 C-PL minus B-PL CI>0"].all())
    print(f"\n{o}: R1 in {int(s['R1 C-B CI>0'].sum())}/{len(s)}, R2 in {int(s['R2 C-PL minus B-PL CI>0'].sum())}/{len(s)}, R3 in {int(s['R3 |B - B(primary)|<0.05'].sum())}/{len(s)} "
          f"-> design effect {'ROBUST' if ok else 'NOT robust (report which rule failed)'} to this outcome definition")
print("\nwritten:", path)
