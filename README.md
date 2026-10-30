# Septic shock prediction: design effects in MIMIC-IV and eICU-CRD

Code for the manuscript "How design choices inflate apparent performance in early septic shock prediction: a controlled analysis across MIMIC-IV and eICU-CRD".

**No patient-level data, predictions or trained models are in this repository.** MIMIC-IV v3.1 and eICU-CRD v2.0 are available only to credentialed users through PhysioNet (https://physionet.org). Anyone who runs this code needs their own approved access.

## Contents

| File | Purpose |
|---|---|
| `notebook/ICUmodelprediction.ipynb` | Builds the cohorts, hourly SOFA, outcome and features from the raw tables (DuckDB). Outputs are stripped. |
| `scripts/extended_analysis_v3.py` | Primary model (random forest) and comparators, discrimination, calibration, decision curves, thresholds, heterogeneity, dataset shift, ablations, subgroups, lead-time analyses, SHAP. This is the version used for the manuscript. |
| `scripts/ladder_analysis.py` | Controlled experiment: only the end of the predictor window changes (designs B, C, C-P, C-L, C-PL; horizons 6 and 24 h). |
| `scripts/diagnose_sofa.py`, `diagnose_cns.py`, `diagnose_gcs_missing.py`, `diagnose_gcs_raw.py` | Diagnostics of why SOFA and qSOFA performed at or below chance in MIMIC-IV. |
| `scripts/sensitivity_future_info.py` | Re-evaluates saved predictions in stays whose infection criterion was already fulfilled at the prediction time. |
| `requirements.txt` | Package versions of the run reported in the manuscript. |

## Order of execution

1. Run the notebook to produce, in the project folder, `cohort_final_mimic.parquet`, `features_mimic.parquet`, `features_eicu.parquet`, `sofa_hourly.parquet`, `sofa_hourly_eicu.parquet` and the other intermediate tables.
2. `python scripts/extended_analysis_v3.py --out <project folder> [--save_preds]`. The run flags a MISMATCH if the primary random forest does not give AUROC 0.778 (temporal test) and 0.783 (eICU-CRD) within 0.005. `--save_preds` writes patient-level predictions to `results/private/`; they must stay private.
3. `python scripts/ladder_analysis.py --out <project folder>`.
4. `python scripts/diagnose_sofa.py`, `diagnose_cns.py`, `diagnose_gcs_missing.py`, `diagnose_gcs_raw.py` (each with `--out <project folder>`).
5. `python scripts/sensitivity_future_info.py --out <project folder> --mimic_base <MIMIC-IV folder> --eicu_src <eICU-CRD folder> --preds_dir <project folder>/results/private`.

The scripts were run in Google Colab with the paths of the original project; adjust the `--out` and data paths to your environment. Random seed 42 is used throughout.

## Outputs

Aggregated tables and figures only (CSV, PNG, `results_summary.txt`, `results_status.json`). Every number in the manuscript is traced to an output file in its Evidence map.

## Data access, ethics and licence

- Data: PhysioNet credentialed data use agreements (MIMIC-IV, eICU-CRD).
- Licence: to be chosen by the author when the repository is created.
- Citation: to be added with the Zenodo DOI after the first release.
