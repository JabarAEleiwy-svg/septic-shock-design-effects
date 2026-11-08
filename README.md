# Septic shock prediction: design effects in MIMIC-IV and eICU-CRD

Code for the manuscript "How design choices inflate apparent performance in early septic shock prediction: a controlled analysis across MIMIC-IV and eICU-CRD".

**No patient-level data, predictions or trained models are in this repository.** MIMIC-IV v3.1 and eICU-CRD v2.0 are available only to credentialed users through PhysioNet (https://physionet.org). Anyone who runs this code needs their own approved access.

## Contents

| File | Purpose |
|---|---|
| `notebook/ICUmodelprediction.ipynb` | Builds the cohorts, hourly SOFA, outcome and features from the raw tables (DuckDB). Outputs are stripped. |
| `scripts/extended_analysis_v3.py` | Primary model (random forest) and comparators, discrimination, calibration, decision curves, thresholds, heterogeneity, dataset shift, ablations, subgroups, lead-time analyses, SHAP. This is the version used for the manuscript. |
| `scripts/ladder_analysis.py` | Controlled experiment: only the end of the predictor window changes (designs B, C, C-P, C-L, C-PL; horizons 6 and 24 h), plus matched ablations of the predictive design (B-P, B-L, B-PL: B without the vasopressor-derived inputs / without lactate / without both) and the paired contrasts C-P minus B-P, C-L minus B-L and C-PL minus B-PL. `--model RF` (default) or `--model LightGBM` selects the learner. |
| `scripts/outcome_definitions.py`, `scripts/summarize_outcome_sensitivity.py` | Sensitivity analysis of the outcome definition. `ladder_analysis.py --outcome vaso_any` (vasopressor only) or `--outcome vaso_2h` (vasopressor for at least two consecutive hours) re-labels the SAME stays and uses the SAME predictors, so that lactate is not part of the outcome; stays with the alternative outcome already present at the index are removed. The run first re-derives the primary outcome from the hourly table and stops unless it reproduces the labels of the primary features exactly. `summarize_outcome_sensitivity.py` collects the primary and alternative runs into `outcome_sensitivity_summary.csv` and applies three decision rules that were fixed before the runs (see the docstring). |
| `scripts/sequence_analysis.py` | The controlled experiment repeated with a sequence model (GRU) that reads the hourly time course of the 24 h before the prediction time (designs B, C, B-PL, C-PL; same cohort, outcome, split, analysis set and paired cluster bootstrap as `ladder_analysis.py`). It first builds the hourly tensors and checks (self-check, k = 1) that they reproduce the `*_last` and SOFA columns of `features_mimic/eicu.parquet`. |
| `tests/test_outcomes.py` | Tests of the outcome definitions on synthetic tables (rules against an independent pandas implementation, relabelling, hourly-grid check, and an end-to-end run of the primary and both alternative outcomes; the landmark tables of the primary run must stay unchanged). No patient data. Run `python tests/test_outcomes.py`. |
| `tests/test_sequence.py`, `tests/synthetic_tables.py` | Tests of `sequence_analysis.py` on synthetic tables with the schema of the project files (bin edges, self-check, no use of records at or after the prediction time, whole pipeline). No patient data. Run `python tests/test_sequence.py`. |
| `scripts/zhou_design_analysis.py` | EMULATION (not replication) of the design of Zhou et al. (Int J Med Inform 2026) in this project's cohort: predictors from the first 24 h of ICU stay ("first" value or mean/min/max/last), outcome = septic shock at any time (E1) versus shock after the window only (E2), with and without the vasopressor-derived and lactate inputs (PL), plus the E1 model evaluated on early and on late events. MIMIC-IV out-of-fold and eICU-CRD external; paired cluster bootstrap. Self-checks S1-S3 against the primary tables (the run stops if they fail). Absolute AUROCs are not comparable with the published 0.785 (SAPS II, OASIS, comorbidities, several exclusions and the hyper-parameters are not available or not specified). |
| `tests/test_zhou.py`, `tests/synthetic_zhou.py` | Tests of `zhou_design_analysis.py` on synthetic tables: cohort rules, shock onset and every feature column against an independent pandas implementation, self-checks accept good and reject corrupted primary tables, no use of records at or after t0 + 24 h, whole pipeline and direction of the effects. No patient data. |
| `scripts/diagnose_sofa.py`, `diagnose_cns.py`, `diagnose_gcs_missing.py`, `diagnose_gcs_raw.py` | Diagnostics of why SOFA and qSOFA performed at or below chance in MIMIC-IV. |
| `scripts/sensitivity_future_info.py` | Re-evaluates saved predictions in stays whose infection criterion was already fulfilled at the prediction time. |
| `requirements.txt` | Package versions of the run reported in the manuscript. |

## Order of execution

1. Run the notebook to produce, in the project folder, `cohort_final_mimic.parquet`, `features_mimic.parquet`, `features_eicu.parquet`, `sofa_hourly.parquet`, `sofa_hourly_eicu.parquet` and the other intermediate tables.
2. `python scripts/extended_analysis_v3.py --out <project folder> [--save_preds]`. The run flags a MISMATCH if the primary random forest does not give AUROC 0.778 (temporal test) and 0.783 (eICU-CRD) within 0.005. `--save_preds` writes patient-level predictions to `results/private/`; they must stay private.
3. `python scripts/ladder_analysis.py --out <project folder>` (random forest; outputs in `<project folder>/ladder/`), then `python scripts/ladder_analysis.py --out <project folder> --model LightGBM` (outputs in `<project folder>/ladder_lightgbm/`; the landmark feature tables in `ladder/` are reused). Designs B, C, C-P, C-L and C-PL must reproduce the values of the manuscript (for example, at H = 6 h in the MIMIC-IV test set, AUROC 0.849, 0.995, 0.889, 0.974 and 0.848); the run prints them first.
3a. `python scripts/ladder_analysis.py --out <project folder> --outcome vaso_any` and `--outcome vaso_2h` (after step 3; outputs in `<project folder>/ladder_outcome_vaso_any/` and `ladder_outcome_vaso_2h/`; the landmark tables of step 3 are reused and not changed), then `python scripts/summarize_outcome_sensitivity.py --out <project folder>`. Control for the stay selection: `--outcome primary_kept_vaso_any` and `--outcome primary_kept_vaso_2h` run the PRIMARY outcome on exactly the stays that remain under the alternative rule, so that the effect of removing stays can be separated from the effect of changing the outcome definition; the summary script includes them when their folders exist. At H = 6 h in the MIMIC-IV test set, design B should give AUROC 0.791 (`vaso_any`), 0.785 (`vaso_2h`), 0.856 (`primary_kept_vaso_any`) and 0.853 (`primary_kept_vaso_2h`); the summary script prints the rule check. Add `--model LightGBM` to the commands for the second learner.
3b. `python scripts/sequence_analysis.py --out <project folder> --only_build` (builds the tensors in `<project folder>/sequence/` and runs the self-check; they contain patient-level data and must stay private), then `python scripts/sequence_analysis.py --out <project folder>` (GPU runtime recommended; 4 designs x 2 horizons x 5 networks). Outputs: `sequence_results.csv`, `sequence_paired_differences.csv`, `sequence_training.csv`, `sequence_log.txt`.
3c. `python scripts/zhou_design_analysis.py --out <project folder> --only_build` (builds the emulated cohorts in `<project folder>/zhou_emulation/` and runs self-checks S1-S3; patient-level, keep private), then `python scripts/zhou_design_analysis.py --out <project folder>` (about 1 h on CPU; `--model LightGBM` for the sensitivity analysis). Then `python scripts/zhou_design_analysis.py --out <project folder> --variants full,L --tag L` adds the input set without lactate only (SOFA kept) to separate files; the shared `full` rows must reproduce the first run exactly (the log says so).
4. `python scripts/diagnose_sofa.py`, `diagnose_cns.py`, `diagnose_gcs_missing.py`, `diagnose_gcs_raw.py` (each with `--out <project folder>`).
5. `python scripts/sensitivity_future_info.py --out <project folder> --mimic_base <MIMIC-IV folder> --eicu_src <eICU-CRD folder> --preds_dir <project folder>/results/private`.

The scripts were run in Google Colab with the paths of the original project; adjust the `--out` and data paths to your environment. Random seed 42 is used throughout.

## Outputs

Aggregated tables and figures only (CSV, PNG, `results_summary.txt`, `results_status.json`). Every number in the manuscript is traced to an output file in its Evidence map.

## Data access, ethics and licence

- Data: PhysioNet credentialed data use agreements (MIMIC-IV, eICU-CRD).
- Licence: MIT (see `LICENSE`).
- Citation: cite the Zenodo record of the release you use. v1.0.0: 10.5281/zenodo.23823508; v1.1.0: 10.5281/zenodo.23966353; the DOI of v1.2.0 is added after its release.
