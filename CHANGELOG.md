# Changelog

## v1.3.0 (DOI to be added after the release)

- `scripts/window_cumulative.py` (new): window designs that separate the end of the predictor window from its length (B, B_wide, C, C_star, B_short, C_short, each also without vasopressor-derived and lactate inputs), paired cluster-bootstrap contrasts, three random-forest seeds, data-availability table, and the lead-time analysis with symmetric conditioning of non-events (evaluation only). The rebuilt features are checked against the primary and ladder feature tables before any model is fitted. Options `--test_groups` and `--tag` restrict the MIMIC-IV test set to given anchor-year groups (calendar-period sensitivity).
- `scripts/revision_audits.py` (new): (1) lactate measurement by outcome and the stays with a vasopressor after the index hour but no lactate within +-6 h of any vasopressor hour (cannot be classified as shock); (2) AUROC of the primary models by anchor-year group; (3) eICU-CRD recording of vasopressors and lactate by hospital and AUROC in hospital subsets. The primary outcome is re-derived from the hourly tables and must reproduce the labels, otherwise the run stops. Only aggregated outputs are written.
- `scripts/outcome_definitions.py`, `scripts/ladder_analysis.py`: new outcome rule `--outcome primary_post` (and the control `primary_kept_primary_post`): a vasopressor hour after the index hour and a lactate >2 mmol/L within +-6 h of it charted after the index hour, so that no lactate in the predictor window of design B can contribute to the label. Same stays, predictors and models; only the label differs. Existing rules and outputs are unchanged.
- `tests/test_edges.py` (new): records placed exactly at the window edges are in or out as intended in both builders. `tests/test_outcomes.py`: end-to-end test of `primary_post` added.

## v1.2.0 (DOI 10.5281/zenodo.24053027)

- `scripts/outcome_definitions.py` (new) and `scripts/ladder_analysis.py --outcome {primary,vaso_any,vaso_2h}`: sensitivity analysis of the outcome definition without any lactate criterion (vasopressor only, or vasopressor for at least two consecutive hours). Same stays and predictors; only the label changes; stays with the alternative outcome present at the index are removed. The default (`primary`) is unchanged. The run stops unless the primary outcome re-derived from the hourly table reproduces the primary labels exactly.
- `--outcome primary_kept_vaso_any|primary_kept_vaso_2h`: control that keeps the primary labels on the stays retained under the alternative definition (separates stay selection from outcome definition).
- `scripts/summarize_outcome_sensitivity.py` (new): one summary table and three decision rules fixed before the runs.
- `tests/test_outcomes.py` (new): synthetic-data tests, including an end-to-end run.

## v1.1.0 (DOI 10.5281/zenodo.23966353)

- `scripts/ladder_analysis.py`: added the matched ablations B-P, B-L and B-PL (design B without the vasopressor-derived inputs, without lactate, without both) and the paired contrasts C-P minus B-P, C-L minus B-L and C-PL minus B-PL (`ladder_paired_differences.csv`, comparisons labelled "(matched inputs)").
- `scripts/ladder_analysis.py`: added `--model {RF,LightGBM}`. LightGBM uses the fixed hyper-parameters of `extended_analysis_v3.py` and handles missing values natively; its outputs go to `ladder_lightgbm/`. The landmark feature tables are shared (`ladder/`).
- The results of designs B, C, C-P, C-L and C-PL are unchanged (each design is fitted independently with the same seed).
- `scripts/sequence_analysis.py` (new): the controlled experiment with a GRU on the hourly 24-h time course (designs B, C, B-PL, C-PL), with a self-check against the primary feature tables; `tests/` with synthetic-data tests (bin edges, self-check, absence of look-ahead, pipeline).
- `scripts/zhou_design_analysis.py` (new): emulation of the design of Zhou et al. (first-24-h predictors, shock at any time during the stay) inside the project cohort, with the decomposition into early and late events, the vasopressor/lactate ablation, and self-checks S1-S3 against the primary tables; `tests/test_zhou.py` and `tests/synthetic_zhou.py` (cohort rules, shock onset and all features against an independent implementation, leakage, pipeline).
- `scripts/zhou_design_analysis.py`: `--variants` (full, PL, L = without lactate only) and `--tag` (separate result files); a second run reports whether it reproduces the shared `full` rows of the first run exactly.
- `requirements.txt`: added torch.

## v1.0.0

First release (DOI 10.5281/zenodo.23823508).
