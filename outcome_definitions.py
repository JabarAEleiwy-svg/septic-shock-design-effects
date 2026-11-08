#!/usr/bin/env python3
"""
outcome_definitions.py
======================
Alternative definitions of the septic-shock outcome, derived from the hourly table (sofa_hourly / sofa_hourly_eicu) with the
same SQL logic as the notebook.  Used by `ladder_analysis.py --outcome ...` for the sensitivity analysis whose question is:
"Does the effect of the predictor window and of the outcome-overlap inputs depend on the lactate criterion in the outcome?"

Rules (onset = the hour that is labelled as shock onset; the first hour that satisfies the rule)
  primary   any_pressor = 1 AND max(lactate) over the hours t-6 .. t+6 is > 2          (the manuscript definition)
  vaso_any  any_pressor = 1                                                            (no lactate in the outcome)
  vaso_2h   any_pressor = 1 at hour t AND at hour t+1                                   (no lactate; a vasopressor that
            runs for at least two consecutive hours, which excludes brief peri-procedural exposures)
Both alternatives do not use lactate at all.  onset(vaso_any) <= onset(primary) for every stay.

Input table columns: stay_id, hr, lactate, any_pressor.  `hr` is a TIMESTAMP on an hourly grid (MIMIC-IV) or an integer hour
(eICU-CRD); `hr_is_ts` tells which.  The grid must be contiguous per stay (checked by `grid_gaps`).
"""
import duckdb, pandas as pd

RULES = ("primary", "vaso_any", "vaso_2h")


def onset_frame(con, src, rule, hr_is_ts):
    """DataFrame(stay_id, onset) for stays that have an onset under `rule`.  `src` is a SQL table or read_parquet(...) expression."""
    if rule not in RULES: raise ValueError(rule)
    step = "INTERVAL '1 hour'" if hr_is_ts else "1"
    if rule == "primary":
        q = f"""SELECT stay_id, MIN(hr) AS onset FROM (
                  SELECT stay_id, hr, any_pressor,
                         MAX(lactate) OVER (PARTITION BY stay_id ORDER BY hr ROWS BETWEEN 6 PRECEDING AND 6 FOLLOWING) AS lac_win
                  FROM {src}) WHERE any_pressor = 1 AND lac_win > 2 GROUP BY stay_id"""
    elif rule == "vaso_any":
        q = f"SELECT stay_id, MIN(hr) AS onset FROM {src} WHERE any_pressor = 1 GROUP BY stay_id"
    else:
        q = f"""SELECT stay_id, MIN(hr) AS onset FROM (
                  SELECT stay_id, hr, any_pressor,
                         LEAD(any_pressor) OVER w AS nxt, LEAD(hr) OVER w AS nxt_hr
                  FROM {src} WINDOW w AS (PARTITION BY stay_id ORDER BY hr))
                WHERE any_pressor = 1 AND nxt = 1 AND nxt_hr = hr + {step} GROUP BY stay_id"""
    return con.execute(q).df()


def grid_gaps(con, src, hr_is_ts):
    """number of stays whose hourly grid has a gap (must be 0, otherwise the +-6-row window is not +-6 h)"""
    step = "INTERVAL '1 hour'" if hr_is_ts else "1"
    return int(con.execute(f"""SELECT COUNT(DISTINCT stay_id) FROM (
        SELECT stay_id, hr, LAG(hr) OVER (PARTITION BY stay_id ORDER BY hr) AS prv FROM {src}) WHERE prv IS NOT NULL AND hr <> prv + {step}""").fetchone()[0])


def relabel(raw, index, onset, rule_name, log=print):
    """Replace shock / hours_to_shock of `raw` (stay_id, shock, hours_to_shock, ...) by the outcome with the given onset.
    index: Series stay_id -> index hour (same type as onset: Timestamp or number).  Stays with an onset at or before the index
    are removed (they are already in shock at the index; same rule as the primary cohort).  Everything else is unchanged."""
    on = onset.set_index("stay_id")["onset"].reindex(raw.stay_id.values)
    idx = index.reindex(raw.stay_id.values)
    h = on.values - idx.values
    h = pd.Series(h, index=raw.index)
    h = h.dt.total_seconds() / 3600.0 if str(h.dtype).startswith("timedelta") else h.astype(float)
    h = h.round(6)
    drop = h.notna() & (h <= 0)
    out = raw.loc[~drop].copy()
    out["hours_to_shock"] = h[~drop].values
    out["shock"] = out.hours_to_shock.notna().astype(int)
    log(f"[{rule_name}] stays {len(raw):,}; removed because the outcome was already present at the index: {int(drop.sum()):,}; "
        f"kept {len(out):,}; outcome after index {int(out.shock.sum()):,} ({100*out.shock.mean():.1f}%)")
    return out
