"""Selection rule for the dev search (Package 4, step 6).

Agreed with Ash on 2026-10-23:

1. T0 tunes its own confidence threshold: pick the highest F2, then the higher
   recall, then the lower threshold. R0 = the dev recall of that T0 config.
2. For T1 and T2 separately, keep configurations with recall >= R0 and pick
   the highest F2. Ties go to the higher recall, then the shorter window, then
   the lower theta, then the config_id that sorts first (fully deterministic).
3. If no configuration reaches R0, pick the highest-recall configuration
   (ties -> higher F2, then the same keys) and flag it.

The rule only reads the search CSV, so anyone can reproduce the selection with
``scripts/select_configs.py``.
"""

from __future__ import annotations

import json
import math

import pandas as pd

RECALL_EPS = 1e-12   # treat recall equal to R0 up to floating-point noise as eligible


def _param(params, key: str) -> float:
    """Read one parameter from the JSON 'params' column; inf if absent."""
    if not isinstance(params, str):
        return math.inf
    value = json.loads(params).get(key)
    return math.inf if value is None else float(value)


def _with_keys(rows: pd.DataFrame) -> pd.DataFrame:
    params = rows["params"] if "params" in rows else pd.Series([None] * len(rows), rows.index)
    return rows.assign(_conf=[_param(p, "conf_thr") for p in params],
                       _window=[_param(p, "window_s") for p in params],
                       _theta=[_param(p, "theta") for p in params])


def _first(rows: pd.DataFrame, primary: list[str]) -> pd.Series:
    keys = primary + ["_conf", "_window", "_theta", "config_id"]
    asc = [False] * len(primary) + [True, True, True, True]
    return rows.sort_values(keys, ascending=asc).iloc[0]


def _clean(row: pd.Series) -> dict:
    return {k: v for k, v in row.to_dict().items() if not k.startswith("_")}


def select(df: pd.DataFrame) -> dict[str, dict]:
    """Return {"T0": row, "T1": row, "T2": row}; each row also has 'flag' and 'r0'."""
    df = _with_keys(df)
    t0 = df[df["condition"] == "T0"]
    if t0.empty:
        raise ValueError("no T0 rows: R0 cannot be computed")
    best_t0 = _first(t0, ["f2", "recall"])
    r0 = float(best_t0["recall"])
    chosen = {"T0": {**_clean(best_t0), "flag": "", "r0": r0}}

    for cond in ("T1", "T2"):
        rows = df[df["condition"] == cond]
        if rows.empty:
            continue
        eligible = rows[rows["recall"] >= r0 - RECALL_EPS]
        if not eligible.empty:
            best, flag = _first(eligible, ["f2", "recall"]), ""
        else:
            best = _first(rows, ["recall", "f2"])
            flag = "no config reached T0 recall; highest-recall config kept"
        chosen[cond] = {**_clean(best), "flag": flag, "r0": r0}
    return chosen
