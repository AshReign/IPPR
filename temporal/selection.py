"""Selection rule for the dev search (Package 4, step 6).

1. R0 = the dev recall of T0.
2. For T1 and T2 separately, keep configurations with recall >= R0 and pick the
   highest F2; ties go to the higher recall, then to the config_id that sorts
   first (so the choice is fully deterministic).
3. If no configuration reaches R0, pick the highest-recall configuration
   (ties -> higher F2, then config_id) and flag it.

The rule only reads the search CSV, so anyone can reproduce the selection with
``scripts/select_configs.py``.
"""

from __future__ import annotations

import pandas as pd

RECALL_EPS = 1e-12   # treat recall equal to R0 up to floating-point noise as eligible


def select(df: pd.DataFrame) -> dict[str, dict]:
    """Return {"T0": row, "T1": row, "T2": row}; each row also has 'flag' and 'r0'."""
    t0 = df[df["condition"] == "T0"]
    if len(t0) != 1:
        raise ValueError(f"expected exactly one T0 row, found {len(t0)}")
    r0 = float(t0.iloc[0]["recall"])
    chosen = {"T0": {**t0.iloc[0].to_dict(), "flag": "", "r0": r0}}

    for cond in ("T1", "T2"):
        rows = df[df["condition"] == cond]
        if rows.empty:
            continue
        eligible = rows[rows["recall"] >= r0 - RECALL_EPS]
        if not eligible.empty:
            best = eligible.sort_values(["f2", "recall", "config_id"],
                                        ascending=[False, False, True]).iloc[0]
            flag = ""
        else:
            best = rows.sort_values(["recall", "f2", "config_id"],
                                    ascending=[False, False, True]).iloc[0]
            flag = "no config reached T0 recall; highest-recall config kept"
        chosen[cond] = {**best.to_dict(), "flag": flag, "r0": r0}
    return chosen
