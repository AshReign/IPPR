"""Paired bootstrap over videos (Package 5, step 7).

Each resample draws the val videos with replacement (fixed seed). The same draw is used for
every method, which is what makes it paired: both sides of a difference are scored on exactly
the same videos. For each resample we pool TP/FP/FN per method over the drawn videos and take
the F2 difference. The interval is the 2.5th-97.5th percentile of those differences.

Frames within a video are not independent, so videos, not frames, are the resampling unit.
"""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from eval.evaluator import scores

PAIRS: List[Tuple[str, str]] = [("T1", "T0"), ("T2", "T0"), ("T2", "T1")]


def paired_bootstrap(counts: Dict[str, pd.DataFrame], pairs=PAIRS,
                     n_resamples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """counts: method -> eval.evaluator.video_counts() output (video_id, tp, fp, fn).
    Returns one row per pair (a, b) with the F2 difference a - b and its 95% interval."""
    methods = sorted({m for pair in pairs for m in pair})
    videos = list(counts[methods[0]].video_id)
    table = {}
    for m in methods:
        c = counts[m].set_index("video_id")
        if set(c.index) != set(videos) or len(c) != len(videos):
            raise ValueError(f"{m} is not scored on the same videos as {methods[0]}")
        table[m] = c.loc[videos, ["tp", "fp", "fn"]].to_numpy(np.float64)

    draws = np.random.default_rng(seed).integers(0, len(videos), (n_resamples, len(videos)))
    f2 = {m: scores(*table[m][draws].sum(axis=1).T)[3] for m in methods}

    rows = []
    for a, b in pairs:
        diff = f2[a] - f2[b]
        point = scores(*table[a].sum(axis=0))[3] - scores(*table[b].sum(axis=0))[3]
        rows.append({"comparison": f"{a}-{b}", "delta_f2": float(point),
                     "ci_low": float(np.percentile(diff, 2.5)),
                     "ci_high": float(np.percentile(diff, 97.5)),
                     "share_positive": float((diff > 0).mean()),
                     "n_videos": len(videos), "n_resamples": n_resamples, "seed": seed})
    return pd.DataFrame(rows)
