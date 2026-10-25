"""Temporal metrics for single-rip videos: fragmentation and detection delay.

These functions do not do any mask matching. The evaluator (Package 5) decides, for each
annotated frame, whether the rip was matched (a true positive at IoU >= 0.5). This module
only looks at that sequence over time.

Definitions (A2 plan, Section 04):
- Track span: from the first to the last annotated frame where the rip is in the ground truth.
- Fragmentation: after the first correct detection, a run of one or more missed frames that is
  followed by another correct detection within the span. A miss at the end is not counted.
- Delay: time of the first correct detection minus the time of the first annotated appearance.
- Frames where the rip is absent from the ground truth are ignored.
"""
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

Frame = Tuple[float, bool, bool]   # (t_sec, gt_present, matched)


@dataclass
class TrackStats:
    video_id: str
    detected: bool
    events: int = 0                  # fragmentation events
    span_min: float = 0.0            # track span in minutes
    delay_s: Optional[float] = None  # seconds to first correct detection


def track_stats(video_id: str, frames: Iterable[Frame]) -> TrackStats:
    """Fragmentation and delay for the single rip in one video."""
    seq = sorted((t, bool(m)) for t, g, m in frames if g)
    if not seq or not any(m for _, m in seq):
        return TrackStats(video_id, detected=False)

    t_first, t_last = seq[0][0], seq[-1][0]
    first_hit = next(i for i, (_, m) in enumerate(seq) if m)
    events, in_gap = 0, False
    for _, matched in seq[first_hit:]:
        if not matched:
            in_gap = True
        elif in_gap:                 # a gap closed by a re-detection
            events += 1
            in_gap = False
    return TrackStats(video_id, detected=True, events=events,
                      span_min=(t_last - t_first) / 60.0,
                      delay_s=seq[first_hit][0] - t_first)


def summarise(stats: List[TrackStats]) -> dict:
    """Pool per-video results into the numbers reported in the results table."""
    det = [s for s in stats if s.detected]
    events = sum(s.events for s in det)
    span = sum(s.span_min for s in det)
    return {
        "n_tracks": len(stats),
        "n_detected": len(det),
        "frag_raw": events,
        "frag_per_track": events / len(det) if det else float("nan"),
        "frag_per_min": events / span if span > 0 else float("nan"),
        "median_delay_s": float(np.median([s.delay_s for s in det])) if det else float("nan"),
    }
