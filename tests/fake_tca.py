"""Stand-in for Package 3's run_tca, used only to test the tuning runner.

It is NOT TCA: it simply thresholds the current frame (like T0 at conf 0) so
the T2 path of the runner can be exercised before the real module exists.
"""

from temporal.t0 import run_t0


def run_tca(video_cache, params):
    return run_t0(video_cache, {"conf_thr": 0.0,
                                "min_area_frac": params.get("min_area_frac", 0.001)})
