"""T0: raw per-frame predictions, the baseline with no temporal processing.

Package 4, step 1: keep instances with ``conf >= conf_thr`` (the threshold
chosen on dev by Package 2) and binarise each soft mask at 0.5.
"""

from __future__ import annotations

from .cache_io import VideoCache
from .common import DEFAULT_MIN_AREA_FRAC, masks_to_results


def run_t0(cache: VideoCache, params: dict) -> list[dict]:
    """Return COCO results for the annotated frames of one video.

    params:
        conf_thr       instance-confidence threshold (from Package 2)
        min_area_frac  shared output filter (default 0.001 = 0.1 % of frame)
    """
    conf_thr = float(params["conf_thr"])
    min_area = float(params.get("min_area_frac", DEFAULT_MIN_AREA_FRAC))

    results: list[dict] = []
    # T0 only looks at the current frame, so unannotated frames can be skipped.
    for frame in cache.annotated_frames():
        # P5 matches predictions in list order: list T0 by descending confidence.
        kept = sorted((i for i in frame.instances if i.conf >= conf_thr),
                      key=lambda i: -i.conf)
        masks = [inst.soft_mask > 0.5 for inst in kept]
        results.extend(masks_to_results(masks, cache, frame.image_id, min_area))
    return results
