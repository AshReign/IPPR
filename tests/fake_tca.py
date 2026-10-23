"""Stand-in for Package 3's run_tca, used only to test the tuning runner.

It is NOT TCA: it follows Package 3's interface (frame dicts in, uncompressed
RLE at working resolution out) but simply thresholds the current frame.
"""

import numpy as np


def _encode_rle(mask):
    pixels = np.asarray(mask, np.uint8).ravel(order="F")
    counts, prev, run = [], 0, 0
    for v in pixels:
        if v != prev:
            counts.append(run)
            run, prev = 0, int(v)
        run += 1
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def run_tca(frames, params=None):
    out = []
    for f in frames:
        if f["annotated"]:
            for d in f["detections"]:
                out.append({"image_id": f["image_id"], "category_id": 1,
                            "segmentation": _encode_rle(d > 0.5), "score": 1.0})
    return out
