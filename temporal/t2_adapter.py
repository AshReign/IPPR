"""T2: calls Package 3's ``run_tca`` and puts its output in the shared format.

Package 3's interface (branch ``package3-tca``, temporal/tca.py)::

    run_tca(frames, params) -> list of COCO-like results
    frames: iterable of {"t_sec", "detections", "annotated", "image_id"},
            one per processed frame, with every candidate's soft mask
    results: uncompressed RLE at the working resolution, score 1.0

The adapter feeds it the same candidate pool T0 and T1 see, then applies the
shared 0.1 % area filter and upsamples to the original resolution, so all three
conditions reach the evaluator in exactly the same form.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
from pycocotools import mask as mask_utils

from .cache_io import VideoCache
from .common import DEFAULT_MIN_AREA_FRAC, encode_result, passes_area_filter, upsample_binary

# Keys owned by Package 4; everything else is passed to run_tca unchanged.
_P4_KEYS = {"min_area_frac"}


def to_p3_frames(cache: VideoCache) -> Iterator[dict]:
    """Stream the cache in Package 3's frame format (lazy, one frame at a time)."""
    for frame in cache.frames:
        yield {"t_sec": float(frame.timestamp),
               "detections": [inst.soft_mask.astype(np.float32) for inst in frame.instances],
               "annotated": frame.annotated,
               "image_id": frame.image_id}


def _decode(rle: dict) -> np.ndarray:
    """Decode an RLE in either compressed or uncompressed (list counts) form."""
    h, w = rle["size"]
    if isinstance(rle["counts"], list):
        rle = mask_utils.frPyObjects(rle, h, w)
    return mask_utils.decode(rle).astype(bool)


def run_t2(cache: VideoCache, params: dict, run_tca: Callable) -> list[dict]:
    """Run TCA on one video and return results in the shared format."""
    min_area = float(params.get("min_area_frac", DEFAULT_MIN_AREA_FRAC))
    tca_params = {k: v for k, v in params.items() if k not in _P4_KEYS}
    results = []
    for res in run_tca(to_p3_frames(cache), tca_params):
        mask = _decode(res["segmentation"])
        if passes_area_filter(mask, min_area):
            results.append(encode_result(upsample_binary(mask, cache.orig_size),
                                         res["image_id"]))
    return results
