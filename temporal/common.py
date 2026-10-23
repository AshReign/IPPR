"""Helpers shared by every condition: the output-area filter and result encoding.

All three conditions (T0, T1, T2) must hand the evaluator the same format:
a COCO results list with RLE masks at the ORIGINAL resolution and score 1.0
(A2 Technical Task Packages, "Shared conventions").
"""

from __future__ import annotations

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from .cache_io import VideoCache

# Package 4, step 3: components smaller than 0.1 % of the frame are dropped.
# Fixed before tuning and applied identically to T0, T1 and T2.
DEFAULT_MIN_AREA_FRAC = 0.001


def passes_area_filter(mask: np.ndarray, min_area_frac: float) -> bool:
    """True if a binary mask covers at least ``min_area_frac`` of its frame.

    The fraction is resolution-independent, so it can be applied at the
    working resolution before upsampling.
    """
    return mask.sum() >= min_area_frac * mask.size


def upsample_binary(mask: np.ndarray, orig_size: tuple[int, int]) -> np.ndarray:
    """Upsample a working-resolution binary mask to the original frame.

    Bilinear interpolation of the 0/1 mask followed by a 0.5 cut gives smooth
    boundaries instead of the blocky edges of nearest-neighbour resizing.
    """
    h, w = orig_size
    up = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return up >= 0.5


def encode_result(mask_orig: np.ndarray, image_id: int) -> dict:
    """Encode one original-resolution binary mask as a COCO result entry."""
    rle = mask_utils.encode(np.asfortranarray(mask_orig.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")   # JSON-serialisable
    return {
        "image_id": int(image_id),
        "category_id": 1,          # RipVIS has a single class
        "segmentation": rle,
        "score": 1.0,              # every condition outputs thresholded masks
    }


def masks_to_results(masks_work: list[np.ndarray], cache: VideoCache,
                     image_id: int) -> list[dict]:
    """Upsample and encode the working-resolution masks of one annotated frame."""
    return [encode_result(upsample_binary(m, cache.orig_size), image_id)
            for m in masks_work]
