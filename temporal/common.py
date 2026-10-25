"""Helpers shared by every condition: the output-area filter and result encoding.

All three conditions (T0, T1, T2) must hand the evaluator the same format:
a COCO results list with RLE masks at the ORIGINAL resolution and score 1.0
(A2 Technical Task Packages, "Shared conventions").
"""

from __future__ import annotations

import cv2
import numpy as np

from eval.evaluator import encode_mask

from .cache_io import VideoCache

# Shared output filter (Ash, 2026-10-24), identical for T0, T1 and T2:
#   1. binarise each output instance (soft mask > 0.5 for T0/TCA, mean > theta for T1),
#   2. upsample it to the original frame resolution,
#   3. drop it if its pixel area is below 0.001 x width x height,
#   4. score the instances that remain.
DEFAULT_MIN_AREA_FRAC = 0.001


def passes_area_filter(mask: np.ndarray, min_area_frac: float) -> bool:
    """True unless the mask's area is below ``min_area_frac`` of its frame.

    Apply it to the ORIGINAL-resolution mask (step 3 above).
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


def encode_result(mask_work: np.ndarray, image_id: int, orig_size: tuple[int, int],
                  min_area_frac: float = DEFAULT_MIN_AREA_FRAC) -> dict | None:
    """One binary instance mask -> COCO result at the original resolution, or None
    if it fails the area filter.

    Upsampling follows P5's ``encode_mask`` rule (bilinear, cut at 0.5); the mask is
    upsampled first so the area filter sees original-resolution pixels.
    """
    h, w = orig_size
    mask_orig = upsample_binary(mask_work, orig_size)
    if not passes_area_filter(mask_orig, min_area_frac):
        return None
    return {
        "image_id": int(image_id),
        "category_id": 1,          # RipVIS has a single class
        "segmentation": encode_mask(mask_orig, h, w),   # already full size: no resize
        "score": 1.0,              # every condition outputs thresholded masks
    }


def masks_to_results(masks_work: list[np.ndarray], cache: VideoCache, image_id: int,
                     min_area_frac: float = DEFAULT_MIN_AREA_FRAC) -> list[dict]:
    """Upsample, filter and encode the masks of one annotated frame, in list order
    (P5 matches predictions in the order they are listed)."""
    out = (encode_result(m, image_id, cache.orig_size, min_area_frac) for m in masks_work)
    return [r for r in out if r is not None]
