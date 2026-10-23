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


def encode_result(mask_work: np.ndarray, image_id: int,
                  orig_size: tuple[int, int]) -> dict:
    """One final binary mask -> COCO result at the original resolution.

    Uses P5's ``encode_mask`` (binarise, bilinear resize, binarise at 0.5), the one
    resize rule shared by T0, T1 and T2 (agreed 2026-10-24).
    """
    h, w = orig_size
    return {
        "image_id": int(image_id),
        "category_id": 1,          # RipVIS has a single class
        "segmentation": encode_mask(mask_work, h, w),
        "score": 1.0,              # every condition outputs thresholded masks
    }


def masks_to_results(masks_work: list[np.ndarray], cache: VideoCache,
                     image_id: int) -> list[dict]:
    """Encode the working-resolution masks of one annotated frame, in list order
    (P5 matches predictions in the order they are listed)."""
    return [encode_result(m, image_id, cache.orig_size) for m in masks_work]
