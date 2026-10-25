"""P2 T0 output under the confirmed Ngoc/P4 working-resolution protocol.

One model instance remains one prediction, including disconnected regions. Work
fields must cover the complete, unpadded working frame and contain probabilities
(not logits, crops, or binary masks reconstructed from original-resolution RLEs).
Selection uses float32-inclusive confidence and work pixels with probability >.5.
The 0.1% cutoff is inclusive and evaluated exactly as 1000 * area >= H * W.

Retained instances are ordered by descending float32 confidence, with their
original candidate_index resolving exact ties. As in P4's COCO adapter, score is
then 1.0; instance_conf preserves the ordering score for audit. P5 must receive
this list unchanged and use min_score=0, without a second confidence filter.

from_encoded is an optimization only when provenance proves each RLE was made
by the same fixed P5 encode_mask from the very work field whose area is supplied.
It cannot derive a working area from an original-resolution RLE. Callers verify
checkpoint, candidate identity, work SHA, transform, and encoder source hashes.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
import numpy as np


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _shape(shape, name: str = "work_hw") -> tuple[int, int]:
    if not isinstance(shape, (Sequence, np.ndarray)) or len(shape) != 2:
        raise ValueError(f"{name} must be (height, width)")
    return tuple(_integer(v, name, 1) for v in shape)


def _probability(value, name: str) -> np.float32:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a numeric probability")
    try:
        original = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric probability") from exc
    if not np.isfinite(original) or not 0 <= original <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return np.float32(original)


def _work_mask(mask) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2 or min(value.shape) <= 0:
        raise ValueError("Each work mask must be a nonempty H x W frame")
    if value.dtype.kind != "f":
        raise ValueError("Work mask must be floating-point probabilities")
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
        raise ValueError("Work mask probabilities must be finite and in [0, 1]")
    return value


def work_binary_area(mask) -> int:
    """Count strict >.5 work pixels, without resize or connected-component split."""
    return int(np.count_nonzero(_work_mask(mask) > 0.5))


def area_keep(area: int, work_hw) -> bool:
    """Inclusive 0.1% of the complete working frame; retain equality exactly."""
    height, width = _shape(work_hw)
    pixels = height * width
    count = _integer(area, "work_binary_area")
    if count > pixels:
        raise ValueError("Work area exceeds the full working frame")
    return 1000 * count >= pixels


def _selection(candidates, work_areas, work_hw, threshold):
    if len(candidates) != len(work_areas):
        raise ValueError("Candidate and working-area counts differ")
    _shape(work_hw)
    cutoff = _probability(threshold, "threshold")
    if cutoff <= 0:
        raise ValueError("Confidence threshold must be positive")
    selected = []
    indices = set()
    image_id = None
    for position, (candidate, area) in enumerate(zip(candidates, work_areas)):
        if not isinstance(candidate, Mapping):
            raise ValueError("Each candidate must be a mapping")
        idx = _integer(candidate.get("candidate_index"), "candidate_index")
        if idx in indices:
            raise ValueError("Repeated candidate_index in one frame")
        indices.add(idx)
        img = _integer(candidate.get("image_id"), "image_id")
        if image_id is not None and img != image_id:
            raise ValueError("A frame adapter cannot mix image IDs")
        image_id = img
        cls = _integer(candidate.get("class_id"), "class_id")
        if cls != 0:
            raise ValueError("Expected single-class P2 class_id=0")
        conf = _probability(candidate.get("conf"), "conf")
        kept_area = area_keep(area, work_hw)
        if conf >= cutoff and kept_area:
            selected.append((position, img, idx, conf))
    selected.sort(key=lambda row: (-float(row[3]), row[2]))
    return selected


def _json_rle(rle) -> dict:
    if not isinstance(rle, Mapping):
        raise ValueError("Expected compressed COCO RLE mapping")
    shape = _shape(rle.get("size"), "RLE size")
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if not isinstance(counts, str):
        raise ValueError("Expected compressed COCO RLE counts string")
    counts.encode("ascii")
    return {"size": list(shape), "counts": counts}


def _prediction(image_id, candidate_index, conf, rle) -> dict:
    return {"image_id": image_id, "category_id": 1, "score": 1.0,
            "segmentation": _json_rle(rle), "candidate_index": candidate_index,
            "instance_conf": float(conf)}


def from_encoded(candidates, rles, work_areas, work_hw, threshold) -> list[dict]:
    """Filter/order previously paired P5 RLEs using independently verified work areas.

    All arguments are parallel sequences for one frame. candidates need image_id,
    candidate_index, class_id=0, and conf. No input mapping or RLE is mutated.
    A same-sized original RLE for every candidate is required, including rejected
    candidates; correctness of the compressed payload is a caller validation gate.
    """
    if len(candidates) != len(rles):
        raise ValueError("Candidate and RLE counts differ")
    encoded = [_json_rle(rle) for rle in rles]
    if len({tuple(rle["size"]) for rle in encoded}) > 1:
        raise ValueError("A frame adapter cannot mix original RLE dimensions")
    selected = _selection(candidates, work_areas, work_hw, threshold)
    return [_prediction(img, idx, conf, encoded[pos]) for pos, img, idx, conf in selected]


def from_work(candidates, masks, original_hw, threshold,
              encode_mask: Callable) -> list[dict]:
    """Apply T0 directly to true work probability fields using fixed P5 encode_mask.

    Area filtering occurs before resizing. Surviving complete instances are
    passed to encode_mask(mask, original_height, original_width); no inverse
    restoration, additional cropping, or component processing is performed.
    masks may be an N x H x W array or a sequence of same-shape H x W arrays.
    """
    height, width = _shape(original_hw, "original_hw")
    if len(candidates) != len(masks):
        raise ValueError("Candidate and working-mask counts differ")
    if not candidates:
        cutoff = _probability(threshold, "threshold")
        if cutoff <= 0:
            raise ValueError("Confidence threshold must be positive")
        return []
    work = [_work_mask(mask) for mask in masks]
    work_hw = work[0].shape
    if any(mask.shape != work_hw for mask in work):
        raise ValueError("Work masks must share one complete frame shape")
    areas = [int(np.count_nonzero(mask > 0.5)) for mask in work]
    selected = _selection(candidates, areas, work_hw, threshold)
    outputs = []
    for pos, img, idx, conf in selected:
        rle = _json_rle(encode_mask(work[pos], height, width))
        if rle["size"] != [height, width]:
            raise ValueError("Encoder output differs from original frame geometry")
        outputs.append(_prediction(img, idx, conf, rle))
    return outputs
