"""Frame-level scorer (Package 5): the official RipVIS matching rule, plus per-frame counts.

Matching copies `compute_pr_f1_f2.py` from RipVIS v1.8.4 exactly:
- Within a frame, predictions are taken in file order. The score is only used to drop
  predictions below `min_score`; it does not change the order.
- Each prediction picks the ground-truth mask with the highest IoU (strictly greater, so on
  a tie the first ground-truth mask in the COCO file wins). It is a TP if that IoU >= 0.5
  and that mask is still unmatched; otherwise it is an FP. It never falls back to its
  second-best mask.
- Every unmatched ground-truth mask is an FN. Each COCO annotation is one ground-truth
  instance, so a rip drawn as several annotations counts as several instances.
- P, R and F2 pool TP/FP/FN over all frames. With no predictions P = 1, with no ground
  truth R = 1 (sklearn's zero_division=1).

What we add on top: TP/FP/FN per frame (for the bootstrap and the temporal metrics), the
false-alarm rate, a reusable ground-truth object so tuning sweeps load the COCO file once,
and `encode_mask`, the one shared way to bring a working-resolution mask to the original
resolution for T0, T1 and T2.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data" / "manifests"
GT_FILES = {"dev": "train/coco_annotations/train.json", "val": "val/coco_annotations/val.json"}
IOU_THR = 0.5


@dataclass
class GroundTruth:
    frames: pd.DataFrame                    # one row per annotated frame, in COCO file order
    shapes: Dict[int, Tuple[int, int]]      # image_id -> (height, width)
    masks: Dict[int, List[dict]]            # image_id -> compressed RLEs, in COCO file order
    coco: Optional[dict] = None             # the subset COCO dict, for the official script


def load_gt(split: str, ripvis_root: Optional[str] = None,
            videos: Optional[Iterable[str]] = None) -> GroundTruth:
    """Ground truth for the annotated frames of one split ('dev' or 'val'),
    optionally restricted to some of its videos."""
    if ripvis_root is None:
        from configs import paths
        ripvis_root = paths()["ripvis_root"]
    coco = json.loads((Path(ripvis_root) / GT_FILES[split]).read_text())

    index = pd.read_csv(MANIFESTS / "frames_index.csv")
    has_rip = pd.read_csv(MANIFESTS / "videos.csv")[["video_id", "has_rip"]]
    index = index[index.split == split].merge(has_rip, on="video_id", how="left")
    if videos is not None:
        index = index[index.video_id.isin(set(videos))]
    keep = set(index.image_id)

    images = [im for im in coco["images"] if im["id"] in keep]
    anns = [a for a in coco["annotations"] if a["image_id"] in keep]
    shapes = {im["id"]: (im["height"], im["width"]) for im in images}
    masks: Dict[int, List[dict]] = {i: [] for i in shapes}
    for a in anns:
        h, w = shapes[a["image_id"]]
        masks[a["image_id"]].append(_to_rle(a["segmentation"], h, w))

    order = {im["id"]: k for k, im in enumerate(images)}
    frames = index.assign(_order=index.image_id.map(order)).sort_values("_order")
    frames = frames.drop(columns="_order").reset_index(drop=True)
    subset = {"images": images, "annotations": anns, "categories": coco["categories"]}
    return GroundTruth(frames, shapes, masks, subset)


def write_gt_subset(gt: GroundTruth, path) -> None:
    """Write the split's ground truth as its own COCO file, for running the official script.
    The official script scores every image in the file, so the full train.json would turn
    all non-dev frames into FNs."""
    Path(path).write_text(json.dumps(gt.coco))


def encode_mask(mask: np.ndarray, height: int, width: int) -> dict:
    """Final binary mask (any resolution) -> compressed RLE at the original resolution.

    The one resize rule shared by T0, T1 and T2: binarise at 0.5, resize with bilinear
    interpolation, binarise again at 0.5. Returns JSON-ready RLE (counts as str)."""
    m = (np.asarray(mask) > 0.5).astype(np.float32)
    if m.shape != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
    rle = mask_utils.encode(np.asfortranarray((m >= 0.5).astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def evaluate(predictions: Iterable[dict], gt: GroundTruth, min_score: float = 0.0,
             iou_thr: float = IOU_THR) -> pd.DataFrame:
    """Score COCO-results predictions. Returns one row per annotated frame with
    n_gt, n_pred, tp, fp, fn (plus video_id, frame_idx, t_sec, image_id, has_rip)."""
    by_image: Dict[int, List[dict]] = {}
    for p in predictions:
        if p.get("score", 1.0) < min_score:
            continue
        img = int(p["image_id"])
        if img not in gt.shapes:
            raise ValueError(f"Prediction for image_id {img}, which is not in this split")
        h, w = gt.shapes[img]
        rle = _to_rle(p["segmentation"], h, w)
        if tuple(rle["size"]) != (h, w):
            raise ValueError(f"image_id {img}: mask size {rle['size']} != original {[h, w]}; "
                             "use encode_mask() to resize first")
        by_image.setdefault(img, []).append(rle)

    rows = []
    for img in gt.frames.image_id:
        preds = by_image.get(img, [])
        tp, fp, fn = match_frame(preds, gt.masks[img], iou_thr)
        rows.append((len(gt.masks[img]), len(preds), tp, fp, fn))
    counts = pd.DataFrame(rows, columns=["n_gt", "n_pred", "tp", "fp", "fn"])
    meta = gt.frames[["video_id", "frame_idx", "t_sec", "image_id", "has_rip"]]
    return pd.concat([meta, counts], axis=1)


def match_frame(preds: List[dict], gts: List[dict], iou_thr: float = IOU_THR) -> Tuple[int, int, int]:
    """TP, FP, FN for one frame under the official rule (see module docstring)."""
    if not preds:
        return 0, 0, len(gts)
    if not gts:
        return 0, len(preds), 0
    iou = mask_utils.iou(preds, gts, [0] * len(gts))       # (n_pred, n_gt)
    matched = set()
    tp = fp = 0
    for row in iou:
        best = int(np.argmax(row))                         # first index wins a tie
        if row[best] > 0 and row[best] >= iou_thr and best not in matched:
            tp += 1
            matched.add(best)
        else:
            fp += 1
    return tp, fp, len(gts) - len(matched)


def scores(tp, fp, fn):
    """P, R, F1, F2 from pooled counts; works on scalars and numpy arrays (the bootstrap).
    As in the official script: no predictions gives P = 1, no ground truth gives R = 1."""
    tp, fp, fn = (np.asarray(x, dtype=np.float64) for x in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 1.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
        f2 = np.where(p + r > 0, 5 * p * r / (4 * p + r), 0.0)
    return p, r, f1, f2


def summarise(per_frame: pd.DataFrame) -> dict:
    """Pooled P, R, F1, F2 and the false-alarm rate.

    False-alarm rate: annotated frames of no-rip videos with at least one prediction, over
    all annotated frames of no-rip videos. In dev and val every frame with no ground truth
    belongs to a no-rip video, so this equals the rate over empty frames."""
    tp, fp, fn = (int(per_frame[c].sum()) for c in ("tp", "fp", "fn"))
    p, r, f1, f2 = (float(x) for x in scores(tp, fp, fn))
    no_rip = per_frame[~per_frame.has_rip.astype(bool)]
    return {
        "precision": p, "recall": r, "f1": f1, "f2": f2,
        "tp": tp, "fp": fp, "fn": fn,
        "false_alarm_rate": float((no_rip.n_pred > 0).mean()) if len(no_rip) else float("nan"),
        "n_frames": len(per_frame), "n_no_rip_frames": len(no_rip),
    }


def video_counts(per_frame: pd.DataFrame) -> pd.DataFrame:
    """TP/FP/FN summed per video, the unit the paired bootstrap resamples."""
    return per_frame.groupby("video_id", sort=False)[["tp", "fp", "fn"]].sum().reset_index()


def temporal_frames(per_frame: pd.DataFrame, video_id: str) -> List[Tuple[float, bool, bool]]:
    """(t_sec, gt_present, matched) per annotated frame, the input of
    eval.temporal_metrics.track_stats. Meant for the single-rip val videos."""
    v = per_frame[per_frame.video_id == video_id]
    return list(zip(v.t_sec.astype(float), v.n_gt > 0, v.tp > 0))


def _to_rle(segm, h: int, w: int) -> dict:
    """Polygon list, uncompressed RLE or compressed RLE -> one compressed RLE."""
    if isinstance(segm, list):                              # polygons: merge into one mask
        return mask_utils.merge(mask_utils.frPyObjects(segm, h, w))
    if isinstance(segm["counts"], list):                    # uncompressed RLE
        return mask_utils.frPyObjects(segm, *segm["size"])
    return segm


def main():
    ap = argparse.ArgumentParser(description="Score COCO-results predictions on dev or val")
    ap.add_argument("--split", choices=["dev", "val"], required=True)
    ap.add_argument("--pred", help="predictions JSON (COCO results list)")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--per-frame", help="optional CSV path for the per-frame counts")
    ap.add_argument("--write-gt", help="write the split's ground-truth subset JSON here")
    args = ap.parse_args()

    gt = load_gt(args.split)
    if args.write_gt:
        write_gt_subset(gt, args.write_gt)
    if args.pred:
        per_frame = evaluate(json.loads(Path(args.pred).read_text()), gt, args.min_score)
        if args.per_frame:
            per_frame.to_csv(args.per_frame, index=False)
        print(json.dumps(summarise(per_frame), indent=2))


if __name__ == "__main__":
    main()
