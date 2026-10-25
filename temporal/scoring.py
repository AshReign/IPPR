"""Package 4's thin layer over Package 5's evaluator (eval/evaluator.py).

The matching itself is P5's ``evaluate`` (the official RipVIS rule). This module
only adds what the dev search needs:

* ``gt_from_coco``: build P5's GroundTruth straight from a COCO file (used by the
  synthetic tests; real runs use P5's ``load_gt("dev")``),
* ``gt_subset``: the ground truth of one video, so each worker scores one video,
* ``score``: per-video counts that can be pooled over videos,
* ``metrics``: pooled P, R, F2 and false-alarm rate with P5's conventions
  (P = 1 with no predictions, R = 1 with no ground truth).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from eval.evaluator import GroundTruth, _to_rle, evaluate
from eval.interim_scoring import parse_file_name


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    norip_frames: int = 0            # annotated frames of no-rip videos
    norip_frames_with_pred: int = 0  # ... of which have >= 1 prediction


COUNT_FIELDS = list(vars(Counts()).keys())


def gt_from_coco(coco_json: str | Path) -> GroundTruth:
    """P5's GroundTruth from a COCO file whose file names follow RipVIS naming."""
    coco = json.loads(Path(coco_json).read_text())
    shapes = {im["id"]: (im["height"], im["width"]) for im in coco["images"]}
    masks = {i: [] for i in shapes}
    for a in coco["annotations"]:
        masks[a["image_id"]].append(_to_rle(a["segmentation"], *shapes[a["image_id"]]))
    rows = []
    for im in coco["images"]:
        vid, fidx = parse_file_name(im["file_name"])
        rows.append({"video_id": vid, "frame_idx": fidx, "t_sec": float("nan"),
                     "image_id": im["id"]})
    frames = pd.DataFrame(rows)
    has_rip = {v: any(masks[i] for i in g.image_id) for v, g in frames.groupby("video_id")}
    frames["has_rip"] = frames.video_id.map(has_rip)
    return GroundTruth(frames, shapes, masks)


def gt_subset(gt: GroundTruth, video_id: str) -> GroundTruth:
    """Ground truth restricted to one video (same object type, same order)."""
    frames = gt.frames[gt.frames.video_id == video_id].reset_index(drop=True)
    ids = set(frames.image_id)
    return GroundTruth(frames, {i: s for i, s in gt.shapes.items() if i in ids},
                       {i: m for i, m in gt.masks.items() if i in ids})


def frame_to_image(gt_video: GroundTruth) -> dict[int, int]:
    """{frame_idx: image_id} for one video, from P1's frames index."""
    return dict(zip(gt_video.frames.frame_idx.astype(int), gt_video.frames.image_id.astype(int)))


def score(results: list[dict], gt_video: GroundTruth) -> Counts:
    """Score one video's predictions with P5's evaluator."""
    pf = evaluate(results, gt_video)
    no_rip = ~pf.has_rip.astype(bool)
    return Counts(tp=int(pf.tp.sum()), fp=int(pf.fp.sum()), fn=int(pf.fn.sum()),
                  norip_frames=int(no_rip.sum()),
                  norip_frames_with_pred=int((no_rip & (pf.n_pred > 0)).sum()))


def metrics(c: Counts) -> dict[str, float]:
    """Same formulas and zero-division conventions as P5's ``summarise``."""
    p = c.tp / (c.tp + c.fp) if c.tp + c.fp else 1.0
    r = c.tp / (c.tp + c.fn) if c.tp + c.fn else 1.0
    f2 = 5 * p * r / (4 * p + r) if p + r else 0.0
    far = c.norip_frames_with_pred / c.norip_frames if c.norip_frames else float("nan")
    return {"precision": p, "recall": r, "f2": f2, "false_alarm_rate": far}
