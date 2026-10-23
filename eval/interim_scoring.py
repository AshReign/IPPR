"""INTERIM evaluator used only for dev tuning until Package 5 delivers theirs.

Package 4 needs P/R/F2 on the dev videos to tune T1 and T2, but the official
evaluator is Package 5's job. This module implements the matching rule as
described in the task packages so tuning can be built and tested now:

  * predictions are processed in order (all scores are 1.0, so input order);
  * each prediction takes its highest-IoU ground-truth mask and is a TP if
    IoU >= 0.5 and that mask is still unmatched;
  * every other prediction is an FP (including all predictions on frames with
    no ground truth); every unmatched ground-truth mask is an FN.

Before the dev search is frozen, replace ``score_video`` with Package 5's
evaluator, or confirm the two give identical TP/FP/FN on the pilot cache.
"""

from __future__ import annotations

import contextlib
import io
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

IOU_THR = 0.5  # RipVIS benchmark threshold

# "RipVIS-143_01416.jpg" -> ("RipVIS-143", 1416); frame numbers are 0-based.
# Some files are not zero-padded (e.g. "RipVIS-035_0005.jpg"), so both numbers
# are parsed as integers and the video id is normalised to three digits, the
# same way as data/build_manifests.py (Package 1).
_NAME_RE = re.compile(r"(RipVIS-(?:NR-)?)(\d+)_(\d+)")


def parse_file_name(file_name: str) -> tuple[str, int]:
    m = _NAME_RE.search(file_name.rsplit("/", 1)[-1])
    if m is None:
        raise ValueError(f"unexpected RipVIS file name: {file_name}")
    return f"{m.group(1)}{int(m.group(2)):03d}", int(m.group(3))


class GroundTruth:
    """Ground-truth masks from a RipVIS COCO file, indexed by video."""

    def __init__(self, coco_json: str):
        with contextlib.redirect_stdout(io.StringIO()):   # silence pycocotools
            self.coco = COCO(coco_json)
        # video_id -> {frame_idx: image_id}
        self.frames_by_video: dict[str, dict[int, int]] = defaultdict(dict)
        for img in self.coco.dataset["images"]:
            vid, fidx = parse_file_name(img["file_name"])
            self.frames_by_video[vid][fidx] = img["id"]

    def rles(self, image_id: int) -> list[dict]:
        """Ground-truth RLEs of one image (empty list for a no-rip frame)."""
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        return [self.coco.annToRLE(a) for a in self.coco.loadAnns(ann_ids)]


def match_image(dt: list[dict], gt: list[dict]) -> tuple[int, int, int]:
    """Apply the matching rule to one image; return (tp, fp, fn)."""
    if not gt:
        return 0, len(dt), 0
    if not dt:
        return 0, 0, len(gt)
    ious = np.asarray(mask_utils.iou(dt, gt, [0] * len(gt)))   # (n_dt, n_gt)
    matched: set[int] = set()
    for i in range(len(dt)):
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= IOU_THR and j not in matched:
            matched.add(j)
    tp = len(matched)
    return tp, len(dt) - tp, len(gt) - tp


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    norip_frames: int = 0            # annotated frames with no ground truth
    norip_frames_with_pred: int = 0  # ... of which have >= 1 prediction

    def __iadd__(self, other: "Counts") -> "Counts":
        for k in vars(self):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        return self


def score_video(results: list[dict], gt: GroundTruth, video_id: str) -> Counts:
    """TP/FP/FN and false-alarm counts over every annotated frame of a video."""
    by_image: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_image[r["image_id"]].append(r["segmentation"])

    c = Counts()
    for image_id in gt.frames_by_video[video_id].values():
        gt_rles, dt = gt.rles(image_id), by_image.get(image_id, [])
        tp, fp, fn = match_image(dt, gt_rles)
        c.tp, c.fp, c.fn = c.tp + tp, c.fp + fp, c.fn + fn
        if not gt_rles:
            c.norip_frames += 1
            c.norip_frames_with_pred += int(bool(dt))
    return c


def metrics(c: Counts) -> dict[str, float]:
    """Precision, recall, F2 and false-alarm rate from pooled counts.

    Undefined ratios (e.g. precision with no predictions) are reported as 0.
    """
    p = c.tp / (c.tp + c.fp) if c.tp + c.fp else 0.0
    r = c.tp / (c.tp + c.fn) if c.tp + c.fn else 0.0
    f2 = 5 * p * r / (4 * p + r) if (4 * p + r) else 0.0
    far = (c.norip_frames_with_pred / c.norip_frames) if c.norip_frames else 0.0
    return {"precision": p, "recall": r, "f2": f2, "false_alarm_rate": far}
