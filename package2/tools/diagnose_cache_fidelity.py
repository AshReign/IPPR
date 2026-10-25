"""CPU-only analysis of saved cache failures, without changing the production codec.

Compare known-box-support normalization and logit-domain reconstruction to the
baseline. This consumes preserved failed arrays/SQLite samples; it does not run
the model, alter gates, or substitute diagnostic variants for an approved codec.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import ops

from p2.cache_io import VideoCacheReader
from p2.soft_masks import restore_work_masks


def resize(value, shape):
    return F.interpolate(value[None], shape, mode="bilinear", align_corners=False)[0]


def iou(lhs, rhs):
    union = (lhs | rhs).sum().item()
    return (lhs & rhs).sum().item() / union if union else 1.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summary = json.loads((args.validation / "summary.json").read_text())
    records = []
    for failed in summary["failures"]:
        if not isinstance(failed["cache_iou"], (float, int)) or failed["cache_iou"] >= .98:
            continue
        stem = f"{failed['video_id']}_{int(failed['frame_idx']):06d}"
        failure_path = args.validation / "failures" / f"{stem}_{failed['instance_idx']}.npz"
        with np.load(failure_path, allow_pickle=False) as data:
            native = torch.from_numpy(data["soft_probabilities"].copy())[None] > .5
            shape = native.shape[-2:]
        reader = VideoCacheReader(args.validation / "roundtrip_samples" / (stem+".sqlite"))
        try:
            frame = reader.read(0)
        finally:
            reader.close()
        candidate = failed["candidate_index"]
        work = torch.from_numpy(frame["soft_mask"][candidate:candidate+1]).float()
        boxes = torch.from_numpy(frame["boxes"][candidate:candidate+1])
        original_h, original_w = shape
        work_h, work_w = work.shape[-2:]
        working_boxes = boxes * torch.tensor([work_w/original_w, work_h/original_h]*2)
        support = ops.crop_mask(torch.ones_like(work), working_boxes)
        support_up = resize(support, shape)
        baseline = restore_work_masks(work, shape, boxes)
        normalized = ops.crop_mask(torch.where(support_up > 0,
            resize(work, shape)/support_up.clamp_min(1e-12), 0), boxes)
        work_logits = torch.logit(work.clamp(1e-6, 1-1e-6))*support
        restored_logits = resize(work_logits, shape)/support_up.clamp_min(1e-12)
        logit_normalized = ops.crop_mask(torch.where(support_up > 0, restored_logits.sigmoid(), 0), boxes)
        xx = torch.arange(original_w)[None, None, :]
        yy = torch.arange(original_h)[None, :, None]
        boundary_width = int(np.ceil(max(original_h/work_h, original_w/work_w)))+1
        x1, y1, x2, y2 = boxes[0].ceil().int()
        edge = ((xx-x1 < boundary_width) | (x2-1-xx < boundary_width) |
                (yy-y1 < boundary_width) | (y2-1-yy < boundary_width))
        differences = (baseline > .5) ^ native
        support_original = ops.crop_mask(torch.ones_like(baseline), boxes).bool()
        records.append({**failed, "baseline_recomputed_iou": iou(baseline > .5, native),
            "probability_support_normalized_iou": iou(normalized > .5, native),
            "logit_support_normalized_iou": iou(logit_normalized > .5, native),
            "native_area_pixels": int(native.sum()), "working_shape": [work_h, work_w],
            "baseline_disagreement_pixels": int(differences.sum()),
            "baseline_disagreement_near_box_edge_pixels": int((differences & edge).sum()),
            "boundary_band_width_original_pixels": boundary_width,
            "original_box_pixels_with_zero_reconstruction_support": int((support_original & (support_up == 0)).sum())})
    aggregate = {"diagnostic_only": True, "input_report": str(args.validation),
        "checkpoint_sha256": summary["checkpoint_sha256"], "baseline_failures": len(records),
        "probability_normalized_failures_remaining": sum(row["probability_support_normalized_iou"] < .98 for row in records),
        "logit_normalized_failures_remaining": sum(row["logit_support_normalized_iou"] < .98 for row in records),
        "rows": records,
        "note": "Only previously failed cases analyzed. Any proposed codec change requires full original sample rerun; these variants are not acceptance results."}
    args.output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in aggregate.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
