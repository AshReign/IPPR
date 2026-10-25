"""CPU-only attribution of fp16, cropping and resampling error on preserved failures.

No production codec changes. Reuses the same checkpoint/images/instance identities.
Diagnostic variants are neither final outputs nor acceptance results.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import ops

from p2.cache_io import VideoCacheReader
from p2.soft_masks import PREDICT_DEFAULTS, SoftMaskPredictor, restore_work_masks, sign_preserving_sigmoid


def resize(value, shape):
    return F.interpolate(value[None], shape, mode="bilinear", align_corners=False)[0]


def iou(lhs, rhs):
    union = (lhs | rhs).sum().item()
    return (lhs & rhs).sum().item()/union if union else 1.


class DiagnosticPredictor(SoftMaskPredictor):
    def construct_result(self, pred, img, orig_img, img_path, proto):
        result = super().construct_result(pred, img, orig_img, img_path, proto)
        channels, ph, pw = proto.shape
        logits = (pred[:, 6:].float() @ proto.float().view(channels, -1)).view(-1, ph, pw)
        top, left, bottom, right = result.p2_transform["prototype_crop_tlbr"]
        work_logits = resize(logits[:, top:bottom, left:right], result.p2_transform["work_shape_hw"])
        raw_probabilities = sign_preserving_sigmoid(work_logits)
        oh, ow = orig_img.shape[:2]
        wh, ww = raw_probabilities.shape[-2:]
        boxes = result.p2_cache_candidates.boxes.xyxy * pred.new_tensor([ww/ow, wh/oh]*2)
        result.diag_float_work = ops.crop_mask(raw_probabilities, boxes)
        result.diag_uncropped_work = raw_probabilities
        result.diag_work_logits = work_logits
        result.diag_work_support = ops.crop_mask(torch.ones_like(raw_probabilities), boxes)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    summary = json.loads((args.validation / "summary.json").read_text())
    checkpoint = Path(summary["checkpoint"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == summary["checkpoint_sha256"]
    grouped = defaultdict(list)
    for failure in summary["failures"]:
        if isinstance(failure["cache_iou"], (float, int)) and failure["cache_iou"] < .98:
            grouped[failure["image_rel"]].append(failure)
    model = YOLO(str(checkpoint))
    records = []
    for image_rel, failed_rows in grouped.items():
        image = cv2.imread(str(args.dataset_root/image_rel))
        result = model.predict(image, predictor=DiagnosticPredictor, device=torch.device("cpu"), **PREDICT_DEFAULTS)[0]
        sample = failed_rows[0]
        stem = f"{sample['video_id']}_{int(sample['frame_idx']):06d}"
        reader = VideoCacheReader(args.validation/"roundtrip_samples"/(stem+".sqlite"))
        try:
            cached = reader.read(0)
        finally:
            reader.close()
        if not np.array_equal(result.p2_cache_candidates.boxes.data[:, :4].cpu().numpy(), cached["boxes"]):
            raise RuntimeError("Candidate boxes/order changed on CPU replay; cannot attribute codec error.")
        if not np.array_equal(result.p2_cache_candidates.boxes.conf.cpu().numpy(), cached["conf"]):
            raise RuntimeError("Candidate confidences changed on CPU replay.")
        for failure in failed_rows:
            ci = failure["candidate_index"]
            native_idx = failure["instance_idx"]
            assert int(result.candidate_index[native_idx]) == ci
            native = result.soft_masks[native_idx:native_idx+1] > .5
            failure_path = args.validation/"failures"/f"{stem}_{native_idx}.npz"
            with np.load(failure_path, allow_pickle=False) as data:
                if not torch.equal(native[0], torch.from_numpy(data["soft_probabilities"]) > .5):
                    raise RuntimeError("Native mask changed on CPU replay.")
            box = result.p2_cache_candidates.boxes.xyxy[ci:ci+1]
            shape = native.shape[-2:]
            support_up = resize(result.diag_work_support[ci:ci+1], shape)
            float_work = result.diag_float_work[ci:ci+1]
            quantized_work = torch.from_numpy(cached["soft_mask"][ci:ci+1]).float()
            float_cropped = restore_work_masks(float_work, shape, box)
            quantized_cropped = restore_work_masks(quantized_work, shape, box)
            normalized_float = ops.crop_mask(torch.where(support_up>0,
                resize(float_work, shape)/support_up.clamp_min(1e-12), 0), box)
            normalized_half = ops.crop_mask(torch.where(support_up>0,
                resize(quantized_work, shape)/support_up.clamp_min(1e-12), 0), box)
            no_work_crop = ops.crop_mask(resize(result.diag_uncropped_work[ci:ci+1], shape), box)
            composed_logits = ops.crop_mask(sign_preserving_sigmoid(resize(result.diag_work_logits[ci:ci+1], shape)), box)
            records.append({**failure,
                "float32_cropped_iou": iou(float_cropped>.5, native),
                "float16_cropped_iou": iou(quantized_cropped>.5, native),
                "float32_support_normalized_iou": iou(normalized_float>.5, native),
                "float16_support_normalized_iou": iou(normalized_half>.5, native),
                "float32_uncropped_work_probabilities_iou": iou(no_work_crop>.5, native),
                "float32_composed_logits_iou": iou(composed_logits>.5, native),
                "fp16_changed_original_binary_pixels": int(((float_cropped>.5)^(quantized_cropped>.5)).sum()),
                "native_area_pixels": int(native.sum()),
            })
        print(json.dumps({"frames_completed": len({row['image_rel'] for row in records}),
                          "frames_total": len(grouped), "instances_analyzed": len(records)}), flush=True)
        del result
    metric_names = [key for key in records[0] if key.endswith("_iou") and key.startswith("float")] if records else []
    aggregate = {"diagnostic_only": True, "checkpoint_sha256": summary["checkpoint_sha256"],
        "same_cpu_replay_candidates_and_native_masks_verified": True,
        "frames_analyzed": len(grouped), "instances_analyzed": len(records),
        "variant_summary": {key: {"min_iou": min(row[key] for row in records),
                                  "failures_below_0_98": sum(row[key]<.98 for row in records)} for key in metric_names},
        "rows": records,
        "note": "Only baseline failures analyzed. Production unchanged; any fix requires the complete original sample and GPU validation."}
    args.output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in aggregate.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
