"""Prepare/run direct-vs-reused candidate and native-RLE equality checks.

Default device is CUDA. This tool never trains or scores ground truth. CPU mode
is an implementation check only and can never claim the CUDA requirement passed.
With --checkpoint and --source-inventory, real annotated dev images are forwarded
once each; all 17 NMS threshold checks reuse that raw prediction tensor.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

P2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P2_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_dev_predictions import (PROOF_SCHEMA, REAL_SAMPLE_SEED, atomic_json, native_original_rles,
    real_sample_ordinals, runtime_identity, sha256, verify_source_inventory)


def compare_raw(raw, name):
    import torch
    from p2.soft_masks import confidence_keep, inclusive_nms
    original = raw.clone()
    base = inclusive_nms(raw, nc=1, threshold=.1)[0]
    rows = []
    for number in range(10, 91, 5):
        threshold = number / 100
        direct = inclusive_nms(raw, nc=1, threshold=threshold)[0]
        reused = base[confidence_keep(base[:, 4], threshold)]
        equal = torch.equal(reused, direct)
        rows.append({"threshold": threshold, "direct_count": len(direct), "reused_count": len(reused),
                     "exact_box_conf_class_coefficients_and_order_equal": equal,
                     "direct_candidate_ids_on_failure": direct[:, 6].cpu().tolist() if not equal else None,
                     "reused_candidate_ids_on_failure": reused[:, 6].cpu().tolist() if not equal else None})
    unchanged = torch.equal(original, raw)
    return {"name": name, "raw_candidates": raw.shape[-1], "retained_at_point1": len(base),
            "input_unchanged": unchanged, "pass": unchanged and all(x["exact_box_conf_class_coefficients_and_order_equal"] for x in rows),
            "threshold_results": rows}


def synthetic_cases(device):
    import torch
    generator = torch.Generator().manual_seed(4264)
    raw = torch.rand(1, 7, 207, generator=generator)
    raw[:, :2] *= 640
    raw[:, 2:4] = raw[:, 2:4] * 150 + 1
    raw[0, 5] = torch.arange(raw.shape[-1])
    for index, number in enumerate(range(10, 91, 5)):
        threshold = np.float32(number / 100)
        raw[0, 4, index*3:index*3+3] = torch.tensor([
            np.nextafter(threshold, np.float32(-np.inf)), threshold,
            np.nextafter(threshold, np.float32(np.inf))])
    raw[0, :4, 61] = raw[0, :4, 60]
    raw[0, 4, 61] = raw[0, 4, 60]
    yield "boundaries_adjacent_floats_overlap_and_tie", raw.to(device)
    # 401 non-overlapping boxes exercise the max_det=300 cap and tied rankings.
    capped = torch.zeros(1, 7, 401)
    index = torch.arange(401)
    capped[0, 0] = (index % 21) * 28 + 10
    capped[0, 1] = torch.div(index, 21, rounding_mode="floor") * 28 + 10
    capped[0, 2:4] = 10
    capped[0, 4] = .55
    capped[0, 4, :70] = .9
    capped[0, 4, 370:] = .1
    capped[0, 5] = index
    yield "401_nonoverlap_tied_scores_max_det_300", capped.to(device)
    # A separate >300-candidate case avoids ties, isolating the cap invariant.
    distinct = capped.clone()
    distinct[0, 4] = torch.linspace(.1, .95, 401)
    yield "401_nonoverlap_unique_scores_max_det_300", distinct.to(device)


def compare_native_rles(prototypes, coefficients, boxes, shape, name):
    import torch
    from pycocotools import mask as mask_utils
    from ultralytics.utils import ops
    standard = ops.process_mask_native(prototypes, coefficients, boxes, shape).bool().cpu()
    encoded = native_original_rles(prototypes, coefficients, boxes, shape)
    rows = []
    for index, record in enumerate(encoded):
        decoded = torch.from_numpy(mask_utils.decode(record["rle"]).astype(np.bool_))
        rows.append({"candidate_index": index, "area": record["mask_area"],
                     "native_pixels_equal": bool(torch.equal(decoded, standard[index]))})
    return {"name": name, "shape": list(shape), "instances": len(rows), "pass": all(r["native_pixels_equal"] for r in rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-inventory", type=Path, default=P2_ROOT / "reports/dev_selection_sources.json")
    parser.add_argument("--real-limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh report path to retain previous equivalence evidence")
    if args.real_limit < 1:
        raise ValueError("real-limit must be positive")
    import torch
    import ultralytics
    from ultralytics.utils import nms, ops
    import p2.soft_masks as soft_masks
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not silently substitute CPU")
    native_cases = []
    real_cases = []
    inventory = None
    indices = []
    actual_runtime = None
    if args.checkpoint:
        import cv2
        cv2.setNumThreads(1)
        from ultralytics import YOLO
        from ultralytics.engine.results import Results
        from ultralytics.utils import ops
        inventory = json.loads(args.source_inventory.read_text())
        verify_source_inventory(inventory)
        class CaptureRaw(soft_masks.SoftMaskPredictor):
            def postprocess(self, preds, img, orig_imgs):
                self.captured_raw = preds[0].detach().clone()
                self.captured_proto = (preds[1][-1] if isinstance(preds[1], tuple) else preds[1]).detach()
                self.captured_input_hw = img.shape[2:]
                if not isinstance(orig_imgs, list):
                    orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
                return [Results(orig_img, path="dev-equivalence-only", names=self.model.names,
                                boxes=img.new_empty((0, 6))) for orig_img in orig_imgs]
        model = YOLO(str(args.checkpoint))
        # Fixed seed/order chosen before predictions; a larger scope extends it.
        indices = real_sample_ordinals(len(inventory["frames"]), args.real_limit)
        for ordinal in indices:
            source = inventory["frames"][ordinal]
            content = Path(source["image_path"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != source["image_sha256"]:
                raise ValueError("Dev image changed since source inventory")
            image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (source["height"], source["width"]):
                raise ValueError("Real dev image dimensions differ")
            result = model.predict(image, device=str(device), predictor=CaptureRaw, **soft_masks.PREDICT_DEFAULTS)[0]
            raw = model.predictor.captured_raw
            if len(model.names) != 1 or model.predictor.model.fp16:
                raise ValueError("Real proof requires a float32 single-class rip model")
            current_runtime = runtime_identity(raw.device, str(raw.dtype).removeprefix("torch."))
            if actual_runtime is not None and current_runtime != actual_runtime:
                raise ValueError("Runtime flags changed between proof samples")
            actual_runtime = current_runtime
            proof = compare_raw(raw, source["image_rel"])
            proof["source_image_sha256"] = source["image_sha256"]
            proof["source_ordinal"] = ordinal
            real_cases.append(proof)
            pred = soft_masks.inclusive_nms(raw, nc=1, threshold=.1)[0]
            # Original-mask comparison uses a bounded first 8 raw candidates.
            # NMS equivalence above covers ALL candidates, including >300 caps.
            pred = pred[:8].clone()
            if len(pred):
                pred[:, :4] = ops.scale_boxes(model.predictor.captured_input_hw, pred[:, :4], image.shape)
                native_cases.append(compare_native_rles(model.predictor.captured_proto[0], pred[:, 6:], pred[:, :4], image.shape[:2], source["image_rel"]))
            del result, image, raw, pred
    # Run synthetic cases under the same flags set by the real predictor setup.
    cases = [compare_raw(raw, name) for name, raw in synthetic_cases(device)]
    generator = torch.Generator().manual_seed(831)
    for shape in ((79, 173), (173, 79), (128, 128)):
        prototypes = torch.randn(4, 40, 40, generator=generator).to(device)
        coefficients = torch.randn(17, 4, generator=generator).to(device)
        h, w = shape
        boxes = torch.tensor([[0, 0, w, h], [.2, 3.7, w-3.2, h-5.1], [2, 2, 3, 3],
                              [0, 0, 0, 0], [w/3, h/3, w/2, h/2], [5, 5, w, h], [0, 0, w-2, h-1]], device=device).repeat(3, 1)[:17]
        native_cases.append(compare_native_rles(prototypes, coefficients, boxes, shape, "synthetic_full_vs_sliced_native"))
    final_runtime = runtime_identity(device)
    if actual_runtime is not None and final_runtime != actual_runtime:
        raise ValueError("Synthetic and real proof runtime flags differ")
    actual_runtime = actual_runtime or final_runtime
    all_equal = all(x["pass"] for x in cases + native_cases + real_cases)
    real_nonempty = sum(x["retained_at_point1"] for x in real_cases)
    report = {"schema": PROOF_SCHEMA, "kind": "p2_dev_candidate_reuse_equivalence", "device": str(device), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "collector_sha256": sha256(Path(__file__).with_name("collect_dev_predictions.py")),
        "soft_masks_sha256": sha256(soft_masks.__file__), "inference": soft_masks.PREDICT_DEFAULTS, "nms": soft_masks.NMS_FIXED,
        "source_sha256": {"collector": sha256(Path(__file__).with_name("collect_dev_predictions.py")),
                          "verifier": sha256(__file__), "soft_masks": sha256(soft_masks.__file__),
                          "installed_ops": sha256(ops.__file__), "installed_nms": sha256(nms.__file__)},
        "source_inventory_signature": inventory["signature"] if inventory else None,
        "source_inventory_sha256": sha256(args.source_inventory) if inventory else None,
        "real_sampling": {"method": "numpy.default_rng.permutation-prefix", "seed": REAL_SAMPLE_SEED, "ordinals": indices},
        "real_native_candidate_limit": 8, "runtime": actual_runtime,
        "synthetic_nms_cases": cases, "native_rle_cases": native_cases, "real_cases": real_cases,
        "all_equal": all_equal, "real_nonempty_candidates": real_nonempty,
        "cuda_requirement_met": device.type == "cuda" and all_equal and real_nonempty > 0 and len(real_cases) >= 5,
        "scope_note": "Exact backend-specific equality on recorded cases. Real native masks compare first up to 8 candidates per image; all-candidate NMS uses all 17 thresholds. Synthetic native checks cover 17 masks. No metric/area/component rule implemented."}
    report["status"] = "FAIL" if not all_equal else ("CUDA_WITH_REAL_OUTPUT_PASS" if report["cuda_requirement_met"] else "CPU_ONLY" if device.type == "cpu" else "CUDA_SYNTHETIC_ONLY")
    atomic_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("status", "device", "all_equal", "real_nonempty_candidates", "cuda_requirement_met")}, indent=2))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
