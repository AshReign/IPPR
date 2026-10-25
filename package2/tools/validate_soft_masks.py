"""Real-frame P2 mask gates. Run with frozen dev manifest and an explicit checkpoint.

This creates LOCAL reports and optional local serialization roundtrips. It neither
scores ground-truth performance nor accesses final-val. A temporary pilot model
must be labelled --model-status temporary; final validation uses frozen weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from p2.cache_io import VideoCacheReader, VideoCacheWriter, canonical
import p2.soft_masks as mask_module
from p2.soft_masks import NMS_FIXED, PREDICT_DEFAULTS, SoftMaskPredictor, restore_work_masks, strict_nms_prefilter


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_frames(manifest, count, seed):
    with open(manifest, newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "dev"]
    if count < 200 or len(rows) < count:
        raise ValueError("Mask validation requires at least 200 distinct dev annotated frames.")
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        groups.setdefault(row["video_id"], []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    # Round-robin random samples preserve representation of all 17 dev videos.
    selected = []
    while len(selected) < count:
        for video in sorted(groups):
            if groups[video] and len(selected) < count:
                selected.append(groups[video].pop())
    return selected


def binary_iou(lhs, rhs):
    intersection = (lhs & rhs).sum().item()
    union = (lhs | rhs).sum().item()
    return intersection / union if union else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("upstream/data/manifests/frames_index.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--cache-frames", type=int, default=50)
    parser.add_argument("--standard-spotchecks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Never approve production export, even if both numerical gates pass.")
    parser.add_argument("--model-status", choices=["temporary", "frozen"], required=True)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        raise ValueError("CPU thread count must be positive.")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    # Locked select_device("cpu") resets threads to its global NUM_THREADS.
    # Its supported torch.device input returns directly, preserving this process's
    # explicit small CPU budget without editing library source or shared settings.
    runtime_device = torch.device("cpu") if args.device == "cpu" else args.device
    if args.cache_frames not in (0,) and not 50 <= args.cache_frames <= args.frames:
        raise ValueError("Cache gate requires 50 or more selected annotated dev frames, or 0 to skip.")
    if not 0 <= args.standard_spotchecks <= args.frames:
        raise ValueError("Standard predictor spotchecks must be between zero and selected frame count.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory to preserve earlier validation evidence.")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = select_frames(args.manifest, args.frames, args.seed)
    cache_indices = set(random.Random(args.seed + 1).sample(range(len(selected)), args.cache_frames))
    for row in selected:
        if not (args.dataset_root / row["image_rel"]).is_file():
            raise FileNotFoundError(args.dataset_root / row["image_rel"])
    checkpoint_hash = sha256(args.checkpoint)
    module_hash = sha256(mask_module.__file__)
    prediction_config = {"inference": PREDICT_DEFAULTS, "nms": NMS_FIXED,
                         "soft_mask_module_sha256": module_hash}
    prediction_hash = hashlib.sha256(canonical(prediction_config).encode()).hexdigest()
    (args.output / "sampled_frames.json").write_text(json.dumps({"seed": args.seed,
        "manifest_sha256": sha256(args.manifest), "frames": selected,
        "cache_indices": sorted(cache_indices)}, indent=2), encoding="utf-8")
    model = YOLO(str(args.checkpoint))
    standard_model = YOLO(str(args.checkpoint)) if args.standard_spotchecks else None
    spotcheck_indices = set(random.Random(args.seed+2).sample(range(args.frames), args.standard_spotchecks))
    spotchecks = []
    validation_runtime = None
    failures, inverse_diagnostics = [], []
    stats = {"frames": 0, "nonempty_frames": 0, "instances": 0, "softmask_failed_instances": 0,
             "cache_frames": 0, "cache_nonempty_frames": 0, "cache_instances": 0, "cache_failed_instances": 0,
             "softmask_min_iou": 1., "cache_min_iou": 1., "native_empty_candidates": 0}
    start = time.perf_counter()
    csv_path = args.output / "per_instance.csv"
    fields = ["video_id", "frame_idx", "image_rel", "instance_idx", "candidate_index", "native_empty", "confidence",
              "softmask_iou", "softmask_pass", "cache_iou", "cache_pass"]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=fields)
        writer.writeheader()
        for index, frame in enumerate(selected):
            image = cv2.imread(str(args.dataset_root / frame["image_rel"]))
            if image is None:
                raise RuntimeError(f"Could not decode {frame['image_rel']}")
            result = model.predict(image, predictor=SoftMaskPredictor, device=runtime_device,
                **PREDICT_DEFAULTS)[0]
            if validation_runtime is None:
                actual_device = result.boxes.data.device
                validation_runtime = {"device_type": actual_device.type,
                    "device_name": torch.cuda.get_device_name(actual_device) if actual_device.type == "cuda" else (platform.processor() or platform.machine()),
                    "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version(), "ultralytics_version": ultralytics.__version__,
                    "inference_dtype": "float32", "batch": 1,
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "python_version": sys.version.split()[0], "numpy_version": np.__version__,
                    "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
                    "device_requested": args.device, "platform": platform.platform()}
            stats["frames"] += 1
            count = len(result.boxes)
            if index in spotcheck_indices:
                standard_config = {**PREDICT_DEFAULTS, "conf": strict_nms_prefilter(.1)}
                direct = standard_model.predict(image, device=runtime_device, **standard_config)[0]
                boxes_equal = torch.equal(result.boxes.data, direct.boxes.data)
                masks_equal = ((result.masks is None and direct.masks is None) or
                    (result.masks is not None and direct.masks is not None and
                     torch.equal(result.masks.data, direct.masks.data)))
                spotchecks.append({"video_id": frame["video_id"], "frame_idx": frame["frame_idx"],
                    "instances": count, "boxes_equal": boxes_equal, "masks_equal": masks_equal,
                    "pass": boxes_equal and masks_equal})
                del direct
            stats["nonempty_frames"] += int(count > 0)
            stats["instances"] += count
            restored = restored_all = None
            if index in cache_indices:
                # Exercise the actual production encoder and SQLite writer/reader.
                # The synthetic one-frame container uses row0; original index/t_sec
                # are recorded in signed metadata, not silently renumbered source data.
                cache_dir = args.output / "roundtrip_samples"
                cache_dir.mkdir(exist_ok=True)
                cache_path = cache_dir / f"{frame['video_id']}_{int(frame['frame_idx']):06d}.sqlite"
                candidates = result.p2_cache_candidates
                metadata = {**candidates.p2_transform, "nms": candidates.p2_nms,
                    "model_status": args.model_status, "checkpoint_sha256": checkpoint_hash,
                    "soft_mask_module_sha256": module_hash, "prediction_config_sha256": prediction_hash,
                    "source_video": frame["video_id"], "source_frame_idx": int(frame["frame_idx"]),
                    "source_t_sec": float(frame["t_sec"]), "purpose": "one-annotated-frame production-serialization validation"}
                cache_writer = VideoCacheWriter(cache_path, metadata, resume=False)
                try:
                    cache_writer.append(0, 0., None, candidates.boxes.xyxy.cpu().numpy(),
                        candidates.boxes.conf.cpu().numpy(), candidates.work_soft_masks.cpu().numpy())
                    cache_writer.finish(1)
                finally:
                    cache_writer.close()
                reader = VideoCacheReader(cache_path)
                try:
                    stored = reader.read(0)
                    if not reader.complete or len(reader) != 1:
                        raise RuntimeError("Roundtrip sample cache is incomplete.")
                finally:
                    reader.close()
                restored_all, inverse_details = restore_work_masks(stored["soft_mask"], image.shape[:2], stored["boxes"],
                    transform=metadata, return_diagnostics=True)
                inverse_diagnostics.extend({"video_id": frame["video_id"], "frame_idx": int(frame["frame_idx"]),
                    "candidate_index": ci, **detail} for ci, detail in enumerate(inverse_details))
                restored = restored_all[result.candidate_index.cpu()].to(result.soft_masks.device)
                stats["cache_frames"] += 1
                stats["cache_nonempty_frames"] += int(count > 0)
                stats["cache_instances"] += len(candidates.boxes)
            for instance in range(count):
                soft_binary = result.soft_masks[instance] > .5
                standard = result.masks.data[instance].bool()
                soft_iou = binary_iou(soft_binary, standard)
                cache_iou = binary_iou(restored[instance] > .5, soft_binary) if restored is not None else None
                stats["softmask_min_iou"] = min(stats["softmask_min_iou"], soft_iou)
                stats["softmask_failed_instances"] += int(soft_iou < .99)
                if cache_iou is not None:
                    stats["cache_min_iou"] = min(stats["cache_min_iou"], cache_iou)
                    stats["cache_failed_instances"] += int(cache_iou < .98)
                row = {"video_id": frame["video_id"], "frame_idx": frame["frame_idx"],
                    "image_rel": frame["image_rel"], "instance_idx": instance,
                    "candidate_index": int(result.candidate_index[instance]), "native_empty": False,
                    "confidence": float(result.boxes.conf[instance]), "softmask_iou": soft_iou,
                    "softmask_pass": soft_iou >= .99, "cache_iou": "" if cache_iou is None else cache_iou,
                    "cache_pass": "" if cache_iou is None else cache_iou >= .98}
                writer.writerow(row)
                if soft_iou < .99 or (cache_iou is not None and cache_iou < .98):
                    failures.append(row)
                    fail_dir = args.output / "failures"
                    fail_dir.mkdir(exist_ok=True)
                    np.savez_compressed(fail_dir / f"{frame['video_id']}_{int(frame['frame_idx']):06d}_{instance}.npz",
                        standard_binary=standard.cpu().numpy(), soft_probabilities=result.soft_masks[instance].cpu().numpy(),
                        restored_probabilities=np.empty((0,), dtype=np.float32) if restored is None else restored[instance].cpu().numpy(),
                        box=result.boxes.xyxy[instance].cpu().numpy())
            # The production cache retains every NMS row. A row dropped by the
            # official native-empty rule has an all-zero binary reference and
            # must not reappear as a false-positive after cache reconstruction.
            for empty_position, candidate_index in enumerate(result.p2_native_empty_candidate_index.tolist()):
                stats["native_empty_candidates"] += 1
                soft_iou = 0. if bool(result.p2_native_empty_soft_has_positive[empty_position]) else 1.
                stats["softmask_failed_instances"] += int(soft_iou < .99)
                stats["softmask_min_iou"] = min(stats["softmask_min_iou"], soft_iou)
                cache_iou = (0. if bool((restored_all[candidate_index] > .5).any()) else 1.) if restored_all is not None else None
                if cache_iou is not None:
                    stats["cache_failed_instances"] += int(cache_iou < .98)
                    stats["cache_min_iou"] = min(stats["cache_min_iou"], cache_iou)
                row = {"video_id": frame["video_id"], "frame_idx": frame["frame_idx"],
                    "image_rel": frame["image_rel"], "instance_idx": -1,
                    "candidate_index": candidate_index, "native_empty": True,
                    "confidence": float(result.p2_cache_candidates.boxes.conf[candidate_index]),
                    "softmask_iou": soft_iou, "softmask_pass": soft_iou >= .99,
                    "cache_iou": "" if cache_iou is None else cache_iou,
                    "cache_pass": "" if cache_iou is None else cache_iou >= .98}
                writer.writerow(row)
                if soft_iou < .99 or (cache_iou is not None and cache_iou < .98):
                    failures.append(row)
                    fail_dir = args.output / "failures"
                    fail_dir.mkdir(exist_ok=True)
                    np.savez_compressed(fail_dir/f"{frame['video_id']}_{int(frame['frame_idx']):06d}_empty_candidate_{candidate_index}.npz",
                        standard_binary=np.zeros(image.shape[:2], dtype=np.bool_),
                        restored_probabilities=np.empty((0,), dtype=np.float32) if restored_all is None else restored_all[candidate_index].cpu().numpy(),
                        box=result.p2_cache_candidates.boxes.xyxy[candidate_index].cpu().numpy())
            if not count:
                writer.writerow({"video_id": frame["video_id"], "frame_idx": frame["frame_idx"],
                    "image_rel": frame["image_rel"], "instance_idx": -1})
            csv_stream.flush()
            if (index + 1) % 10 == 0:
                print(json.dumps({"progress": index + 1, "total": len(selected), **stats}), flush=True)
            del result, restored, restored_all
    summary = {"model_status": args.model_status, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash, "manifest_sha256": sha256(args.manifest),
        "soft_mask_module_sha256": module_hash, "prediction_config_sha256": prediction_hash,
        "prediction_config": prediction_config,
        "validation_runtime": validation_runtime,
        "validation_purpose": "cpu_diagnostic_only" if validation_runtime["device_type"] == "cpu" else ("diagnostic_only" if args.diagnostic_only else "device_fidelity_validation"),
        "standard_predictor_spotchecks": spotchecks,
        "inference": PREDICT_DEFAULTS, "seed": args.seed, "elapsed_seconds": time.perf_counter()-start,
        "reference": "same forward pass and candidates; installed ops.process_mask_native, original pixels, imgsz640",
        "cache_reference": "uncompressed original-coordinate float32 probabilities under identical weights and preprocessing",
        "cache_pipeline": "CacheSoftMaskPredictor direct work640 logit encoder -> VideoCacheWriter SQLite -> VideoCacheReader -> restore_work_masks; match by original NMS candidate_index",
        "stats": stats, "failures": failures, "inverse_diagnostics": inverse_diagnostics,
        "softmask_gate_pass": stats["frames"] >= 200 and stats["instances"] > 0 and stats["softmask_failed_instances"] == 0 and all(item["pass"] for item in spotchecks),
        "cache_gate_pass": None if args.cache_frames == 0 else stats["cache_frames"] >= 50 and stats["cache_instances"] > 0 and stats["cache_failed_instances"] == 0,
        "coverage_note": "Empty frames retained and counted explicitly; nonempty frames/instances reported; all-instance gates, no mean substitution."}
    summary.update({"status": "PASS" if summary["softmask_gate_pass"] and summary["cache_gate_pass"] else "FAIL",
        "annotated_dev_frames": stats["cache_frames"],
        "all_instances_pass": bool(stats["cache_instances"] > 0 and stats["cache_failed_instances"] == 0),
        "min_instance_iou": stats["cache_min_iou"],
        "cache_instance_count": stats["cache_instances"]})
    summary["production_validation_eligible"] = (summary["status"] == "PASS" and
        validation_runtime["device_type"] == "cuda" and not args.diagnostic_only)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"failures", "inverse_diagnostics"}}, indent=2), flush=True)
    if not summary["softmask_gate_pass"] or summary["cache_gate_pass"] is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
