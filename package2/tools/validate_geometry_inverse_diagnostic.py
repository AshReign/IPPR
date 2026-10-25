"""Check experimental inverse on ALL original 50 sampled frames, CPU only.

Uses persisted baseline SQLite fields, fixed checkpoint and exact original sample.
Production code and original failure evidence remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from p2.cache_io import VideoCacheReader
from p2.soft_masks import PREDICT_DEFAULTS, SoftMaskPredictor
from diagnose_geometry_inverse import geometry_inverse, binary_iou


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
    baseline = json.loads((args.validation/"summary.json").read_text())
    sampling = json.loads((args.validation/"sampled_frames.json").read_text())
    selected = [sampling["frames"][index] for index in sampling["cache_indices"]]
    checkpoint = Path(baseline["checkpoint"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == baseline["checkpoint_sha256"]
    model = YOLO(str(checkpoint))
    rows, frame_rows, inverse_times = [], [], []
    started = time.perf_counter()
    for frame in selected:
        image = cv2.imread(str(args.dataset_root/frame["image_rel"]))
        result = model.predict(image, predictor=SoftMaskPredictor, device=torch.device("cpu"), **PREDICT_DEFAULTS)[0]
        stem = f"{frame['video_id']}_{int(frame['frame_idx']):06d}"
        reader = VideoCacheReader(args.validation/"roundtrip_samples"/(stem+".sqlite"))
        try:
            cached, metadata = reader.read(0), reader.metadata
        finally:
            reader.close()
        if not np.array_equal(cached["boxes"], result.p2_cache_candidates.boxes.xyxy.numpy()):
            raise RuntimeError("Candidate boxes/order changed from baseline.")
        if not np.array_equal(cached["conf"], result.p2_cache_candidates.boxes.conf.numpy()):
            raise RuntimeError("Candidate confidences changed from baseline.")
        crop = metadata["prototype_crop_tlbr"]
        crop_shape = crop[2]-crop[0], crop[3]-crop[1]
        for instance in range(len(result.boxes)):
            candidate = int(result.candidate_index[instance])
            native = result.soft_masks[instance] > .5
            assert torch.equal(native, result.masks.data[instance].bool())
            before = time.perf_counter()
            restored, detail = geometry_inverse(cached["soft_mask"][candidate], cached["boxes"][candidate],
                metadata["original_shape_hw"], crop_shape)
            duration = time.perf_counter()-before
            inverse_times.append(duration)
            value = binary_iou(restored > .5, native)
            row = {"video_id": frame["video_id"], "frame_idx": int(frame["frame_idx"]),
                "image_rel": frame["image_rel"], "candidate_index": candidate, "instance_idx": instance,
                "confidence": float(result.boxes.conf[instance]), "iou": value, "pass": value >= .98,
                "inverse_seconds": duration, "native_area_pixels": int(native.sum()), "geometry": detail}
            rows.append(row)
            if value < .98:
                failure_dir = args.output.with_suffix("")
                failure_dir.mkdir(exist_ok=True)
                np.savez_compressed(failure_dir/f"{stem}_{candidate}.npz", native=native.numpy(),
                    restored=restored.numpy(), work=cached["soft_mask"][candidate], box=cached["boxes"][candidate])
        frame_rows.append({"video_id": frame["video_id"], "frame_idx": int(frame["frame_idx"]),
                           "native_instances": len(result.boxes), "raw_candidates": len(cached["conf"])})
        del result, image
        print(json.dumps({"frames": len(frame_rows), "total_frames": len(selected), "instances": len(rows),
            "failed_instances": sum(not row["pass"] for row in rows), "min_iou": min((row["iou"] for row in rows), default=None)}), flush=True)
    report = {"diagnostic_only": True, "production_unchanged": True, "production_validation_eligible": False,
        "checkpoint_sha256": baseline["checkpoint_sha256"], "device": "cpu", "cpu_threads": torch.get_num_threads(),
        "sample_source": str(args.validation/"sampled_frames.json"), "frames": len(frame_rows),
        "instances": len(rows), "failed_instances": sum(not row["pass"] for row in rows),
        "min_iou": min((row["iou"] for row in rows), default=None),
        "elapsed_seconds": time.perf_counter()-started,
        "inverse_timing": {"calls": len(inverse_times), "total_seconds": sum(inverse_times),
            "mean_seconds": float(np.mean(inverse_times)), "median_seconds": float(np.median(inverse_times)),
            "p95_seconds": float(np.quantile(inverse_times, .95)), "max_seconds": max(inverse_times)},
        "whole_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "memory_scope": "Whole CPU process including both model/native validation tensors and inverse; not inverse-only peak.",
        "frame_records": frame_rows, "instance_records": rows}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key:value for key,value in report.items() if key not in {"frame_records", "instance_records"}}, indent=2))


if __name__ == "__main__":
    main()
