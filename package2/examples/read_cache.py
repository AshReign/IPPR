"""Read one cached frame; optionally restore probabilities to original coordinates.

  python package2/examples/read_cache.py PATH.sqlite --frame 0
  python package2/examples/read_cache.py PATH.sqlite --frame 20 --restore

This prints a schema/shape example, not P5 metrics. Restore only frames needed by
your consumer; restoring every full-resolution mask at once is unnecessary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2), default=1,
                        help="CPU restoration thread budget; PyTorch interop and OpenCV use one thread")
    args = parser.parse_args()
    from p2.cache_io import VideoCacheReader

    reader = VideoCacheReader(args.cache)
    try:
        frame = reader.read(args.frame)
        metadata = reader.metadata
        count = len(frame["conf"])
        result = {"cache_complete": reader.complete, "cached_frames": len(reader),
                  "kind": metadata["run_identity"]["kind"],
                  "frozen": metadata["run_identity"]["gates"]["frozen"],
                  "mask_quality_validated": (metadata["run_identity"]["gates"]["frozen"] or
                                             metadata["run_identity"]["gates"].get("mask_quality_validated", False)),
                  "pilot_validation": metadata["run_identity"]["gates"].get("pilot_validation"),
                  "video_id": metadata["source_identity"]["video_id"],
                  "frame_idx": frame["frame_idx"], "timestamp_seconds": frame["timestamp"],
                  "pts": frame["pts"], "candidate_index": list(range(count)),
                  "boxes_shape": list(frame["boxes"].shape), "boxes_coordinates": "original xyxy",
                  "confidence_shape": list(frame["conf"].shape),
                  "work_mask_shape": list(frame["soft_mask"].shape),
                  "work_mask_dtype": str(frame["soft_mask"].dtype),
                  "soft_mask_format_version": metadata["transform"]["format_version"],
                  "inference_runtime": metadata.get("inference_runtime"),
                  "original_shape_hw": metadata["original_shape_hw"],
                  "empty_detection_frame": count == 0,
                  "checkpoint_sha256": metadata["run_identity"]["checkpoint_sha256"]}
        if args.restore:
            import cv2
            import torch

            torch.set_num_threads(args.cpu_threads)
            torch.set_num_interop_threads(1)
            cv2.setNumThreads(1)
            from p2.soft_masks import restore_work_masks

            original_probabilities, diagnostics = restore_work_masks(
                frame["soft_mask"], metadata["original_shape_hw"], frame["boxes"],
                transform=metadata["transform"], return_diagnostics=True
            )
            # T0/P5 additionally need their agreed confidence, component/area and
            # export/matching protocol. Do not mistake this shape example for it.
            example_binary_masks = original_probabilities > 0.5
            result["restored_probability_shape"] = list(original_probabilities.shape)
            result["restored_probability_dtype"] = str(original_probabilities.dtype)
            result["example_binary_shape"] = list(example_binary_masks.shape)
            result["restoration_status_counts"] = dict(Counter(item["status"] for item in diagnostics))
            result["restoration_fallback_reasons"] = dict(Counter(
                item["reason"] for item in diagnostics if item["status"] == "fallback"
            ))
            if (torch.get_num_threads() != args.cpu_threads or torch.get_num_interop_threads() != 1
                    or cv2.getNumThreads() != 1 or original_probabilities.device.type != "cpu"):
                raise RuntimeError("Restoration changed the requested CPU device/thread budget")
            result["restoration_runtime"] = {"device_type": "cpu", "torch_threads": torch.get_num_threads(),
                                             "torch_interop_threads": torch.get_num_interop_threads(),
                                             "opencv_threads": cv2.getNumThreads()}
        print(json.dumps(result, indent=2))
    finally:
        reader.close()


if __name__ == "__main__":
    main()
