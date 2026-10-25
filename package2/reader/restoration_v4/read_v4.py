"""Optional CPU v4 reader for one raw cached candidate; no model or GT loaded.

The unchanged restorer is relocated by assigning BASE_SOURCE to a byte-identical
bundled v3 module after checking both hashes. Original core/reader stay untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PINS = {"restore.py": "e0f8b2e7ef385266b3e9188ba031f3bb49e618051b39975c8223199183f9fa2b",
        "soft_masks_v3.py": "9021ae722b5e33437e3571c679808b1aae8d2afd90a387f6bfb5ecc9339c101f",
        "cache_io.py": "77b14aa00fb5ebaee22448fb0a9bf040dfc90db2ebc20f889904019cdb7d06ed",
        "SPEC.json": "267aff4f4113d4c6da6145cef67f239377af67d91d7b473bc7bba7e0bf40ad32"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value: raise RuntimeError(message)


def load(name, filename):
    path = HERE / filename
    require(sha(path) == PINS[filename], f"Bundled source changed: {filename}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def stamp(path):
    s = path.stat(); return (s.st_size, s.st_mtime_ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--candidate", type=int, required=True, help="One original NMS candidate_index; keeps memory bounded")
    ap.add_argument("--output", type=Path, required=True, help="New diagnostic JSON; no overwrite")
    args = ap.parse_args()
    cache, output = args.cache.resolve(), args.output.resolve()
    require(cache.is_file() and not output.exists() and cache != output, "Cache must exist and output must be new")
    require(args.frame >= 0 and args.candidate >= 0, "Use nonnegative original frame/candidate indices")
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = output.parent / "reader_revision_runtime"; runtime.mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                      OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", YOLO_AUTOINSTALL="false",
                      YOLO_CONFIG_DIR=str(runtime), MPLCONFIGDIR=str(runtime), XDG_CACHE_HOME=str(runtime))
    def deny_network(event, params):
        if event == "socket.connect": raise RuntimeError("No network is allowed in this local CPU reader")
    sys.addaudithook(deny_network)
    import numpy as np
    import torch
    import cv2
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    before = stamp(cache); cache_sha = sha(cache)
    for filename, checksum in PINS.items(): require(sha(HERE / filename) == checksum, "Pinned bundle changed")
    codec = load("optional_p2_cache_io", "cache_io.py")
    restorer = load("optional_p2_v4_restore", "restore.py")
    # Relocate only this reference path. Both adapter and module verify content.
    restorer.BASE_SOURCE = HERE / "soft_masks_v3.py"
    require(restorer.BASE_SOURCE.is_file() and sha(restorer.BASE_SOURCE) == restorer.BASE_SOFT_MASK_SHA256, "Bundled v3 mismatch")
    reader = codec.VideoCacheReader(cache)
    try:
        require(reader.complete, "Read only a complete cache")
        frame = reader.read(args.frame); metadata = reader.metadata
    finally:
        reader.close()
    n = len(frame["conf"]); require(args.candidate < n, "Requested original candidate is absent; frame may be empty")
    i = args.candidate
    raw = frame["soft_mask"][i:i+1]
    box = frame["boxes"][i:i+1]
    raw_sha = hashlib.sha256(raw.tobytes()).hexdigest(); box_sha = hashlib.sha256(box.tobytes()).hexdigest()
    probability, details = restorer.restore_work_masks(raw, metadata["original_shape_hw"], box,
                                                       transform=metadata["transform"], return_diagnostics=True)
    require(probability.device.type == "cpu" and probability.dtype == torch.float32
            and torch.isfinite(probability).all() and torch.all((probability >= 0) & (probability <= 1)), "Invalid returned probabilities")
    require(hashlib.sha256(raw.tobytes()).hexdigest() == raw_sha and hashlib.sha256(box.tobytes()).hexdigest() == box_sha,
            "Reader changed its raw input arrays")
    require(stamp(cache) == before and sha(cache) == cache_sha, "Cache changed during read-only use")
    for filename, checksum in PINS.items(): require(sha(HERE / filename) == checksum, "Bundle changed during read")
    result = {"schema": "p2-optional-v4-reader-smoke-v1", "status": "CPU_READ_AND_RESTORE_EXECUTED_NOT_A_QUALIFICATION_GATE",
              "qualification_status": "consult separately pinned qualification.json; this reader does not establish qualification",
              "input": {"path": str(cache), "sha256": cache_sha, "video_id": metadata["source_identity"]["video_id"],
                        "split": metadata["source_identity"]["split"], "frame_idx": args.frame, "candidate_index": i,
                        "timestamp": frame["timestamp"], "pts": frame["pts"], "confidence": float(frame["conf"][i]),
                        "raw_work_sha256": raw_sha, "raw_box_sha256": box_sha,
                        "checkpoint_sha256": metadata["run_identity"]["checkpoint_sha256"]},
              "restorer": restorer.restorer_manifest(), "bundled_reference_path": str(restorer.BASE_SOURCE),
              "output": {"shape": list(probability.shape), "dtype": str(probability.dtype), "device": str(probability.device),
                         "minimum": float(probability.min()), "maximum": float(probability.max()),
                         "binary_area_at_gt_half": int(torch.count_nonzero(probability > .5)),
                         "float32_probability_sha256": hashlib.sha256(probability.numpy().tobytes()).hexdigest()},
              "diagnostics": details, "source_pins": PINS, "adapter_sha256": sha(Path(__file__)),
              "original_cache_and_raw_arrays_unchanged": True, "model_run": False, "native_or_gt_reference_loaded": False,
              "limitations": ["unmodified original model fields only; never T1/TCA modified maps", "not lossless or universal native-mask recovery", "no mask-quality score or IoU was computed"],
              "runtime": {"python": sys.version, "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                          "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "torchvision", "ultralytics", "opencv-python")}}}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(output), "sha256": sha(output), "diagnostics": details,
                      "no_model_or_gt": True}, ensure_ascii=False))


if __name__ == "__main__": main()
