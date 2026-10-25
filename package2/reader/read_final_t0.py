"""Read one final P2 SQLite frame and produce the confirmed T0 COCO predictions.

No model, GPU, GT, or legacy inverse is used. The cache retains raw candidates;
the adapter applies the frozen .10 confidence, working-area filter, stable score
order and pinned P5 encode. Supply the caller's real COCO image_id explicitly.
This example does not implement T1/TCA.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--image-id", type=int, required=True)
    parser.add_argument("--output", type=Path, help="New JSON file; otherwise write JSON to stdout")
    args = parser.parse_args()
    root = args.p2_root.resolve()
    if args.frame_idx < 0 or args.image_id < 0:
        raise ValueError("frame_idx and image_id must be nonnegative integers")
    pins = {"p2/t0_output.py": "1a578f1dada74423bb053489a50991f16733a41eaea6903d3c31ac8d3ae1163a",
            "p5_evaluator.py": "b539ae46869407ffdfce9a8d4582d0a72bd722bdead816453d4cc706ab49f8c7",
            "p2/cache_io.py": "77b14aa00fb5ebaee22448fb0a9bf040dfc90db2ebc20f889904019cdb7d06ed"}
    for relative, expected in pins.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Pinned reader/adapter/encoder source differs: {relative}")
    cache_io = module("example_cache_io", root / "p2/cache_io.py")
    adapter = module("example_t0_adapter", root / "p2/t0_output.py")
    encoder = module("example_p5_encoder", root / "p5_evaluator.py")
    reader = cache_io.VideoCacheReader(args.cache)
    try:
        metadata = reader.metadata
        identity = metadata["run_identity"]
        if not reader.complete or identity["kind"] != "final" or not identity["gates"]["frozen"]:
            raise ValueError("This example requires a complete frozen final cache")
        if identity["checkpoint_sha256"] != "278ba795436c1299621b68a6f775a4af8b284afcab9c242384abff877a6906ca":
            raise ValueError("This example is pinned to selected epoch5.pt")
        if identity["gates"]["t0_confidence"] != .1:
            raise ValueError("Cache frozen T0 confidence differs")
        if identity["cache_io_module_sha256"] != pins["p2/cache_io.py"]:
            raise ValueError("Cache was written by a different codec implementation")
        if metadata["transform"]["work_has_padding"] or metadata["soft_mask_dtype"] != "float16":
            raise ValueError("Unexpected working-mask geometry or dtype")
        frame = reader.read(args.frame_idx)
        candidates = [{"image_id": args.image_id, "candidate_index": index, "class_id": 0, "conf": float(conf)}
                      for index, conf in enumerate(frame["conf"])]
        predictions = adapter.from_work(candidates, frame["soft_mask"], metadata["original_shape_hw"],
                                        "0.10", encoder.encode_mask)
        result = {"video_id": metadata["source_identity"]["video_id"], "frame_idx": frame["frame_idx"],
                  "timestamp": frame["timestamp"], "pts": frame["pts"], "image_id": args.image_id,
                  "raw_candidate_count": len(candidates), "t0_confidence": .1, "predictions": predictions,
                  "note": "Pass predictions in this exact list order to fixed P5 with min_score=0; timestamp is actual decoded PTS time."}
    finally:
        reader.close()
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
