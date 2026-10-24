"""Runtime of the full pipeline on one machine: decoding, YOLO, and T0/T1/T2.

Two passes per video, so each stage is timed on its own:

1. Model pass. Every frame is decoded with PyAV (as Package 2 did) and run
   through the frozen checkpoint with Package 2's ``CacheSoftMaskPredictor``
   (imgsz 640, batch 1, conf 0.1, NMS IoU 0.7, max_det 300, retina masks, fp32).
   The timed model step includes copying boxes, confidences and working soft
   masks back to the CPU, because post-processing needs them there.
   The candidate count per frame is compared with the cache as a sanity check.
2. Post-processing pass. The video's cache is read into memory (timed
   separately as cache reading), then T0, T1 and T2 run with the frozen
   configurations. For timing, every frame is treated as annotated, so each
   method encodes an output for every frame, as it would when deployed.

Everything runs in one process, one video at a time. A short warm-up on the
first video is run before any timing and discarded. End-to-end FPS per
condition = frames / (decode + model + post-processing); cache reading is
reported separately and not included.

Outputs (in --out-dir): fps_per_video.csv, fps_summary.csv, fps_meta.json.

Example:
    python scripts/measure_fps.py --videos-dir ../ripvis_val_videos \
        --p2-root ~/Downloads/IPPR_P2_Formal_val/package2 --device mps \
        --tca-module p3.tca_p3 --out-dir runs/fps_val
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_frozen import load_frozen  # noqa: E402
from scripts.tune_dev import git_commit, load_tca, machine_info  # noqa: E402
import temporal.common as common  # noqa: E402
import temporal.t2_adapter as t2_adapter  # noqa: E402
from temporal.cache_io import load_video_cache  # noqa: E402
from temporal.t0 import run_t0  # noqa: E402
from temporal.t1 import run_t1  # noqa: E402
from temporal.t2_adapter import run_t2  # noqa: E402

CONDITIONS = ("T0", "T1", "T2")


class OutputClock:
    """Time spent producing evaluation output, inside each method's total.

    Wraps the shared ``encode_result`` (upsample to original resolution, area
    filter, COCO RLE) and T2's decoding of TCA's working-resolution RLE, so the
    post-processing time can be split into the method itself and the output
    step that is only needed for scoring.
    """

    def __init__(self):
        self.seconds = 0.0
        for mod, name in ((common, "encode_result"), (t2_adapter, "encode_result"),
                          (t2_adapter, "_decode")):
            setattr(mod, name, self._timed(getattr(mod, name)))

    def _timed(self, fn):
        def wrapper(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.seconds += time.perf_counter() - t
        return wrapper


OUTPUT_CLOCK: OutputClock | None = None


class Timer:
    """perf_counter timer that waits for the GPU queue before reading the clock."""

    def __init__(self, device: str):
        import torch
        if device.startswith("cuda"):
            self.sync = torch.cuda.synchronize
        elif device == "mps":
            self.sync = torch.mps.synchronize
        else:
            self.sync = lambda: None

    def now(self) -> float:
        self.sync()
        return time.perf_counter()


def decoded_frames(video: Path, max_frames: int | None):
    """Yield (seconds spent decoding, BGR frame), in presentation order."""
    import av
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1      # as in Package 2's export
        frames = container.decode(stream)
        n = 0
        while max_frames is None or n < max_frames:
            t = time.perf_counter()
            try:
                frame = next(frames)
            except StopIteration:
                return
            image = frame.to_ndarray(format="bgr24")
            yield time.perf_counter() - t, image
            n += 1


def predict(model, image, device, inference, predictor_cls):
    r = model.predict(source=image, predictor=predictor_cls, device=device, batch=1,
                      **inference)[0]
    # The copy to the CPU is part of the model step: T0/T1/T2 run on the CPU.
    r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    r.work_soft_masks.cpu().numpy()
    return len(conf)


def model_pass(model, video: Path, cache, args, inference, predictor_cls, timer) -> dict:
    counts = [len(f.instances) for f in cache.frames] if cache is not None else None
    decode_s = model_s = 0.0
    n = mismatched = 0
    for dt, image in decoded_frames(video, args.max_frames):
        t = timer.now()
        k = predict(model, image, args.device, inference, predictor_cls)
        model_s += timer.now() - t
        decode_s += dt
        if counts is not None and (n >= len(counts) or counts[n] != k):
            mismatched += 1
        n += 1
    return {"frames": n, "decode_s": decode_s, "model_s": model_s,
            "candidate_count_mismatch_frames": mismatched if counts is not None else None}


def post_pass(cache_path: Path, args, frozen, run_tca) -> dict:
    t = time.perf_counter()
    lazy = load_video_cache(cache_path, args.p2_root / "reader")
    frames = list(lazy.frames[: args.max_frames] if args.max_frames else lazy.frames)
    read_s = time.perf_counter() - t
    lazy.close()
    for f in frames:                     # every frame gets an output during timing
        f.image_id = f.frame_idx
    cache = dataclasses.replace(lazy, frames=frames,
                                image_ids={f.frame_idx: f.frame_idx for f in frames})
    out = {"cache_read_s": read_s}
    runners = {"T0": lambda p: run_t0(cache, p), "T1": lambda p: run_t1(cache, p),
               "T2": lambda p: run_t2(cache, p, run_tca)}
    for cond in CONDITIONS:
        OUTPUT_CLOCK.seconds = 0.0
        t = time.perf_counter()
        res = runners[cond](frozen[cond]["params"])
        out[f"post_{cond}_s"] = time.perf_counter() - t
        out[f"output_{cond}_s"] = OUTPUT_CLOCK.seconds
        out[f"outputs_{cond}"] = len(res)
    return out


def summarise(per_video: pd.DataFrame) -> pd.DataFrame:
    """Pooled over videos: total frames / total seconds for each stage."""
    n = per_video["frames"].sum()
    rows = []

    def add(stage, seconds):
        rows.append({"stage": stage, "seconds": seconds, "frames": n,
                     "ms_per_frame": 1000 * seconds / n, "fps": n / seconds})

    dec, mod = per_video["decode_s"].sum(), per_video["model_s"].sum()
    add("decode", dec)
    add("model (YOLO + copy to CPU)", mod)
    add("cache read (not in end-to-end)", per_video["cache_read_s"].sum())
    for c in CONDITIONS:
        post, output = per_video[f"post_{c}_s"].sum(), per_video[f"output_{c}_s"].sum()
        add(f"post-processing {c} (total)", post)
        add(f"  {c} method only", post - output)
        add(f"  {c} output: upsample + area filter + RLE", output)
    for c in CONDITIONS:
        post = per_video[f"post_{c}_s"].sum()
        add(f"end-to-end {c} (decode + model + {c} total)", dec + mod + post)
        add(f"end-to-end {c}, no output step", dec + mod + post - per_video[f"output_{c}_s"].sum())
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-dir", required=True, type=Path, help="<video_id>.mp4 files")
    ap.add_argument("--p2-root", required=True, type=Path,
                    help="extracted Package 2 folder (p2/, reader/, weights/, cache/)")
    ap.add_argument("--weights", type=Path, help="default: <p2-root>/weights/epoch5.pt")
    ap.add_argument("--cache-dir", type=Path, help="default: <p2-root>/cache/final-val")
    ap.add_argument("--split", default="val", choices=["dev", "val"])
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/videos.csv"))
    ap.add_argument("--videos", nargs="+", help="optional subset of video IDs")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--warmup", type=int, default=20, help="warm-up frames, not timed")
    ap.add_argument("--max-frames", type=int, help="per video; for smoke tests only")
    ap.add_argument("--frozen-dir", default=str(ROOT / "configs/frozen"))
    ap.add_argument("--tca-module", default="temporal.tca")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    args.p2_root = args.p2_root.expanduser().resolve()
    weights = args.weights or args.p2_root / "weights/epoch5.pt"
    cache_dir = args.cache_dir or args.p2_root / ("cache/final-val" if args.split == "val"
                                                  else "cache/dev")

    sys.path.insert(0, str(args.p2_root))
    import torch
    import ultralytics
    from ultralytics import YOLO
    from p2.soft_masks import CacheSoftMaskPredictor, PREDICT_DEFAULTS

    global OUTPUT_CLOCK
    OUTPUT_CLOCK = OutputClock()
    inference = dict(PREDICT_DEFAULTS)
    frozen = load_frozen(Path(args.frozen_dir))
    run_tca = load_tca(args.tca_module)
    man = pd.read_csv(args.manifest)
    videos = args.videos or sorted(man.loc[man["split"] == args.split, "video_id"])
    missing = [v for v in videos if not (args.videos_dir / f"{v}.mp4").exists()]
    if missing:
        sys.exit(f"missing videos: {missing}")

    timer = Timer(args.device)
    model = YOLO(str(weights))
    for _, image in decoded_frames(args.videos_dir / f"{videos[0]}.mp4", args.warmup):
        predict(model, image, args.device, inference, CacheSoftMaskPredictor)
    timer.now()

    rows = []
    start = time.time()
    for i, v in enumerate(videos, 1):
        cache_path = cache_dir / f"{v}.sqlite"
        cache = load_video_cache(cache_path, args.p2_root / "reader") if cache_path.exists() else None
        row = {"video_id": v, **model_pass(model, args.videos_dir / f"{v}.mp4", cache, args,
                                           inference, CacheSoftMaskPredictor, timer)}
        if cache is not None:
            orig = cache.orig_size
            cache.close()
            row.update(height=orig[0], width=orig[1], **post_pass(cache_path, args, frozen, run_tca))
        rows.append(row)
        print(f"[{i}/{len(videos)}] {v}: {row['frames']} frames, "
              f"model {1000 * row['model_s'] / max(row['frames'], 1):.1f} ms/frame", flush=True)
    wall = time.time() - start

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_video = pd.DataFrame(rows)
    per_video.to_csv(args.out_dir / "fps_per_video.csv", index=False)
    summary = summarise(per_video.dropna(subset=["cache_read_s"]))
    summary.to_csv(args.out_dir / "fps_summary.csv", index=False)
    meta = {
        "split": args.split, "videos": videos, "device": args.device,
        "warmup_frames": args.warmup, "max_frames": args.max_frames,
        "inference": inference, "predictor": "p2.soft_masks.CacheSoftMaskPredictor",
        "weights": str(weights), "decode": "PyAV, thread_count=1, bgr24",
        "post_processing": "single process, CPU, every frame treated as annotated; "
                           "output step = upsample to original resolution + area filter + "
                           "COCO RLE (+ decoding TCA's RLE for T2)",
        "candidate_count_note": "MPS vs the CUDA-built cache; frames whose candidate "
                                "count differs (numerics near conf 0.1 / NMS)",
        "frozen": {c: f["config_id"] for c, f in frozen.items()},
        "tca_module": args.tca_module, "git_commit": git_commit(),
        "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                     "numpy": np.__version__},
        "machine": machine_info(), "wall_seconds": round(wall, 1),
    }
    (args.out_dir / "fps_meta.json").write_text(json.dumps(meta, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
