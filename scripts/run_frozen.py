"""Run the frozen T0/T1/T2 configurations on one split and write COCO results.

No tuning happens here: the parameters come from configs/frozen/{t0,t1,t2}.yaml,
which were selected on dev (scripts/select_configs.py). Each video runs in its
own worker process; the annotated frames are mapped to COCO image IDs with P1's
data/manifests/frames_index.csv, so no ground truth is needed to run this.

Outputs (in --out-dir):
    <split>_T0.json, <split>_T1.json, <split>_T2.json
        COCO results lists (image_id, category_id, segmentation, score=1.0).
        Masks are compressed RLE at the original resolution, area filter applied.
        Keep the list order when scoring: within a frame it is the matching order
        (P5's evaluate with min_score=0).
    <split>_run_meta.json  frozen config IDs and their file hashes, git commit,
        machine and timing.
    <split>_scores.csv     only with --ripvis-root: pooled P, R, F2 per condition
        (a sanity check; the reported val scores come from Package 5).

Example:
    python scripts/run_frozen.py --split val --cache-dir package2/cache/final-val \
        --p2-reader package2/reader --tca-module p3.tca_p3 --out-dir runs/val_frozen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tune_dev import git_commit, load_tca, machine_info  # noqa: E402
from temporal.cache_io import load_video_cache  # noqa: E402
from temporal.t0 import run_t0  # noqa: E402
from temporal.t1 import run_t1  # noqa: E402
from temporal.t2_adapter import run_t2  # noqa: E402

CONDITIONS = ("T0", "T1", "T2")
_STATE: dict = {}


def load_frozen(frozen_dir: Path) -> dict[str, dict]:
    """The frozen configuration of each condition, plus the SHA-256 of its file."""
    out = {}
    for cond in CONDITIONS:
        path = frozen_dir / f"{cond.lower()}.yaml"
        raw = path.read_bytes()
        cfg = yaml.safe_load(raw)
        if cfg["condition"] != cond:
            raise ValueError(f"{path}: condition is {cfg['condition']}, expected {cond}")
        cfg["file_sha256"] = hashlib.sha256(raw).hexdigest()
        out[cond] = cfg
    return out


def frame_ids(frames_index: Path, split: str) -> dict[str, dict[int, int]]:
    """{video_id: {frame_idx: image_id}} for the annotated frames of one split."""
    df = pd.read_csv(frames_index)
    df = df[df["split"] == split]
    return {v: dict(zip(g["frame_idx"].astype(int), g["image_id"].astype(int)))
            for v, g in df.groupby("video_id")}


def _init_worker(frozen: dict, ids: dict, tca_module: str, p2_reader: str | None) -> None:
    _STATE.update(frozen=frozen, ids=ids, p2_reader=p2_reader,
                  run_tca=load_tca(tca_module))


def process_video(cache_path: str) -> tuple[str, dict[str, list[dict]], dict[str, float]]:
    frozen = _STATE["frozen"]
    cache = load_video_cache(cache_path, _STATE["p2_reader"])
    ids = _STATE["ids"].get(cache.video_id)
    if not ids:
        raise ValueError(f"{cache.video_id}: no annotated frames in frames_index.csv")
    cache.attach_image_ids(ids)
    results, seconds = {}, {}
    runners = {"T0": lambda p: run_t0(cache, p),
               "T1": lambda p: run_t1(cache, p),
               "T2": lambda p: run_t2(cache, p, _STATE["run_tca"])}
    for cond in CONDITIONS:
        t = time.perf_counter()
        results[cond] = runners[cond](frozen[cond]["params"])
        seconds[cond] = time.perf_counter() - t
    cache.close()
    return cache.video_id, results, seconds


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["dev", "val"])
    ap.add_argument("--cache-dir", required=True, help="one <video_id>.sqlite per video")
    ap.add_argument("--p2-reader", help="Package 2's 'reader' folder (needed for .sqlite)")
    ap.add_argument("--frozen-dir", default=str(ROOT / "configs/frozen"))
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/videos.csv"))
    ap.add_argument("--frames-index", default=str(ROOT / "data/manifests/frames_index.csv"))
    ap.add_argument("--tca-module", default="temporal.tca",
                    help="module that provides run_tca (Package 3)")
    ap.add_argument("--ripvis-root", help="optional: also score with P5's evaluator")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    frozen = load_frozen(Path(args.frozen_dir))
    man = pd.read_csv(args.manifest)
    videos = sorted(man.loc[man["split"] == args.split, "video_id"])
    missing = [v for v in videos if not (Path(args.cache_dir) / f"{v}.sqlite").exists()]
    if missing:
        sys.exit(f"missing caches for {len(missing)} {args.split} videos: {missing[:5]}")
    ids = frame_ids(Path(args.frames_index), args.split)

    start = time.time()
    with Pool(args.workers, initializer=_init_worker,
              initargs=(frozen, ids, args.tca_module, args.p2_reader)) as pool:
        per_video = {v: (res, sec) for v, res, sec in pool.imap_unordered(
            process_video, [str(Path(args.cache_dir) / f"{v}.sqlite") for v in videos])}
    wall = time.time() - start

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for cond in CONDITIONS:
        # Videos in sorted order; each video's list keeps its matching order.
        preds = [r for v in videos for r in per_video[v][0][cond]]
        (out / f"{args.split}_{cond}.json").write_text(json.dumps(preds))

    meta = {
        "split": args.split, "videos": videos, "git_commit": git_commit(),
        "frozen": {c: {"config_id": f["config_id"], "params": f["params"],
                       "file_sha256": f["file_sha256"]} for c, f in frozen.items()},
        "tca_module": args.tca_module,
        "n_predictions": {c: sum(len(per_video[v][0][c]) for v in videos) for c in CONDITIONS},
        "seconds_per_condition": {c: sum(per_video[v][1][c] for v in videos)
                                  for c in CONDITIONS},
        "machine": machine_info(), "wall_seconds": round(wall, 1),
    }
    (out / f"{args.split}_run_meta.json").write_text(json.dumps(meta, indent=2))

    if args.ripvis_root:
        from eval.evaluator import evaluate, load_gt, summarise
        gt = load_gt(args.split, args.ripvis_root, videos=videos)
        rows = []
        for cond in CONDITIONS:
            preds = json.loads((out / f"{args.split}_{cond}.json").read_text())
            rows.append({"condition": cond, "config_id": frozen[cond]["config_id"],
                         **summarise(evaluate(preds, gt))})
        scores = pd.DataFrame(rows)
        scores.to_csv(out / f"{args.split}_scores.csv", index=False)
        print(scores.to_string(index=False))
    print(f"done: {len(videos)} videos, {wall:.0f} s -> {out}")


if __name__ == "__main__":
    main()
