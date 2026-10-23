"""Package 4 dev search: score every T0/T1/T2 configuration on the dev videos.

Each dev video runs in its own worker process (step 5). For every configuration
the script scores the video with Package 5's evaluator (eval/evaluator.py, the
official RipVIS rule), logs TP, FP, FN and the no-rip frame counts per video,
then pools them over videos to get P, R, F2 and the false-alarm rate.

Outputs (in --out-dir):
    dev_search_per_video.csv  one row per (configuration, video)
    dev_search.csv            one row per configuration, pooled over videos
    dev_search_meta.json      inputs, git commit, machine (CPU, RAM) and timing

All conditions start from the same candidate pool (conf >= 0.1); T0 tunes its
own confidence threshold over the grid in the search space.

Example:
    python scripts/tune_dev.py --cache-dir cache/ --p2-reader p2_reader/ \
        --ripvis-root C:/data/ripvis --out-dir runs/dev_search
    (--ripvis-root defaults to configs/paths.yaml; --gt-json is for synthetic tests)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.evaluator import load_gt  # noqa: E402
from temporal.cache_io import load_video_cache  # noqa: E402
from temporal.scoring import (COUNT_FIELDS, Counts, frame_to_image, gt_from_coco,  # noqa: E402
                              gt_subset, metrics, score)
from temporal.search import (load_search_space, t0_config, t0_configs,  # noqa: E402
                             t1_configs, t2_configs)
from temporal.t0 import run_t0  # noqa: E402
from temporal.t1 import run_t1_theta_sweep  # noqa: E402
from temporal.t2_adapter import run_t2  # noqa: E402

# Per-process state, filled once by the pool initialiser so the ground truth is
# not re-loaded (or pickled) for every video.
_STATE: dict = {}


def git_commit() -> str:
    """Current commit hash, with '-dirty' if there are uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                        text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not-a-git-repo"


def machine_info() -> dict:
    """CPU and RAM of the tuning machine, recorded with every run (Ash, step 11)."""
    info = {"hostname": platform.node(), "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(), "python": platform.python_version()}
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        info["ram_gb"] = None
    return info


def load_tca(module_name: str):
    """Import Package 3's run_tca(video_cache, params) by module name."""
    return importlib.import_module(module_name).run_tca


def build_gt(gt_json: str | None, ripvis_root: str | None, videos: list[str]):
    """P5's dev ground truth (real runs) or one built from a COCO file (tests)."""
    if gt_json:
        return gt_from_coco(gt_json)
    return load_gt("dev", ripvis_root, videos=videos)


def _init_worker(gt_args: tuple, configs: list[dict], tca_module: str | None,
                 p2_reader: str | None = None) -> None:
    _STATE["p2_reader"] = p2_reader
    _STATE["gt"] = build_gt(*gt_args)
    _STATE["configs"] = configs
    _STATE["run_tca"] = load_tca(tca_module) if tca_module else None


def _row(cfg: dict, video_id: str, counts: Counts, seconds: float) -> dict:
    return {"config_id": cfg["config_id"], "condition": cfg["condition"],
            "video_id": video_id, **vars(counts), "seconds": seconds}


def process_video(cache_path: str) -> list[dict]:
    """Score every configuration on one video. Runs inside a worker process."""
    gt, configs = _STATE["gt"], _STATE["configs"]
    cache = load_video_cache(cache_path, _STATE.get("p2_reader"))
    gt = gt_subset(gt, cache.video_id)
    if gt.frames.empty:
        raise ValueError(f"{cache.video_id}: no annotated frames in the dev ground truth")
    cache.attach_image_ids(frame_to_image(gt))
    rows: list[dict] = []

    for cfg in (c for c in configs if c["condition"] == "T0"):
        t = time.perf_counter()
        res = run_t0(cache, cfg["params"])
        rows.append(_row(cfg, cache.video_id, score(res, gt), time.perf_counter() - t))

    # T1: the window mean is shared by all thresholds of the same window.
    by_window: dict[float, list[dict]] = defaultdict(list)
    for cfg in (c for c in configs if c["condition"] == "T1"):
        by_window[cfg["params"]["window_s"]].append(cfg)
    for window_s, cfgs in by_window.items():
        t = time.perf_counter()
        thetas = [c["params"]["theta"] for c in cfgs]
        sweep = run_t1_theta_sweep(cache, window_s, thetas,
                                   cfgs[0]["params"]["min_area_frac"])
        per_cfg = (time.perf_counter() - t) / len(cfgs)   # amortised, not official FPS
        for cfg in cfgs:
            counts = score(sweep[cfg["params"]["theta"]], gt)
            rows.append(_row(cfg, cache.video_id, counts, per_cfg))

    for cfg in (c for c in configs if c["condition"] == "T2"):
        t = time.perf_counter()
        res = run_t2(cache, cfg["params"], _STATE["run_tca"])
        rows.append(_row(cfg, cache.video_id, score(res, gt), time.perf_counter() - t))
    cache.close()
    return rows


def pool_over_videos(per_video: pd.DataFrame, configs: list[dict]) -> pd.DataFrame:
    """Sum counts over videos and compute the pooled metrics per configuration."""
    params = {c["config_id"]: json.dumps(c["params"], sort_keys=True) for c in configs}
    rows = []
    for (cid, cond), g in per_video.groupby(["config_id", "condition"], sort=True):
        counts = Counts(**{k: int(g[k].sum()) for k in COUNT_FIELDS})
        rows.append({"config_id": cid, "condition": cond, "params": params[cid],
                     "n_videos": len(g), **vars(counts), **metrics(counts),
                     "seconds_total": float(g["seconds"].sum())})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True,
                    help="one <video_id>.sqlite (Package 2) or <video_id>.pkl per video")
    ap.add_argument("--p2-reader", help="Package 2's 'reader' folder (needed for .sqlite)")
    ap.add_argument("--ripvis-root", help="RipVIS folder for P5's load_gt (default: configs/paths.yaml)")
    ap.add_argument("--gt-json", help="COCO file instead of P5's dev split (synthetic tests only)")
    ap.add_argument("--manifest", help="CSV with video_id,split; only split=='dev' is used")
    ap.add_argument("--t0-conf", type=float,
                    help="optional: score a single T0 threshold instead of the grid")
    ap.add_argument("--search-space", default=str(ROOT / "configs/search_space.yaml"))
    ap.add_argument("--conditions", nargs="+", default=["T0", "T1", "T2"])
    ap.add_argument("--tca-module", default="temporal.tca",
                    help="module that provides run_tca (Package 3)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    space = load_search_space(args.search_space)
    # T0 is always needed for R0.
    configs = [t0_config(args.t0_conf, space)] if args.t0_conf is not None else t0_configs(space)
    if "T1" in args.conditions:
        configs += t1_configs(space)
    if "T2" in args.conditions:
        configs += t2_configs(space)
    tca_module = args.tca_module if "T2" in args.conditions else None

    cache_paths = sorted([*Path(args.cache_dir).glob("*.sqlite"),
                          *Path(args.cache_dir).glob("*.pkl")])
    if args.manifest:
        man = pd.read_csv(args.manifest)
        dev = set(man.loc[man["split"] == "dev", "video_id"])
        cache_paths = [p for p in cache_paths if p.stem in dev]
    if not cache_paths:
        sys.exit("no cache files selected")

    start = time.time()
    with Pool(args.workers, initializer=_init_worker,
              initargs=((args.gt_json, args.ripvis_root, [p.stem for p in cache_paths]),
                        configs, tca_module, args.p2_reader)) as pool:
        per_video_rows = [r for rows in pool.imap_unordered(
            process_video, [str(p) for p in cache_paths]) for r in rows]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_video = pd.DataFrame(per_video_rows).sort_values(["config_id", "video_id"])
    per_video.to_csv(out / "dev_search_per_video.csv", index=False)
    pooled = pool_over_videos(per_video, configs)
    pooled.to_csv(out / "dev_search.csv", index=False)
    meta = {"git_commit": git_commit(), "search_space": args.search_space,
            "t0_conf": args.t0_conf, "videos": [p.stem for p in cache_paths], "scorer": "eval.evaluator (P5)",
            "n_configs": len(configs), "machine": machine_info(), "wall_seconds": round(time.time() - start, 1)}
    (out / "dev_search_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"scored {len(configs)} configs on {len(cache_paths)} videos -> {out}")


if __name__ == "__main__":
    main()
