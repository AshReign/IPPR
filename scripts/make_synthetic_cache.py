"""Builds a small synthetic dataset in the agreed cache + COCO formats.

Lets Package 4 run and test the whole dev search before Package 2's real cache
exists. The scenes are simple on purpose: elliptical "rips" drift slowly, the
fake segmenter misses some frames and fires short-lived false positives, so
temporal processing has something to fix.

    python scripts/make_synthetic_cache.py --out-dir runs/synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from temporal.cache_io import Frame, Instance, VideoCache, save_video_cache  # noqa: E402

ORIG = (180, 320)   # (H, W) original resolution
WORK = (90, 160)    # (h, w) cached soft-mask resolution
DURATION_S = 6.0
ANNOT_RATE = 5.0    # annotated frames per second, as in most RipVIS videos

# video_id, fps, rip centres (x, y) at t = 0 in original pixels
VIDEOS = [
    ("RipVIS-901", 30.0, [(110, 90)]),
    ("RipVIS-902", 60.0, [(80, 80), (230, 100)]),
    ("RipVIS-903", 24.0, [(160, 95)]),
    ("RipVIS-NR-901", 30.0, []),
]


def ellipse(size, cx, cy, ax, ay) -> np.ndarray:
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    return ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0


def to_work(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(mask.astype(np.float32), (WORK[1], WORK[0]),
                      interpolation=cv2.INTER_AREA)


def make_dataset(out_dir: str | Path, seed: int = 0,
                 p_detect: float = 0.8, p_false: float = 0.12) -> dict[str, Path]:
    """Write caches, a COCO ground-truth file and a manifest; return their paths."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    images, anns, manifest = [], [], []
    img_id = ann_id = 1

    for vid, fps, centres in VIDEOS:
        n_frames = int(DURATION_S * fps)
        step = int(round(fps / ANNOT_RATE))
        frames = []
        for fidx in range(n_frames):
            t = fidx / fps
            truths = [ellipse(ORIG, cx + 3 * t, cy, 22, 34) for cx, cy in centres]
            instances = []
            for gt in truths:
                if rng.random() < p_detect:     # the fake segmenter sometimes misses
                    soft = gaussian_filter(to_work(gt), 1.5) * rng.uniform(0.75, 1.0)
                    instances.append(Instance(np.zeros(4, np.float32),
                                              float(rng.uniform(0.4, 0.95)),
                                              np.clip(soft, 0, 1).astype(np.float16)))
            if rng.random() < p_false:          # one-frame false positive (foam, glare)
                fx, fy = rng.uniform(10, WORK[1] - 10), rng.uniform(10, WORK[0] - 10)
                soft = gaussian_filter(ellipse(WORK, fx, fy, 7, 7).astype(np.float32), 1.0)
                instances.append(Instance(np.zeros(4, np.float32),
                                          float(rng.uniform(0.15, 0.6)),
                                          soft.astype(np.float16)))
            frames.append(Frame(fidx, t, instances))

            if fidx % step == 0:                # annotated frame -> COCO entries
                images.append({"id": img_id, "file_name": f"{vid}_{fidx:05d}.jpg",
                               "height": ORIG[0], "width": ORIG[1]})
                for gt in truths:
                    rle = mask_utils.encode(np.asfortranarray(gt.astype(np.uint8)))
                    anns.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                 "iscrowd": 0, "area": float(gt.sum()),
                                 "bbox": [float(v) for v in mask_utils.toBbox(rle)],
                                 "segmentation": {"size": rle["size"],
                                                  "counts": rle["counts"].decode()}})
                    ann_id += 1
                img_id += 1

        save_video_cache(VideoCache(vid, fps, ORIG, WORK, frames), out / "cache" / f"{vid}.pkl")
        manifest.append({"video_id": vid, "split": "dev", "has_rip": int(bool(centres)),
                         "fps": fps, "n_frames": n_frames})

    gt_path = out / "gt.json"
    gt_path.write_text(json.dumps({"images": images, "annotations": anns,
                                   "categories": [{"id": 1, "name": "rip_currents"}]}))
    man_path = out / "manifest.csv"
    keys = list(manifest[0])
    man_path.write_text("\n".join([",".join(keys)] +
                                  [",".join(str(m[k]) for k in keys) for m in manifest]) + "\n")
    return {"cache_dir": out / "cache", "gt_json": gt_path, "manifest": man_path}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    paths = make_dataset(args.out_dir, args.seed)
    print({k: str(v) for k, v in paths.items()})
