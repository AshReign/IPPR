"""Agreement check (Package 5, step 4): our evaluator vs the official compute_pr_f1_f2.py.

Usage (from the repo root, with the virtual environment active):
    python -m scripts.check_agreement --cache-dir <P2 delivery folder>
    python -m scripts.check_agreement --cache-dir <P2 delivery folder> --restore-check

For each confidence threshold, builds T0-style predictions from a P2 cache (keep conf >= thr,
mask = soft > 0.5, cache order, encode_mask to the original resolution, score = 1.0, annotated
frames only), then scores them with both evaluators. P, R, F1 and F2 must be identical.

--restore-check also measures what the shared resize rule (encode_mask) costs against P2's
exact inverse (restore_work_masks), which only works on raw model output and so cannot be
used for T1/T2: per-instance IoU between the two masks, and T0 scores under each.

Writes the log to report/logs/ (committed) and the prediction files to outputs/agreement/.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from eval.evaluator import encode_mask, evaluate, load_gt, summarise, write_gt_subset

REPO = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.1, 0.25, 0.5]


def open_caches(cache_dir: Path):
    """video_id -> VideoCacheReader for every .sqlite file in a P2 delivery."""
    sys.path.insert(0, str(cache_dir / "reader"))
    from p2.cache_io import VideoCacheReader
    readers = {}
    for db in sorted(cache_dir.glob("*.sqlite")):
        r = VideoCacheReader(db)
        if not r.complete:
            raise ValueError(f"{db.name} is not marked complete")
        readers[r.metadata["source_identity"]["video_id"]] = r
    return readers


def annotated_frames(gt, readers):
    """Yield (image_id, cache frame) for every annotated frame, joined on (video_id, frame_idx)."""
    for row in gt.frames.itertuples():
        r = readers[row.video_id]
        if tuple(r.metadata["original_shape_hw"]) != gt.shapes[row.image_id]:
            raise ValueError(f"{row.video_id}: cache and ground-truth image sizes differ")
        yield row.image_id, r, r.read(row.frame_idx)


def t0_predictions(gt, readers, thr: float, restore: bool = False):
    preds = []
    for img, r, f in annotated_frames(gt, readers):
        keep = f["conf"] >= np.float32(thr)
        if not keep.any():
            continue
        h, w = gt.shapes[img]
        masks = f["soft_mask"][keep]
        if restore:
            from p2.soft_masks import restore_work_masks
            masks = restore_work_masks(masks, (h, w), f["boxes"][keep],
                                       transform=r.metadata["transform"]).numpy()
        preds += [{"image_id": int(img), "category_id": 1, "score": 1.0,
                   "segmentation": encode_mask(m, h, w)} for m in masks]
    return preds


def load_official(ripvis_root: Path):
    spec = importlib.util.spec_from_file_location("official", ripvis_root / "compute_pr_f1_f2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restore_iou(gt, readers):
    """Per-instance IoU between encode_mask(soft) and P2's exact restore, both > 0.5."""
    from p2.soft_masks import restore_work_masks
    from pycocotools import mask as mask_utils
    ious, fallbacks = [], 0
    for img, r, f in annotated_frames(gt, readers):
        if not len(f["conf"]):
            continue
        h, w = gt.shapes[img]
        exact, details = restore_work_masks(f["soft_mask"], (h, w), f["boxes"],
                                            transform=r.metadata["transform"], return_diagnostics=True)
        fallbacks += sum(d["status"] == "fallback" for d in details)
        for soft, ref in zip(f["soft_mask"], exact.numpy()):
            a, b = encode_mask(soft, h, w), encode_mask(ref, h, w)
            if mask_utils.area(a) or mask_utils.area(b):
                ious.append(float(mask_utils.iou([a], [b], [0])[0][0]))
    ious = np.array(ious)
    return {"n_instances": len(ious), "n_fallback": fallbacks,
            "iou_min": float(ious.min()), "iou_p5": float(np.percentile(ious, 5)),
            "iou_median": float(np.median(ious)), "share_iou_ge_0.98": float((ious >= 0.98).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, help="extracted P2 delivery (has reader/ and *.sqlite)")
    ap.add_argument("--restore-check", action="store_true")
    ap.add_argument("--log", default=str(REPO / "report" / "logs" / "agreement_pilot.json"))
    args = ap.parse_args()

    from configs import paths
    ripvis_root = Path(paths()["ripvis_root"])
    cache_dir = Path(args.cache_dir)
    out = REPO / "outputs" / "agreement"
    out.mkdir(parents=True, exist_ok=True)

    readers = open_caches(cache_dir)
    gt = load_gt("dev", ripvis_root, videos=readers)
    write_gt_subset(gt, out / "gt.json")
    official = load_official(ripvis_root)

    log = {"cache_dir": cache_dir.name, "videos": sorted(readers),
           "n_frames": len(gt.frames), "n_gt": int(gt.frames.n_gt.sum()), "runs": []}
    for thr in THRESHOLDS:
        preds = t0_predictions(gt, readers, thr)
        path = out / f"t0_conf{thr}.json"
        path.write_text(json.dumps(preds))
        t = time.perf_counter()
        ours = summarise(evaluate(preds, gt))
        t_ours = time.perf_counter() - t
        run = {"conf_thr": thr, "n_pred": len(preds), "ours": ours, "seconds_ours": round(t_ours, 3)}
        if preds:                     # the official script crashes on an empty prediction list
            t = time.perf_counter()
            theirs = official.compute_pr_f1_f2(str(out / "gt.json"), str(path), 0.5, 0.0)
            run.update(official=theirs, seconds_official=round(time.perf_counter() - t, 3),
                       identical=all(ours[k] == theirs[k] for k in ("precision", "recall", "f1", "f2")))
        log["runs"].append(run)
        print(f"conf>={thr}: {len(preds)} preds, F2 ours={ours['f2']:.6f}",
              f"official={run['official']['f2']:.6f} identical={run['identical']}" if preds else "(no preds)")

    if args.restore_check:
        log["restore_check"] = restore_iou(gt, readers)
        preds = t0_predictions(gt, readers, THRESHOLDS[0], restore=True)
        log["restore_check"]["t0_conf0.1_exact_restore"] = summarise(evaluate(preds, gt))
        print("restore check:", json.dumps(log["restore_check"], indent=2))

    for r in readers.values():
        r.close()
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text(json.dumps(log, indent=2))
    print("log written to", args.log)


if __name__ == "__main__":
    main()
