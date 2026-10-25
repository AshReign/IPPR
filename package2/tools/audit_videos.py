"""CPU-only provenance, full decoding, timestamps, and sampled media alignment audit.

Reads only P1 metadata/image references, original videos and sampled JPEG images.
No segmentation labels are read and no model is loaded. Stored frame_idx is the
zero-based sequential decoder index; timestamp is presentation time relative to
the first decoded frame. Pixel comparisons are evidence, not proof of labels.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_entries(value):
    if isinstance(value, dict):
        if "path" in value and "size_bytes" in value:
            yield value
        for child in value.values():
            yield from source_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from source_entries(child)


def metrics(image, decoded):
    import numpy as np
    if image.shape != decoded.shape:
        return {"same_shape": False, "image_shape": list(image.shape), "decoded_shape": list(decoded.shape)}
    # Work in row chunks to bound temporary memory even for 4K frames.
    sum_abs = sum_sq = count = exact = max_abs = 0
    for y in range(0, image.shape[0], 128):
        diff = image[y:y + 128].astype(np.int16) - decoded[y:y + 128].astype(np.int16)
        absolute = np.abs(diff)
        sum_abs += int(absolute.sum(dtype=np.int64))
        sum_sq += float(np.square(diff.astype(np.float64)).sum())
        exact += int(np.count_nonzero(diff == 0))
        count += diff.size
        max_abs = max(max_abs, int(absolute.max()))
    mse = sum_sq / count
    return {"same_shape": True, "mae_0_255": sum_abs / count, "rmse_0_255": math.sqrt(mse),
            "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
            "exact_channel_fraction": exact / count, "max_absolute_channel_difference": max_abs,
            "exact_pixels": exact == count}


def audit_one(job):
    # Set before imports to avoid oversubscribing the CPU while training runs.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import av
    import cv2
    cv2.setNumThreads(1)
    begin = time.monotonic()
    video_id = job["row"]["video_id"]
    row = job["row"]
    source = Path(job["source"])
    out = Path(job["out"])
    progress_path = out / f"video_{video_id}_progress.json"
    result_path = out / f"video_{video_id}.json"
    expected = job["source_entry"]
    report = {"schema_version": 1, "video_id": video_id, "split": row["split"], "source_split": row["source_split"],
              "status": "RUNNING", "started_at_utc": utc(), "worker_pid": os.getpid(),
              "source": str(source), "source_url": expected.get("pinned_download_url"),
              "dataset_revision": job["revision"], "manifest_sha256": job["manifest_sha256"],
              "p1_metadata": row, "errors": [], "warnings": [],
              "frame_index_convention": "zero-based sequential presentation-order decoder index",
              "timestamp_convention": "frame.pts * frame.time_base minus first decoded presentation time",
              "does_not_read_segmentation_labels_or_run_models": True}
    def progress(phase, count=0):
        atomic_json(progress_path, {"video_id": video_id, "phase": phase, "decoded_frames": count,
                    "expected_frames": int(float(row["n_frames"])), "pid": os.getpid(), "updated_at_utc": utc(),
                    "elapsed_seconds": time.monotonic() - begin})
    try:
        progress("hashing")
        report["size_bytes"] = source.stat().st_size
        report["sha256"] = sha256(source)
        report["expected_size_bytes"] = expected["size_bytes"]
        report["expected_lfs_sha256"] = expected.get("lfs_sha256")
        report["source_integrity_pass"] = (report["size_bytes"] == expected["size_bytes"] and
                                           report["sha256"] == expected.get("lfs_sha256"))
        if not report["source_integrity_pass"]:
            raise ValueError("Original video size/SHA256 differs from pinned official manifest")
        annotated = sorted(job["annotated"], key=lambda x: int(x["frame_idx"]))
        ann_by_index = {int(x["frame_idx"]): x for x in annotated}
        if len(ann_by_index) != len(annotated):
            report["errors"].append("Duplicate annotated frame indices in P1 frames_index.csv")
        sample_rows = [annotated[i] for i in sorted({0, len(annotated) // 2, len(annotated) - 1})]
        sample_images = {}
        candidates_by_idx = {}
        alignment = {}
        for sample in sample_rows:
            idx = int(sample["frame_idx"])
            img_path = Path(job["dataset"]) / row["source_split"] / "extracted/sampled_images/images" / Path(sample["image_rel"]).name
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot decode annotated JPEG {img_path}")
            sample_images[idx] = image
            alignment[idx] = {"frame_idx": idx, "image_rel": sample["image_rel"], "image_path": str(img_path),
                              "image_sha256": sha256(img_path), "p1_t_sec": float(sample["t_sec"]), "comparisons": {}}
            for candidate in range(max(0, idx - 1), idx + 2):
                candidates_by_idx.setdefault(candidate, []).append(idx)
        timestamps_path = out / f"video_{video_id}_timestamps.csv.gz"
        ann_timing = []
        first_abs = last_abs = last_timestamp = None
        n_frames = missing_pts = nonmonotonic = changing_dimensions = 0
        duration_min = None
        duration_max = 0.0
        with av.open(str(source)) as container, gzip.open(timestamps_path, "wt", encoding="utf-8", newline="") as handle:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            report["stream"] = {"codec": stream.codec_context.name, "width": stream.width, "height": stream.height,
                                "average_rate": str(stream.average_rate), "base_rate": str(stream.base_rate),
                                "guessed_rate": str(stream.guessed_rate), "time_base": str(stream.time_base),
                                "metadata_frames": stream.frames, "duration_ticks": stream.duration,
                                "start_time_ticks": stream.start_time, "decoder_threads": 1}
            report["fps"] = float(stream.average_rate) if stream.average_rate else None
            writer = csv.writer(handle)
            writer.writerow(["frame_idx", "pts", "time_base", "absolute_pts_sec", "timestamp_sec", "p1_annotated", "p1_t_sec", "delta_to_p1_sec"])
            progress("decoding")
            for idx, frame in enumerate(container.decode(stream)):
                n_frames = idx + 1
                if (frame.width, frame.height) != (stream.width, stream.height):
                    changing_dimensions += 1
                if frame.pts is None:
                    missing_pts += 1
                    writer.writerow([idx, "", str(frame.time_base), "", "", int(idx in ann_by_index), "", ""])
                    continue
                abs_pts = frame.pts * frame.time_base
                if first_abs is None:
                    first_abs = abs_pts
                timestamp = float(abs_pts - first_abs)
                if last_abs is not None:
                    delta = float(abs_pts - last_abs)
                    nonmonotonic += int(delta <= 0)
                    duration_min = delta if duration_min is None else min(duration_min, delta)
                    duration_max = max(duration_max, delta)
                last_abs, last_timestamp = abs_pts, timestamp
                p1 = ann_by_index.get(idx)
                p1_time = float(p1["t_sec"]) if p1 else None
                delta_p1 = timestamp - p1_time if p1 else None
                if p1:
                    ann_timing.append({"frame_idx": idx, "timestamp_sec": timestamp, "p1_t_sec": p1_time,
                                       "delta_seconds": delta_p1})
                writer.writerow([idx, frame.pts, str(frame.time_base), float(abs_pts), timestamp,
                                 int(p1 is not None), p1_time, delta_p1])
                if idx in candidates_by_idx:
                    decoded = frame.to_ndarray(format="bgr24")
                    for anchor in candidates_by_idx[idx]:
                        alignment[anchor]["comparisons"][str(idx - anchor)] = metrics(sample_images[anchor], decoded)
                if n_frames % 100 == 0:
                    progress("decoding", n_frames)
        report["decoded_frames"] = n_frames
        report["timestamps_file"] = str(timestamps_path)
        report["timestamps_sha256"] = sha256(timestamps_path)
        report["missing_pts_frames"] = missing_pts
        report["nonincreasing_pts_transitions"] = nonmonotonic
        report["changing_dimensions_frames"] = changing_dimensions
        report["first_absolute_pts_seconds"] = float(first_abs) if first_abs is not None else None
        report["last_relative_timestamp_seconds"] = last_timestamp
        report["frame_interval_seconds"] = {"minimum": duration_min, "maximum": duration_max,
                                             "constant_within_1ns": duration_min is not None and duration_max - duration_min < 1e-9}
        timing_tolerance = max(float(stream.time_base), 1e-6)
        max_timing_difference = max((abs(x["delta_seconds"]) for x in ann_timing), default=None)
        report["annotated_time_alignment"] = {"annotated_frames_expected": len(annotated),
                    "annotated_frames_decoded": len(ann_timing), "tolerance_seconds_one_stream_tick": timing_tolerance,
                    "max_absolute_delta_seconds": max_timing_difference,
                    "pass": len(ann_timing) == len(annotated) and max_timing_difference is not None and max_timing_difference <= timing_tolerance,
                    "records": ann_timing}
        if n_frames != int(float(row["n_frames"])):
            report["errors"].append("Decoded frame count differs from frozen P1 metadata")
        if (report["stream"]["width"], report["stream"]["height"]) != (int(float(row["width"])), int(float(row["height"]))):
            report["errors"].append("Decoded dimensions differ from P1 metadata")
        if report["fps"] is None or not math.isclose(report["fps"], float(row["fps"]), rel_tol=1e-9, abs_tol=1e-8):
            report["errors"].append("PyAV average FPS differs from P1 metadata")
        if missing_pts or nonmonotonic or changing_dimensions:
            report["errors"].append("Missing/nonincreasing PTS or changing video dimensions")
        if not report["annotated_time_alignment"]["pass"]:
            report["errors"].append("P1 annotated t_sec differs from actual PTS-relative timeline beyond one stream tick")
        for idx, item in alignment.items():
            valid = [(int(delta), entry["mae_0_255"]) for delta, entry in item["comparisons"].items() if entry["same_shape"]]
            valid.sort(key=lambda x: (x[1], abs(x[0]), x[0]))
            correct = item["comparisons"].get("0", {})
            item["lowest_mae_offset"] = valid[0][0] if valid else None
            adjacent = [value for delta, value in valid if delta != 0]
            if not correct.get("same_shape"):
                item["status"] = "FAIL_MISSING_OR_WRONG_DIMENSIONS"
            elif correct["exact_pixels"]:
                item["status"] = "EXACT_PIXEL_MATCH_AT_ZERO_BASED_INDEX"
            elif valid[0][0] == 0 and adjacent and min(adjacent) - correct["mae_0_255"] > 0.05:
                item["status"] = "ZERO_BASED_INDEX_BEST_LOCAL_MATCH_LOSSY_IMAGE"
            elif valid[0][0] != 0 and correct["mae_0_255"] - valid[0][1] > 0.05:
                item["status"] = "REVIEW_ADJACENT_FRAME_MATCHES_BETTER"
            else:
                item["status"] = "AMBIGUOUS_ADJACENT_OR_STATIC_FRAMES"
            item["interpretation"] = "JPEG compression and decoder color conversion prevent general exact-pixel proof; only local +/-1 alternatives were compared."
            if item["status"].startswith("FAIL"):
                report["errors"].append(f"Sample frame {idx}: {item['status']}")
            if item["status"].startswith(("REVIEW", "AMBIGUOUS")):
                report["warnings"].append(f"Sample frame {idx}: {item['status']}")
        report["sampled_image_alignment"] = list(alignment.values())
        report["sampled_image_scope"] = "First, middle and last annotated image per video; no general whole-video image/label alignment claim"
        report["status"] = "FAIL" if report["errors"] else ("PASS_WITH_ALIGNMENT_REVIEW" if report["warnings"] else "PASS")
    except Exception as error:
        report["status"] = "FAIL"
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["traceback"] = traceback.format_exc()
    report["completed_at_utc"] = utc()
    report["elapsed_seconds"] = time.monotonic() - begin
    atomic_json(result_path, report)
    progress(report["status"], report.get("decoded_frames", 0))
    return {"video_id": video_id, "split": row["split"], "status": report["status"], "report": str(result_path),
            "decoded_frames": report.get("decoded_frames"), "elapsed_seconds": report["elapsed_seconds"],
            "errors": report["errors"], "warnings": report["warnings"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--only", nargs="+", help="Optional subset for smoke verification")
    args = parser.parse_args()
    project = args.project.resolve()
    manifest_dir = project / "package2/upstream/data/manifests"
    out = project / "package2/reports/video_audit"
    out.mkdir(parents=True, exist_ok=True)
    frozen_ids = {split: (manifest_dir / f"{split}_videos.txt").read_text().splitlines() for split in ("dev", "val")}
    if len(frozen_ids["dev"]) != 17 or len(frozen_ids["val"]) != 30 or set(frozen_ids["dev"]) & set(frozen_ids["val"]):
        raise ValueError("Frozen split count/overlap check failed")
    rows = [r for r in read_csv(manifest_dir / "videos.csv") if r["split"] in frozen_ids]
    if {r["video_id"] for r in rows} != set(frozen_ids["dev"] + frozen_ids["val"]):
        raise ValueError("videos.csv and frozen video lists differ")
    # Explicitly retain only metadata needed for media/timing checks, not n_gt.
    indexed = read_csv(manifest_dir / "frames_index.csv")
    frame_rows = [{key: r[key] for key in ("video_id", "frame_idx", "t_sec", "image_rel")} for r in indexed]
    upstream = json.loads((project / "data/ripvis_manifest.json").read_text())
    sources = {entry["path"]: entry for entry in source_entries(upstream)}
    manifests = {name: sha256(manifest_dir / name) for name in ("videos.csv", "frames_index.csv", "dev_videos.txt", "val_videos.txt")}
    if args.only:
        missing = set(args.only) - {r["video_id"] for r in rows}
        if missing:
            raise ValueError(f"Not in frozen scope: {sorted(missing)}")
        rows = [r for r in rows if r["video_id"] in args.only]
    jobs = []
    for row in rows:
        if row["video_id"] not in frozen_ids[row["split"]]:
            raise ValueError("Per-video split disagrees with frozen list")
        relative = f"{row['source_split']}/videos/{row['video_id']}.mp4"
        jobs.append({"row": row, "source": str(project / "data/ripvis" / relative),
                     "source_entry": sources[relative], "out": str(out), "dataset": str(project / "data/ripvis"),
                     "revision": upstream["sha"], "manifest_sha256": manifests,
                     "annotated": [r for r in frame_rows if r["video_id"] == row["video_id"]]})
    progress_path = project / "package2/reports/video_audit_progress.json"
    results = []
    state = {"schema_version": 1, "status": "RUNNING", "started_at_utc": utc(), "pid": os.getpid(),
             "command": sys.argv, "workers": args.workers, "full_frozen_scope": not args.only,
             "total_videos": len(jobs), "manifest_sha256": manifests, "results": results}
    atomic_json(progress_path, state)
    print(json.dumps({"audit_started": utc(), "pid": os.getpid(), "videos": len(jobs), "workers": args.workers}), flush=True)
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(audit_one, job): job["row"]["video_id"] for job in jobs}
        while pending:
            done, _ = futures.wait(pending, timeout=3, return_when=futures.FIRST_COMPLETED)
            for task in done:
                name = pending.pop(task)
                try:
                    result = task.result()
                except Exception as error:
                    result = {"video_id": name, "status": "FAIL", "errors": [repr(error)]}
                results.append(result)
                print(json.dumps({"completed": len(results), "total": len(jobs), **result}), flush=True)
            state["updated_at_utc"] = utc()
            state["completed_videos"] = len(results)
            atomic_json(progress_path, state)
    state["status"] = "FAIL" if any(r["status"] == "FAIL" for r in results) else "COMPLETE"
    state["completed_at_utc"] = utc()
    state["sum_decoded_frames"] = sum(r.get("decoded_frames") or 0 for r in results)
    atomic_json(progress_path, state)
    atomic_json(project / "package2/reports/video_audit_summary.json", state)
    return 1 if state["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
