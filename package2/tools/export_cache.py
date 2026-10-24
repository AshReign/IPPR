"""Stream full videos into one resumable SQLite P2 cache per video.

Pilot example (run only when the GPU is available):
  python package2/tools/export_cache.py --checkpoint PATH.pt --kind pilot \
      --output package2/outputs/pilot_cache --videos RipVIS-071 RipVIS-140 RipVIS-NR-012 \
      --pilot-validation package2/reports/pilot_mask_validation_01/summary.json

Final requires both --frozen-config and --cache-validation. Any final-val video
additionally requires --dev-cache-validation covering all 17 actual dev caches.
See validate_final_gates for the artifact contract. A pilot cache is always temporary and never frozen.
No final-output confidence/area filtering is applied to raw cache candidates.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "package2"))
INDEX_SCHEMA = "p2-cache-export-index-v1"
RUNTIME_KEYS = ("device_type", "device_name", "torch_version", "torch_cuda_version",
                "cudnn_version", "ultralytics_version", "inference_dtype", "batch",
                "matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_deterministic", "cudnn_benchmark")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, value):
    """Publish a progress snapshot atomically; an interrupted write is not progress."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_path(value, root=ROOT):
    """Accept native, C:/... or /mnt/c/... arguments on either local runtime."""
    text = str(value).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", text):
        text = "/mnt/" + text[0].lower() + text[2:]
    elif os.name == "nt" and re.match(r"^/mnt/[a-z]/", text):
        text = text[5].upper() + ":/" + text[7:]
    path = Path(text)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_final_gates(kind, checkpoint_hash, p1_commit, prediction_config,
                         frozen_path=None, validation_path=None):
    """Gate final export on verified, matching, immutable experiment artifacts.

    Frozen JSON: status=frozen, checkpoint_sha256, p1_commit, inference (exact
    PREDICT_DEFAULTS), dev_selection_report_path (within package2), its SHA256,
    and t0.confidence on the 17-point
    PDF grid. The producer must only create this after real P5 dev selection.
    Validation JSON: status=PASS, checkpoint_sha256, soft_mask_module_sha256,
    prediction_config_sha256, annotated_dev_frames>=50, all_instances_pass=true,
    min_instance_iou>=.98, cache_instance_count>0, and explicit passing >=200-frame
    nonempty native-mask stats with per-instance IoU>=.99. Produced by the actual
    200/50-frame validator. production_validation_eligible must be true and its
    recorded runtime must be CUDA; actual runtime equality is checked after
    production model setup/first prediction, before writing any cache row.
    """
    if kind == "pilot":
        require(not frozen_path and not validation_path, "Pilot must not present itself as a frozen final export")
        return {"status": "temporary_unfrozen", "frozen": False}
    require(frozen_path and validation_path, "Final export requires --frozen-config and --cache-validation")
    frozen = load_json(frozen_path)
    validation = load_json(validation_path)
    require(frozen.get("status") == "frozen", "Final model/T0 artifact is not frozen")
    require(frozen.get("checkpoint_sha256") == checkpoint_hash, "Frozen checkpoint hash differs")
    require(frozen.get("p1_commit") == p1_commit, "Frozen experiment references different P1 source")
    require(frozen.get("inference") == prediction_config["inference"], "Frozen inference configuration differs")
    selection_hash = frozen.get("dev_selection_report_sha256", "")
    require(isinstance(selection_hash, str) and re.fullmatch(r"[a-fA-F0-9]{64}", selection_hash),
            "Frozen artifact must identify the completed P5 dev selection report")
    selection_path_value = frozen.get("dev_selection_report_path")
    require(isinstance(selection_path_value, str) and bool(selection_path_value),
            "Frozen artifact must reference an existing P5 selection report")
    selection_path = local_path(selection_path_value, root=ROOT / "package2")
    try:
        selection_path.relative_to(ROOT / "package2")
    except ValueError as error:
        raise ValueError("P5 selection report must remain within package2") from error
    require(selection_path.is_file() and sha256(selection_path) == selection_hash,
            "P5 selection report is absent or its actual SHA256 differs")
    selection = load_json(selection_path)
    require(selection.get("status") == "selected_from_P5_dev_scores_not_yet_frozen",
            "Referenced selection report is not a completed P5 dev selection")
    epochs = selection.get("completed_epochs_scored")
    require(type(epochs) is int and 1 <= epochs <= 30
            and selection.get("scoring_combinations") == epochs * 17,
            "Selection report does not cover every completed epoch and all 17 thresholds")
    for field in ("score_matrix_sha256", "checkpoint_inventory_sha256", "selection_protocol_sha256"):
        require(isinstance(selection.get(field), str) and re.fullmatch(r"[a-fA-F0-9]{64}", selection[field]),
                f"Selection report lacks valid provenance: {field}")
    threshold = frozen.get("t0", {}).get("confidence")
    require(isinstance(threshold, (float, int)) and math.isfinite(threshold)
            and any(abs(float(threshold) - (0.1 + step * 0.05)) < 1e-9 for step in range(17)),
            "Frozen T0 confidence is outside the PDF's 0.10..0.90 step-0.05 grid")
    winner = selection.get("winner", {})
    require(winner.get("checkpoint_sha256") == checkpoint_hash, "Frozen checkpoint is not the P5-selected winner")
    try:
        selected_threshold = float(winner["threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("P5 selection winner has no valid threshold") from error
    require(math.isfinite(selected_threshold) and abs(selected_threshold - threshold) < 1e-9,
            "Frozen T0 threshold is not the P5-selected winner")
    require(validation.get("status") == "PASS", "Mask/cached-mask validation has not passed")
    require(validation.get("production_validation_eligible") is True,
            "CPU/diagnostic validation cannot authorize production GPU caches")
    validation_runtime = validation.get("validation_runtime")
    validate_production_runtime(validation_runtime, validation_runtime)
    native = validation.get("stats", {})
    require(validation.get("softmask_gate_pass") is True and int(native.get("frames", 0)) >= 200,
            "Native soft-mask gate has not passed on at least 200 frames")
    require(int(native.get("instances", 0)) > 0 and native.get("softmask_failed_instances") == 0,
            "Native soft-mask validation is empty or has failing instances")
    native_minimum = native.get("softmask_min_iou")
    require(isinstance(native_minimum, (float, int)) and math.isfinite(native_minimum) and native_minimum >= .99,
            "Native soft-mask per-instance minimum IoU is below .99")
    require(validation.get("cache_gate_pass") is True, "The saved/reloaded 50-frame cache gate has not passed")
    require(validation.get("checkpoint_sha256") == checkpoint_hash, "Validation used a different checkpoint")
    require(validation.get("soft_mask_module_sha256") == prediction_config["soft_mask_module_sha256"],
            "Validation used different soft-mask code")
    config_hash = hashlib.sha256(canonical(prediction_config).encode()).hexdigest()
    require(validation.get("prediction_config_sha256") == config_hash,
            "Validation used different preprocessing/NMS/encoding rules")
    require(int(validation.get("annotated_dev_frames", 0)) >= 50, "Fewer than 50 dev annotated frames validated")
    require(validation.get("all_instances_pass") is True, "Not every cached-mask instance passed")
    require(int(validation.get("cache_instance_count", 0)) > 0, "Cache validation contains no instances")
    minimum = validation.get("min_instance_iou")
    require(isinstance(minimum, (float, int)) and math.isfinite(minimum) and minimum >= 0.98,
            "Cache restoration IoU is below 0.98")
    return {"status": "frozen", "frozen": True, "frozen_config_sha256": sha256(frozen_path),
            "cache_validation_sha256": sha256(validation_path), "t0_confidence": float(threshold),
            "dev_selection_report_sha256": selection_hash, "validation_runtime": validation_runtime,
            "production_validation_eligible": True}


def validate_production_runtime(actual, validated):
    """Fail closed for CPU diagnostics or backend/precision/runtime mismatches."""
    require(isinstance(actual, dict) and isinstance(validated, dict), "Missing production validation runtime")
    require(all(key in actual and key in validated for key in RUNTIME_KEYS), "Incomplete runtime signature")
    require(actual["device_type"] == validated["device_type"] == "cuda",
            "Final cache production requires the GPU validation runtime, not CPU diagnostics")
    require(actual["inference_dtype"] == validated["inference_dtype"] == "float32"
            and actual["batch"] == validated["batch"] == 1, "Production precision/batch differs")
    require(isinstance(actual["device_name"], str) and bool(actual["device_name"]), "Missing GPU identity")
    differences = [key for key in RUNTIME_KEYS if actual[key] != validated[key]]
    require(not differences, f"Actual production runtime differs from mask validation: {differences}")


def runtime_after_prediction(result):
    """Capture actual backend after predictor setup, which may change torch flags."""
    import torch
    import ultralytics

    device = result.boxes.data.device
    return {"device_type": device.type,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "ultralytics_version": ultralytics.__version__,
            "inference_dtype": "float32", "batch": 1,
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}


def configure_execution_device(device, kind, cpu_threads):
    """Bound temporary CPU work without Ultralytics resetting its thread budget."""
    if str(device) != "cpu":
        return device
    require(kind == "pilot", "CPU export is temporary pilot-only; final production requires validated CUDA")
    require(type(cpu_threads) is int and cpu_threads in (1, 2), "CPU pilot thread budget must be 1 or 2")
    import cv2
    import torch

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    # A torch.device bypasses select_device's CPU branch, which changes OMP threads.
    return torch.device("cpu")


def cpu_runtime_after_prediction(result, cpu_threads):
    import cv2
    import torch

    actual = runtime_after_prediction(result)
    require(actual["device_type"] == "cpu", "Temporary CPU export unexpectedly used a different device")
    require(torch.get_num_threads() == cpu_threads and torch.get_num_interop_threads() == 1
            and cv2.getNumThreads() == 1, "CPU backend changed the bounded thread budget")
    actual.update({"cpu_threads": torch.get_num_threads(),
                   "cpu_interop_threads": torch.get_num_interop_threads(), "opencv_threads": cv2.getNumThreads()})
    return actual


def validate_final_val_cache_gate(report_path, run_identity):
    """Authorize final-val only after the actual complete dev caches passed.

    Authorization is returned separately: putting its hash into run_identity
    would change the identity between dev production and val production and
    create a circular dependency on the caches being validated.
    """
    from p2.cache_io import VideoCacheReader

    require(run_identity.get("kind") == "final" and run_identity.get("gates", {}).get("frozen") is True,
            "Actual-dev-cache validation can only authorize a frozen final export")
    require(report_path, "Final-val export requires --dev-cache-validation")
    report_path = local_path(report_path)
    try:
        report_path.relative_to(ROOT / "package2")
    except ValueError as error:
        raise ValueError("Actual-dev-cache validation report must remain within package2") from error
    report = load_json(report_path)
    require(report.get("schema") == "p2-exported-dev-cache-validation-v1", "Wrong actual-dev-cache report schema")
    require(report.get("status") == "PASS" and report.get("production_validation_eligible") is True,
            "Actual-dev-cache GPU validation has not passed")
    require(report.get("complete_dev_scope") is True, "Actual-dev-cache report does not cover the complete dev scope")
    require(type(report.get("annotated_dev_frames")) is int and report["annotated_dev_frames"] >= 50,
            "Actual-dev-cache validation requires at least 50 annotated dev frames")
    require(type(report.get("instance_count")) is int and report["instance_count"] > 0,
            "Actual-dev-cache validation has no instances")
    require(type(report.get("native_nonempty_instance_count")) is int
            and 0 < report["native_nonempty_instance_count"] <= report["instance_count"],
            "Actual-dev-cache validation has no nonempty native reference masks")
    require(type(report.get("failed_instances")) is int and report["failed_instances"] == 0
            and report.get("all_instances_pass") is True,
            "Some actual-dev-cache instances failed")
    minimum = report.get("min_instance_iou")
    require(isinstance(minimum, (float, int)) and not isinstance(minimum, bool)
            and math.isfinite(minimum) and minimum >= .98, "Actual-dev-cache minimum IoU is below .98")
    require(report.get("checkpoint_sha256") == run_identity["checkpoint_sha256"],
            "Actual-dev-cache validation used different weights")
    require(report.get("prediction_config_sha256") == run_identity["prediction_config_sha256"],
            "Actual-dev-cache validation used different inference configuration")
    require(report.get("soft_mask_module_sha256") == run_identity["prediction_config"]["soft_mask_module_sha256"],
            "Actual-dev-cache validation used different mask code")
    identity_hash = hashlib.sha256(canonical(run_identity).encode()).hexdigest()
    require(report.get("cache_run_identity_sha256") == identity_hash, "Actual-dev-cache run identity differs")
    validate_production_runtime(report.get("validation_runtime"), run_identity["gates"]["validation_runtime"])

    manifests = ROOT / "package2/upstream/data/manifests"
    require(sha256(manifests / "dev_videos.txt") == run_identity["frozen_manifest_sha256"]["dev_videos.txt"],
            "Frozen dev membership changed after production")
    dev_ids = set((manifests / "dev_videos.txt").read_text().split())
    require(len(dev_ids) == 17, "Expected exactly 17 frozen dev videos")
    with (manifests / "videos.csv").open(encoding="utf-8-sig", newline="") as handle:
        videos = {row["video_id"]: row for row in csv.DictReader(handle)}
    files = report.get("cache_files")
    require(isinstance(files, list) and len(files) == 17, "Actual-dev-cache report must list exactly 17 cache files")
    require(all(isinstance(entry, dict) for entry in files), "Invalid dev cache file record")
    require({entry.get("video_id") for entry in files} == dev_ids, "Report cache IDs differ from the frozen dev set")
    verified_files = []
    total_frames = 0
    for entry in files:
        video_id = entry["video_id"]
        row = videos[video_id]
        require(row["split"] == "dev", f"Report contains a non-dev video: {video_id}")
        require(isinstance(entry.get("path"), str) and bool(entry["path"]), f"Missing cache path: {video_id}")
        path = local_path(entry["path"], root=ROOT / "package2")
        try:
            path.relative_to(ROOT / "package2")
        except ValueError as error:
            raise ValueError(f"Validated cache path escapes package2: {video_id}") from error
        require(path.is_file() and sha256(path) == entry.get("sha256"), f"Validated cache SHA256 changed: {video_id}")
        audit = load_json(ROOT / "package2/reports/video_audit" / f"video_{video_id}.json")
        require(audit.get("source_integrity_pass") is True, f"Source integrity audit failed: {video_id}")
        original = ROOT / "data/ripvis" / row["source_split"] / "videos" / f"{video_id}.mp4"
        require(original.stat().st_size == audit["size_bytes"] and sha256(original) == audit["sha256"],
                f"Actual source video changed since dev-cache validation: {video_id}")
        reader = VideoCacheReader(path)
        try:
            require(reader.complete, f"Validated dev cache is incomplete: {video_id}")
            require(reader.metadata.get("run_identity") == run_identity, f"Validated dev cache identity differs: {video_id}")
            source = reader.metadata.get("source_identity", {})
            require(source.get("video_id") == video_id and source.get("split") == "dev"
                    and source.get("sha256") == audit["sha256"], f"Dev cache source provenance differs: {video_id}")
            validate_production_runtime(reader.metadata.get("inference_runtime"), run_identity["gates"]["validation_runtime"])
            count = len(reader)
            require(type(entry.get("frames")) is int and count == entry["frames"] == audit["decoded_frames"]
                    == source.get("expected_decoded_frames") == int(float(row["n_frames"])),
                    f"Dev cache does not cover every audited frame: {video_id}")
            endpoints = reader.db.execute("SELECT MIN(frame_idx),MAX(frame_idx) FROM frames").fetchone()
            require(endpoints == (0, count - 1), f"Dev cache indices are not a contiguous complete sequence: {video_id}")
            total_frames += count
        finally:
            reader.close()
        verified_files.append({"video_id": video_id, "path": str(path), "sha256": entry["sha256"],
                               "frames": count, "source_sha256": audit["sha256"]})
    require(report["annotated_dev_frames"] <= total_frames, "Report sample count exceeds the complete dev timeline")
    return {"schema": report["schema"], "report_path": str(report_path), "report_sha256": sha256(report_path),
            "status": "PASS", "verified_utc": utc_now(), "cache_run_identity_sha256": identity_hash,
            "annotated_dev_frames": report["annotated_dev_frames"], "instance_count": report["instance_count"],
            "native_nonempty_instance_count": report["native_nonempty_instance_count"],
            "min_instance_iou": minimum, "complete_dev_scope": True, "source_hashes_rechecked": True,
            "dev_cache_frames": total_frames, "cache_files": verified_files,
            "validation_runtime": report["validation_runtime"]}


def load_video_inputs(root, requested, kind, manifest_hashes):
    """Use only P1's exact dev/val IDs and already verified source-video reports."""
    manifests = root / "package2/upstream/data/manifests"
    with (manifests / "videos.csv").open(encoding="utf-8-sig", newline="") as handle:
        videos = {row["video_id"]: row for row in csv.DictReader(handle)}
    require(len(requested) == len(set(requested)), "Repeated video IDs")
    if kind == "pilot":
        require(2 <= len(requested) <= 3, "Choose explicitly 2 or 3 dev videos for the pilot cache")
    tasks = []
    for video_id in requested:
        require(video_id in videos, f"Video absent from P1: {video_id}")
        row = videos[video_id]
        require(row["split"] in (("dev",) if kind == "pilot" else ("dev", "val")),
                f"{kind} may not process {video_id} in split {row['split']}")
        report_path = root / "package2/reports/video_audit" / f"video_{video_id}.json"
        report = load_json(report_path)
        require(report.get("source_integrity_pass") is True, f"Source integrity has not passed: {video_id}")
        allowed_timing_discrepancy = "P1 annotated t_sec differs from actual PTS-relative timeline beyond one stream tick"
        require(all(error == allowed_timing_discrepancy for error in report.get("errors", [])),
                f"Unresolved source/decode failure: {video_id}")
        require(all(report.get(key) == 0 for key in
                    ("missing_pts_frames", "nonincreasing_pts_transitions", "changing_dimensions_frames")),
                f"Source PTS/dimensions audit failed: {video_id}")
        require(int(report["decoded_frames"]) == int(float(row["n_frames"])),
                f"Decoded frame count differs from P1: {video_id}")
        require(report.get("split") == row["split"] and report.get("video_id") == video_id,
                f"Audit identity differs: {video_id}")
        require(report.get("manifest_sha256") == manifest_hashes,
                f"Video audit used different frozen manifests: {video_id}")
        source_rel = f"data/ripvis/{row['source_split']}/videos/{video_id}.mp4"
        source = root / source_rel
        require(source.stat().st_size == report["size_bytes"], f"Source size changed: {video_id}")
        require(sha256(source) == report["sha256"], f"Source SHA256 changed: {video_id}")
        timestamps = report_path.with_name(f"video_{video_id}_timestamps.csv.gz")
        require(sha256(timestamps) == report["timestamps_sha256"], f"PTS audit changed: {video_id}")
        require(int(report["decoded_frames"]) > 0, f"Source has no verified frames: {video_id}")
        tasks.append({"video_id": video_id, "split": row["split"], "source": source,
                      "source_rel": source_rel, "audit": report, "audit_path": report_path,
                      "timestamps": timestamps})
    return tasks


def same_timestamp(actual, expected):
    return math.isfinite(actual) and abs(float(actual) - float(expected)) <= 1e-9


def completion_reverification(previous, *, task, path, run_identity, verification, elapsed):
    """Keep measured original-export values when a completed cache is re-read.

    A read-only verification has no new inference samples and cannot replace the
    original export's FPS/seconds with an unrelated readback timing.
    """
    prior_is_complete = previous.get("status") in ("complete", "complete_reverified")
    result = dict(previous) if prior_is_complete else {}
    result.update({"video_id": task["video_id"], "split": task["split"], "path": str(path),
                   "status": "complete_reverified", "kind": run_identity["kind"],
                   "frozen": run_identity["gates"]["frozen"], "file_bytes": path.stat().st_size,
                   "source_sha256": task["audit"]["sha256"],
                   "checkpoint_sha256": run_identity["checkpoint_sha256"], **verification})
    result["latest_operation"] = "readback_only"
    result["new_frames_this_invocation"] = 0
    result["reverification_count"] = int(previous.get("reverification_count", 0)) + 1
    result["last_reverification"] = {"verified_utc": utc_now(), "elapsed_seconds": elapsed,
                                    "new_frames": 0, "original_export_metrics_preserved": prior_is_complete}
    if not prior_is_complete:
        result["original_export_metrics_available"] = False
    return result


def measured_storage_budget(index, target_frames):
    """Measure this cache and condition any pilot projection on unchanged weights/scenes."""
    frames, size, instances = index["total_frames"], index["total_file_bytes"], index["total_candidate_instances"]
    require(frames > 0 and size >= 0 and instances >= 0 and target_frames > 0,
            "Invalid storage-budget counts")
    budget = {"measured_frames": frames, "measured_sqlite_bytes": size,
              "measured_candidate_instances": instances, "bytes_per_frame": size / frames,
              "bytes_per_candidate_including_empty_frames_and_sqlite_overhead": size / instances if instances else None,
              "size_scope": "SQLite files only; includes frame rows, empty frames, metadata and indexes; excludes videos, checkpoints, reports and temporary SQLite journals",
              "final_storage_size_claim": False, "conditional_projection": None}
    if (index["run_identity"]["kind"] == "pilot" and instances > 0
            and index["run_identity"]["gates"].get("mask_quality_validated") is True):
        projected = math.ceil(size / frames * target_frames)
        budget["conditional_projection"] = {
            "target_frames": target_frames, "projected_sqlite_bytes": projected,
            "projected_sqlite_gib": projected / 2**30,
            "checkpoint_sha256": index["run_identity"]["checkpoint_sha256"],
            "assumption": "Same early checkpoint and comparable scenes, resolutions, candidate counts and probability compression; provisional. Final selected weights and final size will change.",
            "is_final_model_estimate": False}
    else:
        budget["projection_note"] = "No temporary-checkpoint projection from empty or unvalidated pilot caches; final cache sizes are recorded directly."
    return budget


def verify_cache(path, task, expected_metadata=None):
    """Read every frame back with blob checksums, PTS and contiguous-index checks."""
    from p2.cache_io import VideoCacheReader

    reader = VideoCacheReader(path)
    empty = instances = total = 0
    try:
        require(reader.complete, f"Cache is not marked complete: {path}")
        if expected_metadata is not None:
            require(reader.metadata == expected_metadata, f"Final cache metadata differs: {path}")
        require(len(reader) == task["audit"]["decoded_frames"], f"Complete cache frame count differs: {path}")
        with gzip.open(task["timestamps"], "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                require(int(row["frame_idx"]) == total, "PTS audit index is not contiguous")
                frame = reader.read(total)
                require(frame["pts"] == int(row["pts"]), f"Cached PTS differs at frame {total}")
                require(same_timestamp(frame["timestamp"], float(row["timestamp_sec"])),
                        f"Cached timestamp differs at frame {total}")
                count = len(frame["conf"])
                instances += count
                empty += count == 0
                total += 1
        require(total == len(reader), "Cache and PTS audit lengths differ")
        return {"readback_verified_frames": total, "empty_detection_frames": empty,
                "candidate_instances": instances, "complete": True,
                "first_timestamp": reader.read(0)["timestamp"],
                "last_timestamp": reader.read(total - 1)["timestamp"]}
    finally:
        reader.close()


def export_video(task, output, run_identity, inference, model_holder, device, progress_every, resume):
    import av
    import numpy as np
    import torch
    from ultralytics import YOLO
    from p2.cache_io import VideoCacheReader, VideoCacheWriter
    from p2.soft_masks import CacheSoftMaskPredictor

    started = time.perf_counter()
    audit = task["audit"]
    video_id = task["video_id"]
    path = output / f"{video_id}.sqlite"
    progress_path = output / f"{video_id}_progress.json"
    expected_frames = int(audit["decoded_frames"])
    source_identity = {"video_id": video_id, "split": task["split"], "relative_path": task["source_rel"],
                       "sha256": audit["sha256"], "bytes": audit["size_bytes"],
                       "expected_decoded_frames": expected_frames,
                       "timestamps_sha256": audit["timestamps_sha256"],
                       "audit_sha256": sha256(task["audit_path"]),
                       "audit_status": audit["status"], "audit_errors": audit.get("errors", []),
                       "audit_warnings": audit.get("warnings", []),
                       "timestamp_authority": "actual decoded presentation timestamps; P1 t_sec is never substituted"}
    writer = None
    metadata = None
    next_frame = 0
    if path.exists():
        require(resume, f"Cache already exists and resume is disabled: {path}")
        reader = VideoCacheReader(path)
        try:
            require(reader.metadata.get("run_identity") == run_identity, "Resume model/config/implementation differs")
            require(reader.metadata.get("source_identity") == source_identity, "Resume source/audit differs")
            metadata = reader.metadata
            if run_identity["kind"] == "final":
                validate_production_runtime(metadata.get("inference_runtime"), run_identity["gates"]["validation_runtime"])
            was_complete = reader.complete
            next_frame = len(reader)
            if next_frame:
                reader.read(next_frame - 1)  # Verify last committed payload before decoding its source again.
        finally:
            reader.close()
        if was_complete:
            verification = verify_cache(path, task, metadata)
            previous = load_json(progress_path) if progress_path.exists() else {}
            if previous.get("status") not in ("complete", "complete_reverified"):
                index_path = output / "cache_index.json"
                if index_path.exists():
                    previous = load_json(index_path).get("videos", {}).get(video_id, previous)
            result = completion_reverification(previous, task=task, path=path,
                run_identity=run_identity, verification=verification, elapsed=time.perf_counter() - started)
            json_write(progress_path, result)
            return result
        writer = VideoCacheWriter(path, metadata, resume=True)
        require(writer.next_frame == next_frame, "Cache changed while opening for resume")
    resumed_from = next_frame
    inference_seconds = 0.0
    decoded_frames = new_frames = 0
    first_pts = first_time_base = None
    previous_timestamp = None

    def progress(status, error=None):
        # Writer.flush precedes snapshots: completed_frames never counts uncommitted rows.
        if writer is not None:
            writer.flush()
        elapsed = time.perf_counter() - started
        record = {"video_id": video_id, "split": task["split"], "kind": run_identity["kind"],
                  "frozen": run_identity["gates"]["frozen"], "status": status,
                  "updated_utc": utc_now(), "pid": os.getpid(), "path": str(path),
                  "completed_frames": writer.next_frame if writer is not None else 0,
                  "expected_frames": expected_frames, "decoded_frames_this_run": decoded_frames,
                  "new_frames_this_run": new_frames, "resumed_from": resumed_from,
                  "elapsed_seconds": elapsed, "predict_and_transfer_seconds": inference_seconds,
                  "new_frame_export_fps": new_frames / elapsed if elapsed else None,
                  "file_bytes": path.stat().st_size if path.exists() else 0,
                  "last_timestamp": writer.last_timestamp if writer is not None else None}
        if error is not None:
            record["error"] = error
        json_write(progress_path, record)
        print(canonical(record), flush=True)
        return record

    try:
        progress("running")
        with av.open(str(task["source"])) as container, gzip.open(task["timestamps"], "rt", newline="") as pts_handle:
            require(len(container.streams.video) == 1, "Expected one video stream")
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            audited_pts = iter(csv.DictReader(pts_handle))
            for idx, frame in enumerate(container.decode(stream)):
                require(idx < expected_frames, "Decoder yielded more frames than source audit")
                row = next(audited_pts, None)
                require(row is not None and int(row["frame_idx"]) == idx, "PTS audit index/count differs")
                require(frame.pts is not None and frame.time_base is not None, f"Missing PTS on frame {idx}")
                if first_pts is None:
                    first_pts, first_time_base = int(frame.pts), Fraction(frame.time_base)
                require(Fraction(frame.time_base) == first_time_base, "Video time base changed during decoding")
                timestamp = float((int(frame.pts) - first_pts) * first_time_base)
                require(int(frame.pts) == int(row["pts"]) and Fraction(row["time_base"]) == first_time_base,
                        f"Source PTS differs from audit at frame {idx}")
                require(same_timestamp(timestamp, float(row["timestamp_sec"])), f"Source time differs at frame {idx}")
                require(previous_timestamp is None or timestamp > previous_timestamp, "Video timestamps are not increasing")
                previous_timestamp = timestamp
                decoded_frames += 1
                require((frame.height, frame.width) == (audit["stream"]["height"], audit["stream"]["width"]),
                        f"Source dimensions changed at frame {idx}")
                if idx < next_frame:
                    saved = writer.db.execute("SELECT pts,timestamp FROM frames WHERE frame_idx=?", (idx,)).fetchone()
                    require(saved is not None and saved[0] == int(frame.pts) and same_timestamp(saved[1], timestamp),
                            f"Resume committed prefix does not match decoded source frame {idx}")
                    if idx == next_frame - 1:
                        require(same_timestamp(writer.last_timestamp, timestamp), "Resume last timestamp differs")
                    if (idx + 1) % progress_every == 0:
                        progress("verifying_resume_prefix")
                    continue
                image = frame.to_ndarray(format="bgr24")
                if model_holder[0] is None:
                    model_holder[0] = YOLO(run_identity["checkpoint_path_runtime"])
                    require(len(model_holder[0].names) == 1, "Cache checkpoint must have the trained single rip class")
                use_cuda = str(device) != "cpu" and torch.cuda.is_available()
                if use_cuda:
                    torch.cuda.synchronize()
                prediction_start = time.perf_counter()
                result = model_holder[0].predict(source=image, predictor=CacheSoftMaskPredictor,
                                                device=device, batch=1, **inference)[0]
                actual_runtime = None
                if run_identity["kind"] == "final":
                    require(not model_holder[0].predictor.model.fp16, "Production backend unexpectedly uses float16 inference")
                    actual_runtime = runtime_after_prediction(result)
                    validate_production_runtime(actual_runtime, run_identity["gates"]["validation_runtime"])
                elif str(device) == "cpu":
                    require(not model_holder[0].predictor.model.fp16, "CPU pilot backend unexpectedly uses float16 inference")
                    actual_runtime = cpu_runtime_after_prediction(result, run_identity["cpu_execution"]["torch_threads"])
                boxes = result.boxes.xyxy.detach().cpu().numpy().astype("<f4", copy=False)
                confidences = result.boxes.conf.detach().cpu().numpy().astype("<f4", copy=False)
                work = result.work_soft_masks.detach().cpu().numpy().astype("<f2", copy=False)
                indices = result.candidate_index.detach().cpu().numpy()
                require(np.array_equal(indices, np.arange(len(confidences))), "Predictor changed candidate row order")
                require(bool((result.boxes.cls == 0).all()), "Unexpected predicted class in single-class cache")
                if use_cuda:
                    torch.cuda.synchronize()
                inference_seconds += time.perf_counter() - prediction_start
                if writer is None:
                    metadata = {
                        "run_identity": run_identity, "source_identity": source_identity,
                        "work_shape_hw": list(work.shape[1:]),
                        "original_shape_hw": [frame.height, frame.width],
                        "transform": result.p2_transform, "nms": result.p2_nms,
                        "first_pts": first_pts, "time_base": str(first_time_base),
                        "frame_index_convention": "zero-based sequential presentation-order decoder index",
                        "timestamp_convention": "(frame.pts - first_pts) * time_base seconds",
                        "instance_index_convention": "array row = zero-based raw NMS candidate_index",
                        "soft_mask_dtype": "float16", "confidence_dtype": "float32",
                        "boxes_format": "original image xyxy float32",
                        "candidate_confidence_rule": "conf >= float32(0.1)",
                        "candidate_filtering": "no area or mask-empty filter; all NMS candidates retained",
                    }
                    if actual_runtime is not None:
                        metadata["inference_runtime"] = actual_runtime
                    writer = VideoCacheWriter(path, metadata, resume=resume)
                    metadata = writer.metadata
                else:
                    require(result.p2_transform == metadata["transform"] and result.p2_nms == metadata["nms"],
                            "Predictor transform/NMS configuration differs from cache metadata")
                    require(first_pts == metadata["first_pts"] and str(first_time_base) == metadata["time_base"],
                            "Resume time origin differs")
                    if actual_runtime is not None:
                        if run_identity["kind"] == "final":
                            validate_production_runtime(actual_runtime, metadata.get("inference_runtime"))
                        else:
                            require(actual_runtime == metadata.get("inference_runtime"),
                                    "Temporary CPU runtime differs from cache metadata")
                writer.append(idx, timestamp, int(frame.pts), boxes, confidences, work)
                new_frames += 1
                del result, boxes, confidences, work, indices, image
                if (idx + 1) % progress_every == 0:
                    progress("running")
            require(next(audited_pts, None) is None, "Decoder ended before the PTS audit")
        require(decoded_frames == expected_frames and writer is not None, "Full decoded frame count differs")
        writer.finish(expected_frames)
        progress("readback_verification")
    except BaseException as error:
        progress("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", repr(error))
        raise
    finally:
        if writer is not None:
            writer.close()
    verification = verify_cache(path, task, metadata)
    elapsed = time.perf_counter() - started
    result = {"video_id": video_id, "split": task["split"], "path": str(path),
              "status": "complete", "kind": run_identity["kind"], "frozen": run_identity["gates"]["frozen"],
              "file_bytes": path.stat().st_size, "source_sha256": audit["sha256"],
              "checkpoint_sha256": run_identity["checkpoint_sha256"],
              "elapsed_seconds": elapsed, "new_frames": new_frames, "resumed_from": resumed_from,
              "predict_and_transfer_seconds": inference_seconds,
              "predict_and_transfer_fps": new_frames / inference_seconds if inference_seconds else None,
              "new_frame_export_fps": new_frames / elapsed if elapsed else None,
              "timing_scope": "Wall includes sequential decode/resume verification, model initialization if first video, inference, CPU transfer, SQLite writes, and complete readback; prediction timer includes prediction+CPU transfer, first-call warmup included. Source hashing occurs before this timer.",
              **verification}
    json_write(progress_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--kind", choices=("pilot", "final"), required=True)
    parser.add_argument("--batch", type=int, choices=(1,), default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2), default=1,
                        help="Bound CPU pilot inference threads; interop/OpenCV/video decoding each use one thread")
    parser.add_argument("--progress-every", type=int, choices=(20, 50), default=20)
    parser.add_argument("--frozen-config")
    parser.add_argument("--cache-validation")
    parser.add_argument("--dev-cache-validation", help="Passing validation of all 17 actual dev video caches; required before any final-val video")
    parser.add_argument("--pilot-validation", help="Pilot validation summary, including failed/zero-instance results; never a gate")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    import av
    import numpy as np
    import torch
    import ultralytics
    from p2 import cache_io, soft_masks

    execution_device = configure_execution_device(args.device, args.kind, args.cpu_threads)
    checkpoint = local_path(args.checkpoint)
    require(checkpoint.is_file(), f"Checkpoint missing: {checkpoint}")
    output = local_path(args.output)
    try:
        output.relative_to(ROOT / "package2")
    except ValueError as error:
        raise ValueError("Cache output must remain inside this project's package2 directory") from error
    checkpoint_hash = sha256(checkpoint)
    baseline = load_json(ROOT / "package2/upstream_manifest.json")
    manifests = ROOT / "package2/upstream/data/manifests"
    manifest_hashes = {name: sha256(manifests / name) for name in
                       ("videos.csv", "frames_index.csv", "dev_videos.txt", "val_videos.txt")}
    for name, digest in manifest_hashes.items():
        require(digest == baseline["files"]["data/manifests/" + name], f"Frozen manifest changed: {name}")
    prediction_config = {"inference": soft_masks.PREDICT_DEFAULTS, "nms": soft_masks.NMS_FIXED,
                         "soft_mask_module_sha256": sha256(soft_masks.__file__)}
    gates = validate_final_gates(args.kind, checkpoint_hash, baseline["commit"], prediction_config,
                                 local_path(args.frozen_config) if args.frozen_config else None,
                                 local_path(args.cache_validation) if args.cache_validation else None)
    if args.kind == "pilot":
        require(not args.dev_cache_validation, "Pilot export cannot claim final-val authorization")
        require(args.pilot_validation, "Pilot export requires its validation summary for honest temporary-quality metadata")
        pilot_validation_path = local_path(args.pilot_validation)
        pilot_validation = load_json(pilot_validation_path)
        require(pilot_validation.get("checkpoint_sha256") == checkpoint_hash, "Pilot validation used different checkpoint")
        gates["pilot_validation"] = {
            "report_sha256": sha256(pilot_validation_path), "status": pilot_validation.get("status"),
            "annotated_dev_frames": pilot_validation.get("annotated_dev_frames"),
            "cache_instance_count": pilot_validation.get("cache_instance_count"),
            "all_instances_pass": pilot_validation.get("all_instances_pass"),
            "min_instance_iou": pilot_validation.get("min_instance_iou"),
            "prediction_config_sha256": pilot_validation.get("prediction_config_sha256"),
        }
        gates["mask_quality_validated"] = (pilot_validation.get("status") == "PASS"
            and pilot_validation.get("prediction_config_sha256") == hashlib.sha256(canonical(prediction_config).encode()).hexdigest())
        gates["quality_note"] = ("Temporary pilot checkpoint; use only for interface integration. Failed or zero-instance validation does not establish mask quality or representative final-cache storage cost.")
        if args.device == "cpu":
            gates["production_validation_eligible"] = False
            gates["purpose"] = "temporary_cpu_integration_cache_while_gpu_training"
            gates["pilot_validation"]["production_validation_eligible"] = pilot_validation.get("production_validation_eligible", False)
            gates["pilot_validation"]["validation_runtime"] = pilot_validation.get("validation_runtime")
    else:
        require(not args.pilot_validation, "Final export cannot use pilot validation")
    tasks = load_video_inputs(ROOT, args.videos, args.kind, manifest_hashes)
    run_identity = {"kind": args.kind, "gates": gates, "p1_commit": baseline["commit"],
                    "frozen_manifest_sha256": manifest_hashes, "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_path_runtime": str(checkpoint),
                    "prediction_config": prediction_config,
                    "prediction_config_sha256": hashlib.sha256(canonical(prediction_config).encode()).hexdigest(),
                    "cache_io_module_sha256": sha256(cache_io.__file__), "batch": 1,
                    "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                                 "ultralytics": ultralytics.__version__, "numpy": np.__version__, "av": av.__version__},
                    "device": args.device}
    if args.device == "cpu":
        run_identity["cpu_execution"] = {"torch_threads": args.cpu_threads, "torch_interop_threads": 1,
                                         "opencv_threads": 1, "video_decode_threads": 1,
                                         "device_argument_type": "torch.device", "production_validation_eligible": False}
    final_val_authorization = None
    has_final_val = any(task["split"] == "val" for task in tasks)
    if has_final_val:
        require(args.dev_cache_validation, "Final-val export requires --dev-cache-validation after all 17 actual dev caches pass")
    if args.dev_cache_validation:
        final_val_authorization = validate_final_val_cache_gate(args.dev_cache_validation, run_identity)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "cache_index.json"
    if index_path.exists():
        index = load_json(index_path)
        require(index.get("schema") == INDEX_SCHEMA and index.get("run_identity") == run_identity,
                "Output directory already belongs to a different checkpoint/config/kind")
    else:
        index = {"schema": INDEX_SCHEMA, "run_identity": run_identity, "created_utc": utc_now(), "videos": {}}
    if final_val_authorization is not None:
        previous_authorization = index.get("final_val_authorization")
        if previous_authorization and previous_authorization["report_sha256"] != final_val_authorization["report_sha256"]:
            index.setdefault("prior_final_val_authorizations", {})[previous_authorization["report_sha256"]] = previous_authorization
        index["final_val_authorization"] = final_val_authorization
    index["requested_videos"] = args.videos
    index["status"] = "running"
    json_write(index_path, index)
    model_holder = [None]
    try:
        for task in tasks:
            result = export_video(task, output, run_identity, soft_masks.PREDICT_DEFAULTS,
                                  model_holder, execution_device, args.progress_every, not args.no_resume)
            if task["split"] == "val":
                result["final_val_authorization_sha256"] = final_val_authorization["report_sha256"]
                json_write(output / f"{task['video_id']}_progress.json", result)
            index["videos"][task["video_id"]] = result
            index["updated_utc"] = utc_now()
            json_write(index_path, index)
            print(canonical(result), flush=True)
        require(sha256(checkpoint) == checkpoint_hash, "Checkpoint changed during export")
    except BaseException as error:
        index["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        index["error"] = repr(error)
        index["updated_utc"] = utc_now()
        json_write(index_path, index)
        raise
    index.pop("error", None)
    index["status"] = "complete_for_requested_videos"
    index["updated_utc"] = utc_now()
    index["total_file_bytes"] = sum(row["file_bytes"] for row in index["videos"].values())
    index["total_frames"] = sum(row["readback_verified_frames"] for row in index["videos"].values())
    index["total_candidate_instances"] = sum(row["candidate_instances"] for row in index["videos"].values())
    # This flag concerns final-model size claims, not a conditional early-weight estimate.
    index["storage_extrapolation_valid"] = args.kind == "final"
    with (manifests / "videos.csv").open(encoding="utf-8-sig", newline="") as handle:
        target_frames = sum(int(float(row["n_frames"])) for row in csv.DictReader(handle)
                            if row["split"] in ("dev", "val"))
    index["storage_budget"] = measured_storage_budget(index, target_frames)
    index["all_47_frozen_videos_complete"] = (args.kind == "final" and len(index["videos"]) == 47
        and set(index["videos"]) == set((manifests / "dev_videos.txt").read_text().split()
                                        + (manifests / "val_videos.txt").read_text().split()))
    json_write(index_path, index)
    print(canonical({"status": index["status"], "index": str(index_path), "kind": args.kind,
                     "frozen": gates["frozen"], "total_frames": index["total_frames"],
                     "total_file_bytes": index["total_file_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
