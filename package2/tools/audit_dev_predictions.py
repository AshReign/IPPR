"""CPU-only audit of a complete native dev-candidate collection; never P5 scoring.

Run after every checkpoint collection has finished, in the same local path
environment as the source inventory (normally the existing WSL Python runtime):
  python tools/audit_dev_predictions.py --collection outputs/selection_candidates \
      --output reports/dev_prediction_audit.json

Dependencies: NumPy and pycocotools. No torch, model loading or GPU work is used.
The existing reader owns metadata/payload hashes, candidate order, confidence,
box and declared RLE-dimension checks. This audit adds full-scope/provenance
checks, actual RLE decoding/re-encoding, and inclusive threshold counts only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from collect_dev_predictions import (
    FROZEN_MANIFEST_SHA256, KIND, P1_COMMIT, P2_ROOT, THRESHOLDS,
    SelectionCandidateReader, atomic_json, canonical, sha256,
    validate_cuda_equivalence_report, verify_source_inventory,
)

SCHEMA = "p2-native-dev-prediction-audit-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(value, root=P2_ROOT):
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest_json(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def audit_rle(candidate, frame, mask_utils):
    """Decode only this candidate, checking pixels/area and canonical lossless RLE."""
    encoded = {"size": candidate["rle"]["size"], "counts": candidate["rle"]["counts"].encode("ascii")}
    decoded = mask_utils.decode(encoded)
    require(decoded.shape == (frame["height"], frame["width"]) and decoded.dtype == np.uint8,
            "Decoded RLE dimensions/dtype differ from the original image")
    require(not bool((decoded > 1).any()), "Decoded RLE is not binary")
    area = int(np.count_nonzero(decoded))
    require(area == candidate["mask_area"] == int(mask_utils.area(encoded)), "Decoded RLE area differs")
    rebuilt = mask_utils.encode(np.asfortranarray(decoded))
    require(list(rebuilt["size"]) == encoded["size"] and rebuilt["counts"] == encoded["counts"],
            "RLE decode/re-encode is not the canonical lossless representation")
    return area


def audit_database(reader, sources, mask_utils, progress=None):
    """Stream one frame's compressed candidates and one decoded mask at a time."""
    require(reader.complete, "Checkpoint database is not marked complete")
    require(reader.metadata.get("expected_frames") == len(sources) == len(reader),
            "Checkpoint database does not contain every expected dev row")
    require(reader.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite integrity failed")
    require(not reader.db.execute("PRAGMA foreign_key_check").fetchall(), "Orphan candidate rows exist")
    endpoints = reader.db.execute("SELECT MIN(ordinal),MAX(ordinal) FROM frames").fetchone()
    require(endpoints == (0, len(sources) - 1), "Frame ordinals are not a contiguous complete sequence")
    thresholds = np.asarray(THRESHOLDS, dtype=np.float32)
    retained = np.zeros(len(thresholds), dtype=np.int64)
    empty_at_threshold = np.zeros(len(thresholds), dtype=np.int64)
    nonempty_masks_at_threshold = np.zeros(len(thresholds), dtype=np.int64)
    videos = {}
    candidates_total = empty_frames = zero_area_masks = 0
    minimum_confidence, maximum_confidence = None, None
    for ordinal, source in enumerate(sources):
        row = reader.read(ordinal)  # Reuses all stored hash/order/box/confidence checks.
        require(row["frame"] == source, f"Frame source identity/order differs at ordinal {ordinal}")
        transform = row["transform"]
        require(transform.get("original_shape_hw") == [source["height"], source["width"]]
                and transform.get("artifact_kind") == KIND,
                f"Transform dimensions/artifact kind differ at ordinal {ordinal}")
        require(transform.get("mask_encoding") == "original native binary logit>0 COCO RLE; no fp16-work-mask dependency",
                "Unexpected mask representation")
        candidates = row["candidates"]
        require(len(candidates) <= 300, "Native candidate count exceeds max_det=300")
        confidences = np.asarray([candidate["conf"] for candidate in candidates], dtype=np.float32)
        per_frame = np.zeros(len(thresholds), dtype=np.int64)
        frame_nonempty = 0
        for candidate in candidates:
            area = audit_rle(candidate, source, mask_utils)
            selected = np.float32(candidate["conf"]) >= thresholds
            per_frame += selected
            if area:
                frame_nonempty += 1
                nonempty_masks_at_threshold += selected
            else:
                zero_area_masks += 1
        retained += per_frame
        empty_at_threshold += per_frame == 0
        if len(confidences):
            low, high = float(confidences.min()), float(confidences.max())
            minimum_confidence = low if minimum_confidence is None else min(minimum_confidence, low)
            maximum_confidence = high if maximum_confidence is None else max(maximum_confidence, high)
        count = len(candidates)
        candidates_total += count
        empty_frames += count == 0
        video = videos.setdefault(source["video_id"], {"frames": 0, "candidates": 0,
                                                      "empty_frames": 0, "nonempty_masks": 0})
        video["frames"] += 1
        video["candidates"] += count
        video["empty_frames"] += count == 0
        video["nonempty_masks"] += frame_nonempty
        if progress is not None and ((ordinal + 1) % 100 == 0 or ordinal + 1 == len(sources)):
            progress(ordinal + 1, candidates_total)
    require(reader.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == candidates_total,
            "Unexpected candidate rows outside the audited frames")
    require({video: row["frames"] for video, row in videos.items()} == dict(Counter(s["video_id"] for s in sources)),
            "Video coverage differs from the complete source inventory")
    return {"frames": len(sources), "videos": videos, "candidate_instances": candidates_total,
            "empty_prediction_frames": empty_frames, "zero_area_candidate_masks": zero_area_masks,
            "nonempty_candidate_masks": candidates_total - zero_area_masks,
            "minimum_confidence": minimum_confidence, "maximum_confidence": maximum_confidence,
            "threshold_counts": [{"threshold": threshold, "comparison_float32": float(thresholds[index]),
                                  "candidate_instances": int(retained[index]),
                                  "empty_prediction_frames": int(empty_at_threshold[index]),
                                  "nonempty_candidate_masks": int(nonempty_masks_at_threshold[index])}
                                 for index, threshold in enumerate(THRESHOLDS)]}


def checkpoint_inventory(document, root):
    require(document.get("schema") == "p2-checkpoint-inventory-v1"
            and document.get("status") == "completed_formal_run" and document.get("p1_commit") == P1_COMMIT,
            "Need the completed formal checkpoint inventory from the pinned P1 split")
    require(document.get("dev_manifest_sha256") == FROZEN_MANIFEST_SHA256["yolo_dev_images.txt"],
            "Checkpoint inventory uses another dev split")
    entries = document.get("checkpoints", [])
    count = document.get("actual_epochs")
    require(type(count) is int and 1 <= count <= 30 and len(entries) == count, "Incomplete checkpoint inventory")
    expected = {}
    for entry in entries:
        epoch = entry.get("stored_epoch_zero_based")
        require(type(epoch) is int and 0 <= epoch < count and epoch not in expected
                and entry.get("completed_epoch") == epoch + 1 and entry.get("weights") == "EMA",
                "Checkpoint epochs/EMA identity differ")
        path = local_path(entry["path"], root)
        require(path.name == f"epoch{epoch}.pt" and path.is_file(), "Missing native epoch checkpoint")
        require(path.stat().st_size == entry["bytes"] and sha256(path) == entry["sha256"],
                f"Actual checkpoint bytes changed: {path.name}")
        expected[epoch] = dict(entry, resolved_path=str(path))
    require(len({entry["sha256"] for entry in expected.values()}) == count, "Duplicate checkpoint hashes")
    return expected


def audit_collection(collection, inventory_path, checkpoint_path, root, progress=None):
    from pycocotools import mask as mask_utils

    started = time.perf_counter()
    index_path = collection / "index.json"
    require(index_path.is_file(), "Collection index does not exist; collect predictions before auditing")
    input_hashes = {str(path): sha256(path) for path in (index_path, inventory_path, checkpoint_path)}
    inventory = load_json(inventory_path)
    verify_source_inventory(inventory, p2_root=root)  # Exact pinned 2421 rows / 17 videos and evidence.
    sources = inventory["frames"]
    for source in sources:
        require(sha256(source["image_path"]) == source["image_sha256"], "Source image bytes changed after collection")
    expected = checkpoint_inventory(load_json(checkpoint_path), root)
    index = load_json(index_path)
    require(index.get("kind") == KIND and index.get("metric_or_final_selection_performed") is False,
            "Index is not a raw dev-candidate collection")
    require(index.get("source_inventory_signature") == inventory["signature"], "Index uses another source inventory")
    config = index["config"]
    config_sha = digest_json(config)
    require(index.get("prediction_config_sha256") == config_sha, "Index configuration hash differs")
    require(config.get("postprocess_batch") == 1 and config.get("area_or_component_filter") is None
            and config.get("metric") is None and config.get("native_empty_mask_filter") is False,
            "Collection applied unsupported final filtering/scoring")
    require(config.get("inference", {}).get("conf") == .1
            and config["inference"].get("max_det") == 300 and config["inference"].get("imgsz") == 640,
            "Raw collection inference thresholds/dimensions differ")
    for key, path in {"collector": Path(__file__).with_name("collect_dev_predictions.py"),
                      "verifier": Path(__file__).with_name("verify_dev_reuse.py"),
                      "soft_masks": root / "p2/soft_masks.py"}.items():
        require(config["source_sha256"].get(key) == sha256(path), f"Recorded implementation changed: {key}")
        input_hashes[str(path)] = sha256(path)
    entries = index.get("checkpoints", [])
    require(len(entries) == len(expected), "Not all formal checkpoints have been collected")
    require({entry.get("epoch_zero_based") for entry in entries} == set(expected), "Unexpected/repeated collection epochs")
    indexed_paths = {local_path(entry["path"], root) for entry in entries}
    require(len(indexed_paths) == len(entries) and indexed_paths == {p.resolve() for p in collection.glob("*.sqlite")},
            "Collection has missing, duplicate or unindexed SQLite files")
    summaries = []
    for entry in sorted(entries, key=lambda value: value["epoch_zero_based"]):
        epoch = entry["epoch_zero_based"]
        checkpoint = expected[epoch]
        path = local_path(entry["path"], root)
        require(path.parent == collection and entry.get("status") == "complete"
                and entry.get("frames") == 2421 and entry.get("completed_epoch") == epoch + 1
                and entry.get("checkpoint_sha256") == checkpoint["sha256"]
                and entry.get("bytes") == path.stat().st_size, "Collection index entry is incomplete or inconsistent")
        before_hash = sha256(path)
        reader = SelectionCandidateReader(path)
        try:
            metadata = reader.metadata
            require(metadata.get("split") == "dev" and metadata.get("temporary_selection_input_not_frozen_result") is True
                    and metadata.get("p1_commit") == P1_COMMIT, "Database is not raw dev selection input")
            require(metadata.get("source_inventory_signature") == inventory["signature"]
                    and metadata.get("source_inventory_sha256") == input_hashes[str(inventory_path)],
                    "Database source inventory identity differs")
            require(metadata.get("config") == config and metadata.get("prediction_config_sha256") == config_sha,
                    "Database configuration differs")
            identity = metadata["checkpoint"]
            require(identity.get("sha256") == checkpoint["sha256"] and identity.get("epoch_zero_based") == epoch
                    and identity.get("completed_epoch") == epoch + 1
                    and local_path(identity["path"], root) == Path(checkpoint["resolved_path"]),
                    "Database checkpoint identity differs")
            proof_ref = metadata["cuda_equivalence"]
            proof_path = local_path(proof_ref["path"], root)
            proof_sha = sha256(proof_path)
            require(proof_sha == proof_ref["sha256"] == entry.get("cuda_equivalence_report_sha256"),
                    "Actual CUDA equivalence proof hash differs")
            proof = load_json(proof_path)
            runtime = validate_cuda_equivalence_report(proof, checkpoint["sha256"], config, inventory,
                                                      input_hashes[str(inventory_path)])
            require(runtime == proof_ref.get("runtime") and proof_ref.get("schema") == proof["schema"]
                    and proof_ref.get("status") == proof["status"], "Recorded CUDA proof runtime/status differs")
            input_hashes[str(proof_path)] = proof_sha
            counts = audit_database(reader, sources, mask_utils,
                                    (lambda frames, candidates: progress(epoch, frames, candidates)) if progress else None)
            metadata_sha = digest_json(metadata)
        finally:
            reader.close()
        require(sha256(path) == before_hash, "Database changed during its read-only audit")
        summaries.append({"path": str(path), "sha256": before_hash, "bytes": path.stat().st_size,
                          "metadata_sha256": metadata_sha, "checkpoint_sha256": checkpoint["sha256"],
                          "epoch_zero_based": epoch, "completed_epoch": epoch + 1,
                          "cuda_equivalence_report_path": str(proof_path), "cuda_equivalence_report_sha256": proof_sha,
                          "cuda_equivalence_runtime": runtime, **counts})
    for path, expected_sha in input_hashes.items():
        require(sha256(path) == expected_sha, f"Audit input changed during the scan: {path}")
    # A concurrent writer must not invalidate an earlier checkpoint while later files are audited.
    for row in summaries:
        require(sha256(row["path"]) == row["sha256"], "An earlier checkpoint database changed during the audit")
    return {"schema": SCHEMA, "status": "PASS", "audit_scope": "complete native raw dev prediction collection only",
            "p5_metrics_computed": False, "final_model_or_t0_selected": False, "final_video_cache": False,
            "gpu_used": False, "frames_per_checkpoint": 2421, "dev_videos": 17,
            "checkpoint_count": len(summaries), "total_frame_rows": 2421 * len(summaries),
            "source_images_rehashed": len(sources), "source_inventory_signature": inventory["signature"],
            "source_inventory_sha256": input_hashes[str(inventory_path)],
            "checkpoint_inventory_sha256": input_hashes[str(checkpoint_path)],
            "prediction_config_sha256": config_sha, "prediction_config": config,
            "p1_commit": P1_COMMIT, "input_file_sha256": input_hashes, "checkpoints": summaries,
            "threshold_comparison": "stored float32 confidence >= float32 threshold; no filtering/reordering",
            "memory_scope": "one frame of compressed RLE candidates and one decoded original-resolution mask at a time",
            "auditor_sha256": sha256(__file__), "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--inventory", default="reports/dev_selection_sources.json")
    parser.add_argument("--checkpoint-inventory", default="reports/formal_checkpoint_inventory.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = local_path(args.output)
    output.relative_to(P2_ROOT)
    require(not output.exists(), "Use a new audit output path; existing evidence is preserved")
    def progress(epoch, frames, candidates):
        print(canonical({"status": "AUDITING", "completed_epoch": epoch + 1,
                         "verified_frames": frames, "candidate_instances": candidates}), flush=True)
    try:
        report = audit_collection(local_path(args.collection), local_path(args.inventory),
                                  local_path(args.checkpoint_inventory), P2_ROOT, progress)
    except Exception as error:
        atomic_json(output, {"schema": SCHEMA, "status": "FAIL", "error": repr(error),
                             "p5_metrics_computed": False, "gpu_used": False})
        raise
    atomic_json(output, report)
    print(canonical({"status": report["status"], "report": str(output),
                     "checkpoint_count": report["checkpoint_count"], "total_frame_rows": report["total_frame_rows"],
                     "p5_metrics_computed": False}), flush=True)


if __name__ == "__main__":
    main()
