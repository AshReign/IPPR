"""P2 dev selection candidates: one resumable SQLite file per epoch checkpoint.

This is NOT a final video cache and does not implement P5 metrics or P4 output
rules. It visits only frozen annotated dev images, runs each checkpoint once per
image and stores every native post-NMS candidate at float32 confidence >= .1.
Original native binary RLEs are generated from logits, never from fp16 caches.

CPU preparation:
  python tools/collect_dev_predictions.py prepare-sources
GPU collection (after a matching CUDA reuse proof for every checkpoint):
  python tools/collect_dev_predictions.py collect --checkpoint-dir runs/NAME/weights --equivalence-report reports/PROOF.json
CPU consumers use SelectionCandidateReader.iter_frames(threshold), preserving
candidate_index and order. Filtering components, area, score=1 and matching are
deliberately left to the agreed P4/P5 adapter.
Raw collection can precede that adapter: it preserves original-coordinate native
binary masks before component/area filtering. It does not claim P5 evaluation or
working-resolution component filtering, and collection timing is not P6 FPS.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import zlib

import numpy as np

P2_ROOT = Path(__file__).resolve().parents[1]
if str(P2_ROOT) not in sys.path:
    sys.path.insert(0, str(P2_ROOT))
SCHEMA = "p2-dev-selection-candidates-sqlite-v1"
SOURCE_SCHEMA = "p2-dev-image-source-inventory-v1"
KIND = "selection_candidates"
PROOF_SCHEMA = "p2-dev-candidate-reuse-equivalence-v2"
P1_COMMIT = "704f724146792482359ebaa2235cde5c6a41b059"
# Acquisition baseline for the P1 commit above, independent of generated inventories.
FROZEN_MANIFEST_SHA256 = {
    "dev_videos.txt": "c1ac9c74f5b0668ec56844deecfd63bff01badcb5f4ce7cc298c40405ab66c6d",
    "frames_all.csv": "5107e6f19e7193290d29159fe5e1a27d1ca56b7e49695a1d557e2953d7a77a57",
    "frames_index.csv": "36cd9a555c625b4e6ecbf74c802468a8d4d81baff0cf859968307c36e8fef2fb",
    "videos.csv": "f46af188e4e8042e5a31c3ba80e7b78032e0bcad263fd43b140a1279ca7464b9",
    "val_videos.txt": "7f669cbebfa6139225ccbf4766b5bf416d3e4c8f5e624bcaa739b9e9764fa685",
    "yolo_dev_images.txt": "d9341e73e3034b75ef96b3e795b567d47de414063fab437f2be69d9020959792",
    "yolo_train_images.txt": "f2f1d1136b463f304d2e613f86dbcb4010d5f17755b1cd7b7865f5fd93f1fae8",
}
THRESHOLDS = [number / 100 for number in range(10, 91, 5)]
SYNTHETIC_NMS_NAMES = {
    "boundaries_adjacent_floats_overlap_and_tie": 207,
    "401_nonoverlap_tied_scores_max_det_300": 401,
    "401_nonoverlap_unique_scores_max_det_300": 401,
}
REAL_SAMPLE_SEED = 426405
RUNTIME_KEYS = ("device_type", "device_index", "device_name", "device_capability", "torch_version",
    "torch_cuda_version", "torchvision_version", "cudnn_version", "ultralytics_version", "inference_dtype", "batch",
    "matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_deterministic", "cudnn_benchmark",
    "deterministic_algorithms", "float32_matmul_precision", "cublas_workspace_config")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_inventory_signature(document):
    return hashlib.sha256(canonical({k: v for k, v in document.items() if k != "signature"}).encode()).hexdigest()


def verify_frozen_baseline(p2_root=P2_ROOT):
    baseline = json.loads((p2_root / "upstream_manifest.json").read_text())
    if baseline.get("commit") != P1_COMMIT:
        raise ValueError("Upstream acquisition baseline is not the pinned P1 commit")
    for name, expected in FROZEN_MANIFEST_SHA256.items():
        relative = f"data/manifests/{name}"
        if baseline.get("files", {}).get(relative) != expected or sha256(p2_root / "upstream" / relative) != expected:
            raise ValueError(f"P1 frozen manifest differs from acquisition baseline: {name}")


def verify_source_inventory(document, check_stats=True, p2_root=P2_ROOT):
    verify_frozen_baseline(p2_root)
    if document.get("schema") != SOURCE_SCHEMA or document.get("signature") != source_inventory_signature(document):
        raise ValueError("Invalid or modified dev source inventory")
    if document.get("p1_commit") != P1_COMMIT:
        raise ValueError("Inventory is not from the pinned P1 commit")
    if document.get("split") != "dev" or len(document["frames"]) != 2421:
        raise ValueError("Collector requires all 2,421 frozen dev images")
    if len({f["video_id"] for f in document["frames"]}) != 17:
        raise ValueError("Collector requires exactly 17 frozen dev videos")
    frozen_rows = [r for r in read_csv(p2_root / "upstream/data/manifests/frames_index.csv") if r["split"] == "dev"]
    if len(frozen_rows) != len(document["frames"]):
        raise ValueError("Inventory does not cover the frozen dev manifest")
    for index, row in enumerate(document["frames"]):
        if row["ordinal"] != index or not row["image_rel"].startswith("train/images/"):
            raise ValueError("Invalid frozen dev image identity or order")
        frozen = frozen_rows[index]
        if any(row[key] != frozen[key] for key in ("video_id", "image_rel")) or any(
                row[key] != int(frozen[key]) for key in ("frame_idx", "image_id")) or row["p1_t_sec"] != float(frozen["t_sec"]):
            raise ValueError("Inventory frame identity/order differs from frozen P1 dev rows")
        if check_stats:
            stat = Path(row["image_path"]).stat()
            if stat.st_size != row["image_bytes"] or stat.st_mtime_ns != row["image_mtime_ns"]:
                raise ValueError(f"Image changed since source inventory: {row['image_path']}")
    for path, digest in document["input_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Source manifest/timestamp evidence changed: {path}")


def prepare_sources(args):
    """Hash image bytes once, and join P1 metadata with independently audited PTS."""
    verify_frozen_baseline(args.p2_root)
    if args.inventory.exists():
        existing = json.loads(args.inventory.read_text())
        verify_source_inventory(existing, p2_root=args.p2_root)
        print(canonical({"status": "existing_inventory_verified", "path": str(args.inventory), "frames": 2421}))
        return
    import cv2
    cv2.setNumThreads(1)
    manifest = args.p2_root / "upstream/data/manifests/frames_index.csv"
    dev_list = args.p2_root / "upstream/data/manifests/dev_videos.txt"
    dev_ids = set(dev_list.read_text().splitlines())
    rows = [r for r in read_csv(manifest) if r["split"] == "dev"]
    if len(rows) != 2421 or len(dev_ids) != 17 or {r["video_id"] for r in rows} != dev_ids:
        raise ValueError("P1 dev manifest/list/count mismatch")
    input_hashes = {str(p): sha256(p) for p in (manifest, dev_list)}
    timestamp_lookup = {}
    for video in sorted(dev_ids):
        report_path = args.p2_root / f"reports/video_audit/video_{video}.json"
        report = json.loads(report_path.read_text())
        timestamp_file = Path(report["timestamps_file"])
        if not report["source_integrity_pass"] or report["missing_pts_frames"] or report["nonincreasing_pts_transitions"]:
            raise ValueError(f"Source/PTS audit failed for {video}")
        if sha256(timestamp_file) != report["timestamps_sha256"]:
            raise ValueError(f"Timestamp evidence changed for {video}")
        input_hashes[str(report_path)] = sha256(report_path)
        input_hashes[str(timestamp_file)] = report["timestamps_sha256"]
        with gzip.open(timestamp_file, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["p1_annotated"] == "1":
                    timestamp_lookup[(video, int(row["frame_idx"]))] = {
                        "timestamp": float(row["timestamp_sec"]), "pts": int(row["pts"]),
                        "time_base": row["time_base"], "video_sha256": report["sha256"]}
    frames = []
    for index, row in enumerate(rows):
        relative = Path(row["image_rel"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe image path in manifest")
        path = args.dataset_root / relative
        stat = path.stat()
        content = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        frames.append({"ordinal": index, "video_id": row["video_id"], "frame_idx": int(row["frame_idx"]),
            "image_id": int(row["image_id"]), "image_rel": row["image_rel"], "image_path": str(path.resolve()),
            "p1_t_sec": float(row["t_sec"]), "height": image.shape[0], "width": image.shape[1],
            "image_bytes": len(content), "image_mtime_ns": stat.st_mtime_ns,
            "image_sha256": hashlib.sha256(content).hexdigest(),
            **timestamp_lookup[(row["video_id"], int(row["frame_idx"]))]})
    upstream = json.loads((args.p2_root / "upstream_manifest.json").read_text())
    document = {"schema": SOURCE_SCHEMA, "split": "dev", "p1_commit": upstream["commit"],
                "dataset_root": str(args.dataset_root.resolve()), "input_sha256": input_hashes,
                "timestamp_convention": "actual video presentation timestamp relative to first frame; P1 t_sec preserved separately",
                "frames": frames}
    document["signature"] = source_inventory_signature(document)
    verify_source_inventory(document, p2_root=args.p2_root)
    atomic_json(args.inventory, document)
    print(canonical({"status": "source_inventory_created", "path": str(args.inventory), "frames": len(frames)}))


def candidate_digest(frame, transform, candidates):
    digest = hashlib.sha256(canonical({"frame": frame, "transform": transform}).encode())
    for candidate in candidates:
        digest.update(np.asarray(candidate["box"], dtype="<f4").tobytes())
        digest.update(np.asarray([candidate["conf"]], dtype="<f4").tobytes())
        digest.update(canonical({k: candidate[k] for k in ("candidate_index", "class_id", "rle", "mask_area")}).encode())
    return digest.hexdigest()


def validate_candidates(frame, candidates):
    for index, candidate in enumerate(candidates):
        box = np.asarray(candidate["box"], dtype="<f4")
        conf = np.float32(candidate["conf"])
        if candidate["candidate_index"] != index:
            raise ValueError("Candidates must preserve contiguous native NMS order")
        if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(conf) or not np.float32(.1) <= conf <= 1:
            raise ValueError("Invalid float32 box/confidence")
        if box[2] < box[0] or box[3] < box[1] or box[0] < 0 or box[1] < 0 or box[2] > frame["width"] or box[3] > frame["height"]:
            raise ValueError("Box outside original image bounds")
        if candidate["class_id"] != 0:
            raise ValueError("Expected the single rip class 0")
        rle = candidate["rle"]
        if rle["size"] != [frame["height"], frame["width"]] or not isinstance(rle["counts"], str):
            raise ValueError("RLE must describe original image dimensions")
        if not 0 <= candidate["mask_area"] <= frame["height"] * frame["width"]:
            raise ValueError("Invalid binary mask area")


class SelectionCandidateWriter:
    """Atomic per-frame persistence; a failed frame cannot create a partial row."""
    def __init__(self, path, metadata, resume=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not resume:
            raise FileExistsError(self.path)
        self.metadata = dict(metadata, schema=SCHEMA, kind=KIND)
        self.signature = hashlib.sha256(canonical(self.metadata).encode()).hexdigest()
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS frames(
                ordinal INTEGER PRIMARY KEY, identity_json TEXT NOT NULL,
                transform_json TEXT NOT NULL, candidate_count INTEGER NOT NULL, payload_sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS candidates(
                ordinal INTEGER NOT NULL REFERENCES frames(ordinal), candidate_index INTEGER NOT NULL,
                box_f32 BLOB NOT NULL, conf_f32 BLOB NOT NULL, class_id INTEGER NOT NULL,
                rle_height INTEGER NOT NULL, rle_width INTEGER NOT NULL, rle_counts_zlib BLOB NOT NULL,
                mask_area INTEGER NOT NULL, PRIMARY KEY(ordinal,candidate_index));
        """)
        saved = dict(self.db.execute("SELECT key,value FROM metadata"))
        if saved and saved.get("signature") != self.signature:
            self.db.close()
            raise ValueError("Checkpoint/config/source identity differs; refusing resume")
        if not saved:
            self.db.executemany("INSERT INTO metadata VALUES (?,?)", [
                ("spec", canonical(self.metadata)), ("signature", self.signature), ("complete", "false")])
            self.db.commit()
        self.next_frame, first, last = self.db.execute("SELECT COUNT(*),MIN(ordinal),MAX(ordinal) FROM frames").fetchone()
        if self.next_frame and (first != 0 or last != self.next_frame - 1):
            self.db.close()
            raise ValueError("Committed frames are not a contiguous prefix")
        self.complete = saved.get("complete") == "true"
        if self.complete and self.next_frame != self.metadata.get("expected_frames", self.next_frame):
            self.db.close()
            raise ValueError("Completed flag disagrees with expected frame count")
        if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            self.db.close()
            raise ValueError("SQLite integrity check failed")

    def append(self, frame, transform, candidates):
        if self.complete or frame["ordinal"] != self.next_frame:
            raise ValueError("Append requires the next frame of an incomplete file")
        validate_candidates(frame, candidates)
        digest = candidate_digest(frame, transform, candidates)
        with self.db:
            self.db.execute("INSERT INTO frames VALUES (?,?,?,?,?)", (
                self.next_frame, canonical(frame), canonical(transform), len(candidates), digest))
            for candidate in candidates:
                self.db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?)", (
                    self.next_frame, candidate["candidate_index"], np.asarray(candidate["box"], dtype="<f4").tobytes(),
                    np.asarray([candidate["conf"]], dtype="<f4").tobytes(), candidate["class_id"],
                    frame["height"], frame["width"], zlib.compress(candidate["rle"]["counts"].encode("ascii"), 1),
                    int(candidate["mask_area"])))
        self.next_frame += 1

    def finish(self, expected_frames):
        if self.next_frame != expected_frames:
            raise ValueError("Cannot finish an incomplete checkpoint collection")
        if self.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        with self.db:
            self.db.execute("UPDATE metadata SET value='true' WHERE key='complete'")
        self.complete = True

    def close(self):
        self.db.close()


class SelectionCandidateReader:
    def __init__(self, path):
        self.db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        meta = dict(self.db.execute("SELECT key,value FROM metadata"))
        self.metadata = json.loads(meta["spec"])
        if self.metadata.get("schema") != SCHEMA or self.metadata.get("kind") != KIND or hashlib.sha256(canonical(self.metadata).encode()).hexdigest() != meta["signature"]:
            self.db.close()
            raise ValueError("Unsupported or modified metadata")
        self.complete = meta["complete"] == "true"

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]

    def read(self, ordinal):
        record = self.db.execute("SELECT identity_json,transform_json,candidate_count,payload_sha256 FROM frames WHERE ordinal=?", (ordinal,)).fetchone()
        if record is None:
            raise KeyError(ordinal)
        frame, transform = json.loads(record[0]), json.loads(record[1])
        candidates = []
        for idx, box, confidence, cls, h, w, compressed, area in self.db.execute(
                "SELECT candidate_index,box_f32,conf_f32,class_id,rle_height,rle_width,rle_counts_zlib,mask_area FROM candidates WHERE ordinal=? ORDER BY candidate_index", (ordinal,)):
            candidates.append({"candidate_index": idx, "box": np.frombuffer(box, dtype="<f4").copy(),
                "conf": np.frombuffer(confidence, dtype="<f4")[0], "class_id": cls,
                "rle": {"size": [h, w], "counts": zlib.decompress(compressed).decode("ascii")}, "mask_area": area})
        if len(candidates) != record[2] or candidate_digest(frame, transform, candidates) != record[3]:
            raise ValueError(f"Frame {ordinal} candidate data are incomplete or corrupted")
        validate_candidates(frame, candidates)
        return {"frame": frame, "transform": transform, "candidates": candidates}

    def iter_frames(self, threshold=.1):
        threshold = np.float32(threshold)
        if not np.isfinite(threshold) or not np.float32(.1) <= threshold <= np.float32(.9):
            raise ValueError("Selection threshold must be between .10 and .90")
        for ordinal in range(len(self)):
            row = self.read(ordinal)
            row["candidates"] = [c for c in row["candidates"] if c["conf"] >= threshold]
            yield row  # Includes zero-candidate images; order/indices are unchanged.

    def close(self):
        self.db.close()


def native_original_rles(prototypes, coefficients, boxes, original_shape):
    """Full-shape prototype matmul, then one native original mask at a time.

    Interpolation/crop/strict logit threshold are installed Ultralytics ops.
    Peak original-resolution storage is one mask, even with 300 4K candidates.
    CUDA full-versus-sliced equivalence must pass before production collection.
    """
    from pycocotools import mask as mask_utils
    from ultralytics.utils import ops
    channels, ph, pw = prototypes.shape
    if not len(coefficients):
        return []
    logits = (coefficients.float() @ prototypes.float().view(channels, -1)).view(-1, ph, pw)
    encoded = []
    for index in range(len(coefficients)):
        original_logits = ops.scale_masks(logits[index:index+1][None], tuple(original_shape[:2]))[0]
        binary = ops.crop_mask(original_logits, boxes[index:index+1]).gt_(0.)
        mask = np.asfortranarray(binary[0].cpu().numpy(), dtype=np.uint8)
        rle = mask_utils.encode(mask)
        encoded.append({"rle": {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")},
                        "mask_area": int(mask_utils.area(rle))})
        del original_logits, binary, mask
    return encoded


def make_predictor_class():
    import torch
    from ultralytics.engine.results import Results
    from ultralytics.utils import ops
    from p2.soft_masks import CacheSoftMaskPredictor, transform_metadata

    class SelectionCandidatePredictor(CacheSoftMaskPredictor):
        """Raw candidate reader; intentionally bypasses work640 mask creation."""
        def construct_result(self, pred, img, orig_img, img_path, proto):
            pred = pred.clone()
            if len(pred):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            rles = native_original_rles(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])
            cpu = pred[:, :6].detach().float().cpu().numpy()
            candidates = [{"candidate_index": i, "box": cpu[i, :4].copy(), "conf": np.float32(cpu[i, 4]),
                           "class_id": int(cpu[i, 5]), **rles[i]} for i in range(len(cpu))]
            result = Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=None)
            result.selection_candidates = candidates
            result.candidate_index = torch.arange(len(pred), device=pred.device)
            result.p2_transform = transform_metadata(orig_img.shape, img.shape[2:], proto.shape[-2:])
            result.p2_transform["artifact_kind"] = KIND
            result.p2_transform["mask_encoding"] = "original native binary logit>0 COCO RLE; no fp16-work-mask dependency"
            result.p2_transform["candidate_retention"] = "all inclusive post-NMS >=.1 candidates, including zero-area masks"
            return result
    return SelectionCandidatePredictor


def epoch_checkpoints(directory=None, explicit=None):
    paths = list(Path(directory).glob("epoch*.pt")) if directory else [Path(p) for p in explicit or []]
    indexed = []
    for path in paths:
        match = re.fullmatch(r"epoch(\d+)\.pt", path.name)
        if not match:
            raise ValueError(f"Select epochN.pt, not native best/last or pilot aliases: {path}")
        indexed.append((int(match[1]), path.resolve()))
    if not indexed or len({i for i, _ in indexed}) != len(indexed):
        raise ValueError("No unique epoch checkpoints found")
    return sorted(indexed)


def checkpoint_identity(path, filename_epoch):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    try:
        epoch = payload.get("epoch")
        if type(epoch) is not int or epoch != filename_epoch:
            raise ValueError(f"Checkpoint epoch metadata disagrees with numeric filename: {path}")
        if payload.get("ema") is None:
            raise ValueError("Expected native saved EMA checkpoint")
        return {"path": str(path), "sha256": sha256(path), "epoch_zero_based": epoch,
                "completed_epoch": epoch + 1, "model_weight_source": "EMA, as native YOLO checkpoint loader selects",
                "checkpoint_ultralytics_version": payload.get("version"), "train_args": payload.get("train_args")}
    finally:
        del payload


def real_sample_ordinals(frame_count, count):
    """Fixed random ordering; larger scopes extend the same predetermined sample."""
    return np.random.default_rng(REAL_SAMPLE_SEED).permutation(frame_count)[:min(count, frame_count)].tolist()


def runtime_identity(device, inference_dtype="float32"):
    """Call after predictor setup, which may itself change backend flags."""
    import torch
    import torchvision
    import ultralytics
    device = torch.device(device)
    index = (device.index if device.index is not None else torch.cuda.current_device()) if device.type == "cuda" else None
    return {"device_type": device.type, "device_index": index,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "device_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "torchvision_version": torchvision.__version__, "cudnn_version": torch.backends.cudnn.version(),
        "ultralytics_version": ultralytics.__version__, "inference_dtype": inference_dtype, "batch": 1,
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def require_equal_runtime(actual, validated):
    if not isinstance(actual, dict) or not isinstance(validated, dict) or any(
            key not in actual or key not in validated for key in RUNTIME_KEYS):
        raise ValueError("Incomplete CUDA equivalence runtime identity")
    if actual["device_type"] != "cuda" or validated["device_type"] != "cuda":
        raise ValueError("Production collection requires CUDA equivalence, not a CPU diagnostic")
    if actual["inference_dtype"] != "float32" or actual["batch"] != 1 or not actual["device_name"]:
        raise ValueError("Production collection requires identified float32 GPU inference at batch 1")
    changed = [key for key in RUNTIME_KEYS if actual[key] != validated[key]]
    if changed:
        raise ValueError(f"Actual inference runtime differs from CUDA proof: {changed}")


def validate_cuda_equivalence_report(report, checkpoint_sha, config, inventory, inventory_sha):
    """Validate recorded evidence structurally, never trust its PASS flag alone."""
    def need(condition, message):
        if not condition:
            raise ValueError(f"CUDA equivalence gate: {message}")
    need(report.get("schema") == PROOF_SCHEMA and report.get("kind") == "p2_dev_candidate_reuse_equivalence", "unsupported report")
    need(report.get("status") == "CUDA_WITH_REAL_OUTPUT_PASS" and report.get("cuda_requirement_met") is True
         and report.get("all_equal") is True, "full CUDA proof has not passed")
    need(report.get("checkpoint_sha256") == checkpoint_sha, "checkpoint SHA differs")
    need(report.get("source_inventory_signature") == inventory["signature"] and report.get("source_inventory_sha256") == inventory_sha,
         "source inventory differs")
    need(report.get("source_sha256") == config["source_sha256"], "collector/verifier/soft-mask/library source differs")
    need(report.get("inference") == config["inference"] and report.get("nms") == config["nms"], "inference/NMS config differs")
    runtime = report.get("runtime")
    require_equal_runtime(runtime, runtime)
    requested = str(config["device"])
    need(requested in {str(runtime["device_index"]), f"cuda:{runtime['device_index']}"}, "requested CUDA device differs")
    need(runtime["torch_version"] == config["torch"] and runtime["ultralytics_version"] == config["ultralytics"], "runtime version differs")

    def check_nms_case(case):
        need(case.get("pass") is True and case.get("input_unchanged") is True, "NMS case failed or mutated raw input")
        rows = case.get("threshold_results", [])
        need([r.get("threshold") for r in rows] == THRESHOLDS, "all 17 ordered thresholds are required")
        need(type(case.get("raw_candidates")) is int and case["raw_candidates"] > 0, "missing raw candidate count")
        need(type(case.get("retained_at_point1")) is int and 0 <= case["retained_at_point1"] <= 300, "invalid retained candidate count")
        for row in rows:
            need(row.get("exact_box_conf_class_coefficients_and_order_equal") is True, "rows/order/coefficients differ")
            need(type(row.get("direct_count")) is int and 0 <= row["direct_count"] <= 300 and row["direct_count"] == row.get("reused_count"),
                 "direct/reused counts differ")
        need(rows[0]["direct_count"] == case["retained_at_point1"], "base candidate count disagrees")
    synthetic = report.get("synthetic_nms_cases", [])
    need(len(synthetic) == len(SYNTHETIC_NMS_NAMES) and {c.get("name") for c in synthetic} == set(SYNTHETIC_NMS_NAMES),
         "all three boundary/tie/>300 synthetic cases are required")
    for case in synthetic:
        check_nms_case(case)
        need(case["raw_candidates"] == SYNTHETIC_NMS_NAMES[case["name"]], "synthetic raw count differs")
    real = report.get("real_cases", [])
    need(5 <= len(real) <= len(inventory["frames"]), "at least five real dev cases are required")
    sampling = report.get("real_sampling", {})
    expected_ordinals = real_sample_ordinals(len(inventory["frames"]), len(real))
    need(sampling == {"method": "numpy.default_rng.permutation-prefix", "seed": REAL_SAMPLE_SEED,
                      "ordinals": expected_ordinals}, "real examples are not the predetermined sample")
    need([c.get("source_ordinal") for c in real] == expected_ordinals, "real case order/coverage differs")
    real_by_name = {}
    for case, ordinal in zip(real, expected_ordinals):
        check_nms_case(case)
        source = inventory["frames"][ordinal]
        need(case.get("name") == source["image_rel"] and case.get("source_image_sha256") == source["image_sha256"], "real image source differs")
        real_by_name[case["name"]] = (case, source)
    nonempty = sum(c["retained_at_point1"] for c in real)
    need(nonempty > 0 and report.get("real_nonempty_candidates") == nonempty, "real sample has no retained candidates")
    native = report.get("native_rle_cases", [])
    native_real = {}
    native_synthetic = []
    for case in native:
        rows = case.get("rows", [])
        need(case.get("pass") is True and type(case.get("instances")) is int and case["instances"] > 0
             and case["instances"] == len(rows), "incomplete native mask comparison")
        need([r.get("candidate_index") for r in rows] == list(range(len(rows))) and all(r.get("native_pixels_equal") is True for r in rows),
             "native mask pixels/coverage differ")
        if case.get("name") == "synthetic_full_vs_sliced_native":
            native_synthetic.append(case)
        else:
            need(case.get("name") in real_by_name and case["name"] not in native_real, "native mask case not a unique real dev case")
            native_real[case["name"]] = case
    need(len(native_synthetic) == 3 and {tuple(c.get("shape", [])) for c in native_synthetic} == {(79, 173), (173, 79), (128, 128)}
         and all(c["instances"] == 17 for c in native_synthetic), "native synthetic shapes and >8 instances are required")
    need(report.get("real_native_candidate_limit") == 8, "native real comparison scope differs")
    for name, (case, source) in real_by_name.items():
        if case["retained_at_point1"]:
            proof = native_real.get(name)
            need(proof is not None and proof["instances"] == min(8, case["retained_at_point1"])
                 and proof.get("shape") == [source["height"], source["width"]], "missing native comparison for nonempty real case")
        else:
            need(name not in native_real, "empty real case has unexpected native masks")
    return runtime


def collect(args):
    import cv2
    cv2.setNumThreads(1)
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import nms, ops
    import p2.soft_masks as soft_masks
    inventory = json.loads(args.inventory.read_text())
    verify_source_inventory(inventory, p2_root=args.p2_root)
    sources = inventory["frames"]
    predictor = make_predictor_class()
    config = {"inference": soft_masks.PREDICT_DEFAULTS, "nms": soft_masks.NMS_FIXED,
              "mask_representation": "uncompressed native original-coordinate binary before RLE",
              "postprocess_batch": 1, "area_or_component_filter": None, "metric": None,
              "native_empty_mask_filter": False, "candidate_order": "raw post-NMS row index, unchanged",
              "device": args.device, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
              "source_sha256": {"collector": sha256(__file__), "soft_masks": sha256(soft_masks.__file__),
                                "verifier": sha256(Path(__file__).with_name("verify_dev_reuse.py")),
                                "installed_ops": sha256(ops.__file__), "installed_nms": sha256(nms.__file__)}}
    config_sha256 = hashlib.sha256(canonical(config).encode()).hexdigest()
    inventory_sha = sha256(args.inventory)
    reports = {}
    for path in args.equivalence_report:
        proof = json.loads(path.read_text())
        key = proof.get("checkpoint_sha256")
        if key in reports:
            raise ValueError("More than one equivalence report for a checkpoint; select one exact report")
        reports[key] = (proof, path)
    # Gate ALL requested checkpoints before creating a database or loading a GPU model.
    approved = []
    for epoch, checkpoint in epoch_checkpoints(args.checkpoint_dir, args.checkpoint):
        identity = checkpoint_identity(checkpoint, epoch)
        if identity["sha256"] not in reports:
            raise ValueError(f"No full CUDA equivalence report for {checkpoint.name}")
        proof, proof_path = reports[identity["sha256"]]
        runtime = validate_cuda_equivalence_report(proof, identity["sha256"], config, inventory, inventory_sha)
        approved.append((epoch, checkpoint, identity, {"path": str(proof_path.resolve()), "sha256": sha256(proof_path),
                         "schema": PROOF_SCHEMA, "status": proof["status"], "runtime": runtime}))
    args.output.mkdir(parents=True, exist_ok=True)
    prior_index = args.output / "index.json"
    completed_by_epoch = {}
    if prior_index.exists():
        prior = json.loads(prior_index.read_text())
        if prior.get("prediction_config_sha256") != config_sha256 or prior.get("source_inventory_signature") != inventory["signature"]:
            raise ValueError("Output directory contains a different prediction config/source collection")
        completed_by_epoch = {entry["epoch_zero_based"]: entry for entry in prior["checkpoints"]}
    for epoch, checkpoint, identity, equivalence in approved:
        if epoch in completed_by_epoch and completed_by_epoch[epoch]["checkpoint_sha256"] != identity["sha256"]:
            raise ValueError("Output directory already has different weights for this epoch; use a different output directory")
        metadata = {"split": "dev", "temporary_selection_input_not_frozen_result": True,
                    "expected_frames": len(sources), "checkpoint": identity, "config": config,
                    "prediction_config_sha256": config_sha256,
                    "p1_commit": inventory["p1_commit"], "source_inventory_signature": inventory["signature"],
                    "source_inventory_sha256": inventory_sha, "cuda_equivalence": equivalence}
        db_path = args.output / f"epoch{epoch:03d}_{identity['sha256'][:12]}.sqlite"
        writer = SelectionCandidateWriter(db_path, metadata)
        # Read every committed row before accepting resume; never infer success
        # merely from a completed bit, and never redo committed GPU inference.
        reader = SelectionCandidateReader(db_path)
        try:
            for ordinal in range(len(reader)):
                if reader.read(ordinal)["frame"] != sources[ordinal]:
                    raise ValueError("Committed frame source identity changed")
        finally:
            reader.close()
        started = time.perf_counter()
        model = None
        try:
            if not writer.complete:
                model = YOLO(str(checkpoint))
                if len(model.names) != 1:
                    raise ValueError("Selection collection requires the trained single-class rip model")
                for source in sources[writer.next_frame:]:
                    content = Path(source["image_path"]).read_bytes()
                    if hashlib.sha256(content).hexdigest() != source["image_sha256"]:
                        raise ValueError(f"Image bytes changed: {source['image_path']}")
                    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.shape[:2] != (source["height"], source["width"]):
                        raise ValueError("Source image cannot be decoded at expected dimensions")
                    result = model.predict(image, predictor=predictor, device=args.device, **soft_masks.PREDICT_DEFAULTS)[0]
                    if model.predictor.model.fp16:
                        raise ValueError("Predictor unexpectedly enabled float16 inference")
                    require_equal_runtime(runtime_identity(result.boxes.data.device, str(result.boxes.data.dtype).removeprefix("torch.")),
                                          equivalence["runtime"])
                    writer.append(source, result.p2_transform, result.selection_candidates)
                    if writer.next_frame % 25 == 0 or writer.next_frame == len(sources):
                        state = {"kind": KIND, "status": "RUNNING", "checkpoint": identity,
                                 "committed_frames": writer.next_frame, "total_frames": len(sources),
                                 "elapsed_seconds": time.perf_counter() - started, "file": str(db_path)}
                        atomic_json(args.output / "progress.json", state)
                        print(canonical({k: state[k] for k in ("status", "committed_frames", "total_frames", "elapsed_seconds", "file")}), flush=True)
                    del result, image, content
                writer.finish(len(sources))
            completed_by_epoch[epoch] = {"path": str(db_path), "checkpoint_sha256": identity["sha256"],
                                "epoch_zero_based": epoch, "completed_epoch": epoch + 1,
                                "frames": writer.next_frame, "status": "complete", "bytes": db_path.stat().st_size,
                                "cuda_equivalence_report_sha256": equivalence["sha256"]}
        except BaseException as error:
            atomic_json(args.output / "progress.json", {"kind": KIND, "status": "FAILED", "error": repr(error),
                        "checkpoint": identity, "committed_frames": writer.next_frame, "file": str(db_path)})
            raise
        finally:
            writer.close()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        atomic_json(args.output / "index.json", {"kind": KIND, "config": config, "source_inventory_signature": inventory["signature"],
                    "prediction_config_sha256": config_sha256,
                    "metric_or_final_selection_performed": False,
                    "checkpoints": [completed_by_epoch[number] for number in sorted(completed_by_epoch)]})
        atomic_json(args.output / "progress.json", {"kind": KIND, "status": "CHECKPOINT_COMPLETE",
                    "checkpoint": identity, "committed_frames": len(sources), "total_frames": len(sources),
                    "elapsed_seconds": time.perf_counter() - started, "file": str(db_path),
                    "completed_checkpoint_count": len(completed_by_epoch),
                    "metric_or_final_selection_performed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare-sources", "collect"])
    parser.add_argument("--p2-root", type=Path, default=P2_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=P2_ROOT / "dataset")
    parser.add_argument("--inventory", type=Path, default=P2_ROOT / "reports/dev_selection_sources.json")
    parser.add_argument("--output", type=Path, default=P2_ROOT / "outputs/selection_candidates")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--checkpoint-dir", type=Path)
    group.add_argument("--checkpoint", type=Path, nargs="+")
    parser.add_argument("--device", default="0")
    parser.add_argument("--equivalence-report", type=Path, nargs="+", help="Full CUDA proof for every requested checkpoint")
    args = parser.parse_args()
    if args.action == "collect" and not (args.checkpoint_dir or args.checkpoint):
        parser.error("Collect requires --checkpoint-dir or --checkpoint")
    if args.action == "collect" and not args.equivalence_report:
        parser.error("Collect requires --equivalence-report for every checkpoint; CPU/diagnostic proofs are not accepted")
    prepare_sources(args) if args.action == "prepare-sources" else collect(args)


if __name__ == "__main__":
    main()
