"""CPU-only P2 selection using verified work areas, paired P5 RLEs and actual P5.

No model imports, inference, training, final-val access, freezing or source edits.
Each epoch is scored once for the predeclared 17 thresholds. Existing complete
epoch results are immutable and hash-verified before reuse; incomplete output
directories are preserved and rejected. --finalize requires all ten epochs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from decimal import Decimal
import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import zlib

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sys.dont_write_bytecode = True
import cv2
import numpy as np
from pycocotools import mask as mu
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

P5_SHA = "b539ae46869407ffdfce9a8d4582d0a72bd722bdead816453d4cc706ab49f8c7"
P5_COMMIT = "2fa7e07ab42cceb49ac149a391c05df0dcc63a1a"
THRESHOLDS = [f"{i / 100:.2f}" for i in range(10, 91, 5)]
RANKING = ["f2 descending", "recall descending", "precision descending", "completed_epoch ascending", "threshold ascending"]
ADAPTER_RULES = {
    "method": "T0 only",
    "input": "full unpadded aspect-preserving work frame; original P2 v3 float16 probabilities, including save/readback semantics",
    "binarization": "work_probability > 0.5",
    "area_fraction": 0.001,
    "area_scale": "working resolution",
    "area_stage": "before P5 encode_mask resize; each entire model instance",
    "area_comparison": "retain when 1000 * work_binary_area >= work_height * work_width",
    "components": "do not split disconnected T0 regions",
    "ordering": "descending float32 instance confidence, then original candidate_index ascending on ties",
    "encode_mask": "pinned P5: >0.5 to float32, INTER_LINEAR to (original_width, original_height), >=0.5 to uint8, Fortran compressed COCO RLE",
    "category_mapping": "P2 class_id 0 to COCO category_id 1",
    "coco_score": 1.0,
    "p5_min_score": 0.0,
    "p5_iou_threshold": 0.5,
}


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(jsonable(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ro_sqlite(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


class GzipCsv:
    def __init__(self, path, columns):
        self.raw = Path(path).open("xb")
        self.gz = gzip.GzipFile(fileobj=self.raw, filename="", mode="wb", mtime=0)
        self.text = io.TextIOWrapper(self.gz, encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.text, fieldnames=columns)
        self.writer.writeheader()
        self.rows = 0

    def row(self, value):
        self.writer.writerow(jsonable(value))
        self.rows += 1

    def close(self):
        self.text.close()
        self.raw.close()


def write_dataframe(path, frame):
    writer = GzipCsv(path, list(frame.columns))
    for record in frame.to_dict("records"):
        writer.row(record)
    writer.close()


def protocol_check(protocol):
    check(protocol.get("schema") == "p2-dev-selection-protocol-v2-p4-confirmed", "Unexpected selection protocol schema")
    check(protocol.get("split") == "dev", "Only dev is permitted")
    check(protocol.get("confidence_thresholds") == THRESHOLDS, "Confidence grid changed")
    check(protocol.get("comparison") == "conf >= float32(threshold)", "Confidence comparison changed")
    check(protocol.get("ranking") == RANKING, "Predeclared ranking changed")
    check(protocol.get("metric_ties") == "exact equality of supplied full-precision serialized P5 values; never rounded display values",
          "Metric tie handling changed")
    check(protocol.get("p5_commit") == P5_COMMIT and protocol.get("p5_evaluator_sha256") == P5_SHA, "P5 pin changed")
    adapter = protocol.get("output_adapter", {})
    for key, expected in ADAPTER_RULES.items():
        actual = adapter.get(key)
        check(not isinstance(actual, bool) and actual == expected, f"Output adapter rule changed: {key}")


def rank_rows(rows):
    ordered = sorted(rows, key=lambda r: (-Decimal(str(r["f2"])), -Decimal(str(r["recall"])),
                     -Decimal(str(r["precision"])), r["completed_epoch"], Decimal(r["threshold"])))
    return [{**row, "rank": i + 1} for i, row in enumerate(ordered)]


def bind_frame(native, pairing, area_frame, encoded_rows, area_rows):
    """Reject identity/geometry/hash mismatches before exposing any scoring input.

    area_frame is the persisted sidecar frames row; both mask row sequences must
    retain their original candidate_index. No positional coercion or sorting here.
    """
    frame, original = native["frame"], native["candidates"]
    ordinal, image_id, wh, ww, n, source, payload, verified = area_frame
    check(verified == 1, "Area identity gate absent")
    check(pairing["ordinal"] == ordinal and pairing["image_id"] == image_id == frame["image_id"], "Frame identity mismatch")
    check(pairing["source_image_sha256"] == frame["image_sha256"], "Image source SHA mismatch")
    check(pairing["video_id"] == frame["video_id"] and pairing["source_frame_idx"] == frame["frame_idx"], "Video/frame identity mismatch")
    check(pairing["identity_checked"] and pairing["box_conf_byte_equal"] and pairing["original_native_rle_exact_equal"], "B identity gate absent")
    check(pairing["work_payload_sha256"] == payload, "Work payload SHA mismatch")
    check(pairing["work_shape_hw"] == [wh, ww] == list(native["transform"]["work_shape_hw"]), "Work geometry mismatch")
    check(pairing["original_shape_hw"] == [frame["height"], frame["width"]], "Original geometry mismatch")
    check(pairing["candidate_count"] == n == len(original) == len(encoded_rows) == len(area_rows), "Candidate count mismatch")
    indices = list(range(n))
    check([c["candidate_index"] for c in original] == indices, "Native candidate order mismatch")
    check([row[1] for row in encoded_rows] == indices == [row[1] for row in area_rows], "Sidecar candidate order mismatch")
    candidates, rles, areas = [], [], []
    for c, encoded, area in zip(original, encoded_rows, area_rows):
        bo, ci, rh, rw, compressed, original_area, work_sha = encoded
        ao, ai, count, area_sha = area
        check(bo == ao == ordinal and ci == ai == c["candidate_index"], "Candidate identity mismatch")
        check(work_sha == area_sha, "Instance work SHA mismatch")
        check((rh, rw) == (frame["height"], frame["width"]), "Encoded original geometry mismatch")
        check(type(count) is int and 0 <= count <= wh * ww, "Invalid work area")
        rle = {"size": [rh, rw], "counts": zlib.decompress(compressed).decode("ascii")}
        check(int(mu.area(rle)) == original_area, "Encoded RLE area differs from stored value")
        check(c["class_id"] == 0 and np.float32(c["conf"]) >= np.float32(.1), "Class or common candidate prefilter differs")
        candidates.append({**c, "image_id": image_id})
        rles.append(rle)
        areas.append(count)
    return {"ordinal": ordinal, "frame": frame, "candidates": candidates, "rles": rles,
            "areas": areas, "work_hw": (wh, ww), "work_shas": [row[-1] for row in encoded_rows],
            "payload_sha": payload, "source": source}


def file_hashes(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "output_hashes.json"}


def verify_output_hashes(root):
    expected = json.loads((root / "output_hashes.json").read_text())
    check(file_hashes(root) == expected, f"Completed score directory changed: {root}")


class Context:
    def __init__(self, output_root, diagnostic_root, p2_root):
        self.root, self.old, self.p2root = map(lambda p: Path(p).resolve(), (output_root, diagnostic_root, p2_root))
        self.out = self.root / "scores"
        self.out.mkdir(exist_ok=True)
        self.collection = self.p2root / "outputs/selection_candidates_formal_v3_01"
        self.manifests = self.p2root / "upstream/data/manifests"
        self.old_manifest_path = self.old / "DELIVERY_SHA256.json"
        self.old_manifest = json.loads(self.old_manifest_path.read_text())["files"]
        self.protocol_path = self.root / "source/configs/selection_p4_protocol.json"
        self.protocol = json.loads(self.protocol_path.read_text())
        protocol_check(self.protocol)
        self.index_path = self.collection / "index.json"
        self.index = json.loads(self.index_path.read_text())
        self.inventory_path = self.p2root / "reports/formal_checkpoint_inventory.json"
        self.inventory = json.loads(self.inventory_path.read_text())
        check(self.inventory["status"] == "completed_formal_run" and self.inventory["actual_epochs"] == 10, "Formal run inventory incomplete")
        check(len(self.index["checkpoints"]) == 10, "Missing checkpoint collection")
        self.reader_path = self.collection / "code_snapshot/tools/collect_dev_predictions.py"
        self.reader = load_module("p4_score_original_reader", self.reader_path)
        self.cache_path = self.p2root / "p2/cache_io.py"
        self.cache = load_module("p4_score_original_cache", self.cache_path)
        self.t0_path = self.root / "source/p2/t0_output.py"
        self.t0 = load_module("p4_score_adapter", self.t0_path)
        self.p5_path = self.root / "source/p5_evaluator.py"
        check(sha(self.p5_path) == P5_SHA, "Fixed P5 version changed")
        check(sha(self.reader_path) == self.index["config"]["source_sha256"]["collector"], "Reader snapshot changed")
        self.p5 = load_module("p4_score_fixed_p5", self.p5_path)
        self.p5.MANIFESTS = self.manifests
        for name in ["dev_videos.txt", "frames_index.csv", "videos.csv", "yolo_dev_images.txt"]:
            check(sha(self.manifests / name) == self.reader.FROZEN_MANIFEST_SHA256[name], f"Frozen P1 source changed: {name}")
        check(sha(self.manifests / "yolo_dev_images.txt") == self.inventory["dev_manifest_sha256"], "Formal run dev differs")
        self.gt = self.p5.load_gt("dev", str(self.p2root / "dataset"))
        check(len(self.gt.frames) == 2421 and self.gt.frames.image_id.nunique() == 2421, "GT dev frame coverage differs")
        check(set(self.gt.frames.split) == {"dev"}, "Non-dev GT")
        check(set(self.gt.frames.video_id) == set((self.manifests / "dev_videos.txt").read_text().splitlines()), "Dev videos differ")
        check(sum(map(len, self.gt.masks.values())) == 3835, "Dev GT count differs")
        check(self.gt.frames.has_rip.notna().all(), "Missing has_rip")
        self.gt_ids = set(int(x) for x in self.gt.frames.image_id)
        self.common_paths = [self.protocol_path, self.index_path, self.inventory_path, self.reader_path, self.cache_path,
                             self.t0_path, self.p5_path, Path(__file__).resolve(), self.old_manifest_path,
                             self.p2root / "dataset/train/coco_annotations/train.json"]
        self.common_paths += [self.manifests / name for name in ["dev_videos.txt", "frames_index.csv", "videos.csv", "yolo_dev_images.txt"]]
        self.common_hashes = {str(path): sha(path) for path in self.common_paths}

    def old_sha(self, relative):
        item = self.old_manifest.get(relative, self.old_manifest.get(relative.replace("/", "\\")))
        check(item is not None, f"Old manifest has no {relative}")
        return item["sha256"]

    def verify_inputs(self, entry):
        epoch = entry["epoch_zero_based"]
        native_path = self.p2root / entry["path"]
        bpath = self.old / "B" / f"epoch{epoch:03d}.sqlite"
        apath = self.root / "areas" / f"epoch{epoch:03d}.sqlite"
        report_path = apath.with_name(f"epoch{epoch:03d}_report.json")
        report = json.loads(report_path.read_text())
        check(report["complete"] and report["sha256"] == sha(apath), "Area output incomplete or hash changed")
        check(sha(bpath) == self.old_sha(f"B/epoch{epoch:03d}.sqlite"), "Old B database hash changed")
        native = self.reader.SelectionCandidateReader(native_path)
        bdb, adb = ro_sqlite(bpath), ro_sqlite(apath)
        bm, am = (dict(db.execute("SELECT key,value FROM metadata")) for db in (bdb, adb))
        bs, area_spec = json.loads(bm["spec"]), json.loads(am["spec"])
        check(native.complete and len(native) == 2421 and bm["complete"] == am["complete"] == "true", "Incomplete candidate data")
        check(sha(native_path) == bs["native_db_sha256"] == area_spec["native_db_sha256"], "Native source hash mismatch")
        check(area_spec["encoded_db_sha256"] == sha(bpath), "Sidecar encoded source mismatch")
        check(area_spec["mask_rule"] == "float16_work > 0.5" and area_spec["split"] == "dev", "Area definition or split differs")
        check(area_spec["script_sha256"] == report["script_sha256"], "Area source version differs")
        check(bs["p5_source_sha256"] == P5_SHA and bs["p5_commit"] == P5_COMMIT and bs["work_float16_roundtrip"], "Old encode route differs")
        check(native.metadata["split"] == "dev" and native.metadata["config"] == self.index["config"], "Native prediction config changed")
        check(native.metadata["checkpoint"]["sha256"] == bs["checkpoint_sha256"] == area_spec["checkpoint_sha256"] == entry["checkpoint_sha256"], "Checkpoint identity differs")
        checkpoint = self.inventory["checkpoints"][epoch]
        check(checkpoint["completed_epoch"] == epoch + 1 and checkpoint["sha256"] == entry["checkpoint_sha256"], "Checkpoint inventory order differs")
        ckpath = self.p2root / checkpoint["path"]
        check(sha(ckpath) == checkpoint["sha256"], "Checkpoint bytes changed")
        check(native.metadata["config"]["inference"]["conf"] == .1, "Common candidate prefilter changed")
        check(bdb.execute("SELECT count(*) FROM frames").fetchone()[0] == adb.execute("SELECT count(*) FROM frames").fetchone()[0] == 2421, "Sidecar frame coverage differs")
        check(bdb.execute("SELECT count(*) FROM masks").fetchone()[0] == adb.execute("SELECT count(*) FROM areas").fetchone()[0] == bs["expected_candidates"], "Sidecar candidate coverage differs")
        inputs = {**self.common_hashes, **{str(p): sha(p) for p in [native_path, bpath, apath, report_path, ckpath]}}
        return native, bdb, adb, inputs


def work_sample_paths(context, epoch, records):
    candidates = {}
    for base in [context.root / "areas/work_samples", context.old / "B/work_samples"]:
        for path in sorted(base.glob(f"epoch{epoch:03d}_*.sqlite")):
            ordinal = int(path.stem.rsplit("_", 1)[1])
            candidates.setdefault(ordinal, []).append(path)
    if epoch == 4:
        by_frame = {(r["frame"]["video_id"], r["frame"]["frame_idx"]): r["ordinal"] for r in records}
        for path in sorted((context.p2root / "reports/formal_epoch5_gpu_v3_validation/roundtrip_samples").glob("*.sqlite")):
            video, frame = path.stem.rsplit("_", 1)
            key = (video, int(frame))
            check(key in by_frame, "Old epoch5 sample not in dev collection")
            candidates.setdefault(by_frame[key], []).append(path)
    check(bool(candidates), "No real work samples available for reuse validation")
    return candidates


def sample_coverage(records, sample_ordinals):
    """Separate true nonempty samples from videos proven empty in the full pool."""
    all_videos = {r["frame"]["video_id"] for r in records}
    nonempty_videos = {r["frame"]["video_id"] for r in records if r["candidates"]}
    nonempty_samples = {records[i]["frame"]["video_id"] for i in sample_ordinals if records[i]["candidates"]}
    check(nonempty_videos <= nonempty_samples, "A video with real candidates lacks a nonempty direct/reuse sample")
    empty_videos = all_videos - nonempty_videos
    empty_ordinals = {video: [r["ordinal"] for r in records if r["frame"]["video_id"] == video] for video in sorted(empty_videos)}
    return {"sample_videos": sorted({records[i]["frame"]["video_id"] for i in sample_ordinals}),
            "nonempty_sample_videos": sorted(nonempty_samples), "verified_empty_only_video_ordinals": empty_ordinals,
            "covered_videos": sorted(nonempty_samples | empty_videos)}


def validate_reuse(context, epoch, records, dest):
    selected = work_sample_paths(context, epoch, records)
    coverage = sample_coverage(records, selected)
    report = {"status": "complete", "epoch_zero_based": epoch, "sample_frames": len(selected),
              "thresholds": THRESHOLDS, "comparisons": 0, "sample_candidates": 0, "inputs": [], "frames": [],
              "nonempty_sample_frames": sum(bool(records[i]["candidates"]) for i in selected),
              "coverage": coverage, "empty_only_video_frame_comparisons": 0,
              "meaning": "Actual direct from_work + fixed P5 encode compared byte-for-byte canonical COCO with reusable paired RLE route; no inverse restoration."}
    for ordinal, paths in sorted(selected.items()):
        record = records[ordinal]
        work = None
        for path in paths:
            cr = context.cache.VideoCacheReader(path)
            check(cr.complete and len(cr) == 1, "Work sample incomplete")
            saved = cr.read(0)
            metadata = cr.metadata
            cr.close()
            expected_checkpoint = context.index["checkpoints"][epoch]["checkpoint_sha256"]
            check(metadata["checkpoint_sha256"] == expected_checkpoint, "Sample checkpoint differs")
            check(metadata["source_video"] == record["frame"]["video_id"] and metadata["source_frame_idx"] == record["frame"]["frame_idx"], "Sample frame differs")
            sample = saved["soft_mask"]
            check(sample.dtype == np.dtype("<f2") and tuple(sample.shape) == (len(record["candidates"]), *record["work_hw"]), "Sample working geometry/dtype differs")
            check(hashlib.sha256(sample.tobytes()).hexdigest() == record["payload_sha"], "Sample work payload differs")
            for i, candidate in enumerate(record["candidates"]):
                check(hashlib.sha256(sample[i].tobytes()).hexdigest() == record["work_shas"][i], "Sample instance work SHA differs")
                check(candidate["box"].astype("<f4").tobytes() == saved["boxes"][i].astype("<f4").tobytes(), "Sample boxes differ")
                check(np.asarray(candidate["conf"], dtype="<f4").tobytes() == np.asarray(saved["conf"][i], dtype="<f4").tobytes(), "Sample confidences differ")
                check(context.t0.work_binary_area(sample[i]) == record["areas"][i], "Sample work area differs")
            report["inputs"].append({"path": str(path), "sha256": sha(path), "ordinal": ordinal})
            if work is None:
                work = sample
        output_digests = {}
        for threshold in THRESHOLDS:
            direct = context.t0.from_work(record["candidates"], work,
                        (record["frame"]["height"], record["frame"]["width"]), float(threshold), context.p5.encode_mask)
            reused = context.t0.from_encoded(record["candidates"], record["rles"], record["areas"], record["work_hw"], float(threshold))
            a, b = canonical(direct).encode(), canonical(reused).encode()
            check(a == b, f"Direct/reuse COCO differs at {epoch}/{ordinal}/{threshold}")
            output_digests[threshold] = hashlib.sha256(a).hexdigest()
            report["comparisons"] += 1
        report["sample_candidates"] += len(record["candidates"])
        report["frames"].append({"ordinal": ordinal, "image_id": record["frame"]["image_id"],
             "video_id": record["frame"]["video_id"], "candidates": len(record["candidates"]),
             "available_sources_verified": len(paths), "output_sha256_by_threshold": output_digests})
    for video, ordinals in coverage["verified_empty_only_video_ordinals"].items():
        for ordinal in ordinals:
            record = records[ordinal]
            check(not record["candidates"] and not record["areas"] and not record["rles"], "Empty-only coverage contains a candidate")
            for threshold in THRESHOLDS:
                direct = context.t0.from_work([], [], (record["frame"]["height"], record["frame"]["width"]), float(threshold), context.p5.encode_mask)
                reused = context.t0.from_encoded([], [], [], record["work_hw"], float(threshold))
                check(direct == reused == [], "Empty-only direct/reuse output differs")
                report["empty_only_video_frame_comparisons"] += 1
    report["videos"] = coverage["sample_videos"]
    check(len(coverage["covered_videos"]) == 17, "Sample plus verified-empty coverage must include all 17 dev videos")
    dump_new(dest / "reuse_equivalence.json", report)
    return report


def score_epoch(context, entry):
    epoch = entry["epoch_zero_based"]
    start = time.perf_counter()
    dest = context.out / f"epoch{epoch:03d}"
    if dest.exists():
        check((dest / "summary.json").exists(), "Incomplete prior epoch output retained; use a new run directory")
        verify_output_hashes(dest)
        summary = json.loads((dest / "summary.json").read_text())
        check(summary["status"] == "complete", "Prior epoch not complete")
        for path, digest in summary["input_sha256"].items():
            check(sha(path) == digest, f"Input of completed epoch changed: {path}")
        print(canonical({"epoch": epoch + 1, "status": "COMPLETE_REUSED_HASH_VERIFIED"}), flush=True)
        return
    native, bdb, adb, inputs = context.verify_inputs(entry)
    dest.mkdir()
    audit = GzipCsv(dest / "candidate_filter_audit.csv.gz", ["ordinal", "video_id", "frame_idx", "image_id", "candidate_index",
        "conf_float32", "work_h", "work_w", "work_binary_area", "keep_work_area", "encoded_original_area", "keep_original_area",
        "work_original_area_decision_differs", "confidence_rank", "original_nms_position", "work_sha256"])
    records, totals = [], Counter()
    for ordinal in range(2421):
        raw = native.read(ordinal)
        pair = json.loads(bdb.execute("SELECT pairing_json FROM frames WHERE ordinal=?", (ordinal,)).fetchone()[0])
        af = adb.execute("SELECT * FROM frames WHERE ordinal=?", (ordinal,)).fetchone()
        br = bdb.execute("SELECT * FROM masks WHERE ordinal=? ORDER BY candidate_index", (ordinal,)).fetchall()
        ar = adb.execute("SELECT * FROM areas WHERE ordinal=? ORDER BY candidate_index", (ordinal,)).fetchall()
        row = bind_frame(raw, pair, af, br, ar)
        check(row["frame"]["image_id"] in context.gt_ids, "Candidate outside dev GT")
        check(context.gt.shapes[row["frame"]["image_id"]] == (row["frame"]["height"], row["frame"]["width"]), "GT and candidate geometry differ")
        records.append(row)
        order = sorted(row["candidates"], key=lambda c: (-float(np.float32(c["conf"])), c["candidate_index"]))
        rank = {c["candidate_index"]: i for i, c in enumerate(order)}
        totals["frames"] += 1
        totals["raw_candidates"] += len(order)
        totals["raw_zero_candidate_frames"] += int(not order)
        totals["frames_reordered_by_confidence"] += int([c["candidate_index"] for c in order] != [c["candidate_index"] for c in row["candidates"]])
        totals["exact_confidence_tie_adjacent_pairs"] += sum(np.float32(a["conf"]) == np.float32(b["conf"]) for a, b in zip(order, order[1:]))
        for c, rle, area, worksha in zip(row["candidates"], row["rles"], row["areas"], row["work_shas"]):
            keep = context.t0.area_keep(area, row["work_hw"])
            original_area = int(mu.area(rle))
            original_keep = 1000 * original_area >= row["frame"]["height"] * row["frame"]["width"]
            totals["area_retained"] += int(keep)
            totals["area_removed"] += int(not keep)
            totals["empty_work_masks"] += int(area == 0)
            totals["work_original_area_decision_differences"] += int(keep != original_keep)
            audit.row({"ordinal": ordinal, "video_id": row["frame"]["video_id"], "frame_idx": row["frame"]["frame_idx"],
                "image_id": row["frame"]["image_id"], "candidate_index": c["candidate_index"], "conf_float32": float(np.float32(c["conf"])),
                "work_h": row["work_hw"][0], "work_w": row["work_hw"][1], "work_binary_area": area,
                "keep_work_area": keep, "encoded_original_area": original_area, "keep_original_area": original_keep,
                "work_original_area_decision_differs": keep != original_keep, "confidence_rank": rank[c["candidate_index"]],
                "original_nms_position": c["candidate_index"], "work_sha256": worksha})
    audit.close()
    native.close(); bdb.close(); adb.close()
    check({r["frame"]["image_id"] for r in records} == context.gt_ids, "Candidate frame set incomplete")
    reuse = validate_reuse(context, epoch, records, dest)
    for sample in reuse["inputs"]:
        path, digest = sample["path"], sample["sha256"]
        check(path not in inputs or inputs[path] == digest, "Conflicting sample input hash")
        inputs[path] = digest
    prediction_order = GzipCsv(dest / "prediction_order.csv.gz", ["threshold", "image_id", "video_id", "frame_idx",
        "output_order_in_frame", "candidate_index", "instance_conf", "score", "category_id"])
    summaries = []
    old_path = context.old / "B_analysis/checkpoint_conf_summary.json"
    check(sha(old_path) == context.old_sha("B_analysis/checkpoint_conf_summary.json"), "Old comparison summary changed")
    old_rows = {r["threshold"]: r for r in json.loads(old_path.read_text()) if r["epoch_zero_based"] == epoch and r["path"] == "encode"}
    for threshold in THRESHOLDS:
        tstart = time.perf_counter()
        predictions = []
        for row in records:
            output = context.t0.from_encoded(row["candidates"], row["rles"], row["areas"], row["work_hw"], float(threshold))
            for order, p in enumerate(output):
                prediction_order.row({"threshold": threshold, "image_id": p["image_id"], "video_id": row["frame"]["video_id"],
                  "frame_idx": row["frame"]["frame_idx"], "output_order_in_frame": order, "candidate_index": p["candidate_index"],
                  "instance_conf": p["instance_conf"], "score": p["score"], "category_id": p["category_id"]})
            predictions.extend(output)
        per_frame = context.p5.evaluate(predictions, context.gt, min_score=0.0, iou_thr=0.5)
        metrics = context.p5.summarise(per_frame)
        check(len(per_frame) == 2421 and set(int(x) for x in per_frame.image_id) == context.gt_ids, "P5 dropped dev frames")
        check(int(per_frame.n_pred.sum()) == len(predictions), "P5 prediction count differs")
        check(int(per_frame.n_gt.sum()) == 3835, "P5 GT count differs")
        write_dataframe(dest / f"conf_{threshold}_per_frame.csv.gz", per_frame)
        by_video = []
        for video_id, group in per_frame.groupby("video_id", sort=False):
            by_video.append({"video_id": video_id, **context.p5.summarise(group)})
        dump_new(dest / f"conf_{threshold}_per_video.json", by_video)
        old = old_rows[threshold]
        summary = {"epoch_zero_based": epoch, "completed_epoch": epoch + 1, "checkpoint_sha256": entry["checkpoint_sha256"],
            "threshold": threshold, "status": "complete", **metrics,
            "n_predictions": len(predictions), "prediction_list_sha256": hashlib.sha256(canonical(predictions).encode()).hexdigest(),
            "elapsed_seconds": time.perf_counter() - tstart,
            "delta_vs_prior_unfiltered_encode": {k: metrics[k] - old[k] for k in ["tp", "fp", "fn", "precision", "recall", "f2"]}}
        summaries.append(summary)
        print(canonical({"epoch": epoch + 1, "threshold": threshold, "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"], "f2": metrics["f2"]}), flush=True)
    prediction_order.close()
    for path, digest in inputs.items():
        check(sha(path) == digest, f"Scoring input changed during run: {path}")
    dump_new(dest / "summary.json", {"status": "complete", "split": "dev", "epoch_zero_based": epoch,
        "completed_epoch": epoch + 1, "p5_commit": P5_COMMIT, "input_sha256": inputs, "counts": dict(totals),
        "reuse_sample_frames": reuse["sample_frames"], "reuse_comparisons": reuse["comparisons"],
        "reuse_equivalence_sha256": sha(dest / "reuse_equivalence.json"), "rows": summaries,
        "elapsed_seconds": time.perf_counter() - start, "frozen": False})
    dump_new(dest / "output_hashes.json", file_hashes(dest))
    print(canonical({"epoch": epoch + 1, "status": "P5_SELECTION_GRID_COMPLETE_NOT_FROZEN", "elapsed_seconds": time.perf_counter() - start}), flush=True)


def finalize(context):
    targets = [context.out / name for name in ["scores.json", "reuse_equivalence_report.json", "selected_not_frozen.json", "ranking.json", "ranking.csv", "final_summary.json", "final_output_hashes.json"]]
    check(not any(path.exists() for path in targets), "Refusing to overwrite prior finalized selection")
    rows, equivalence, epoch_summaries = [], [], []
    for epoch in range(10):
        dest = context.out / f"epoch{epoch:03d}"
        verify_output_hashes(dest)
        summary = json.loads((dest / "summary.json").read_text())
        check(summary["status"] == "complete" and summary["epoch_zero_based"] == epoch and len(summary["rows"]) == 17, "Epoch scoring incomplete")
        for path, digest in summary["input_sha256"].items():
            check(sha(path) == digest, f"Epoch input changed: {path}")
        rows.extend(summary["rows"])
        epoch_summaries.append(summary)
        eq = json.loads((dest / "reuse_equivalence.json").read_text())
        check(eq["status"] == "complete" and len(eq["coverage"]["covered_videos"]) == 17, "Reuse validation incomplete")
        for sample in eq["inputs"]:
            check(summary["input_sha256"].get(sample["path"]) == sample["sha256"] == sha(sample["path"]),
                  "Reuse sample changed or not bound to epoch provenance")
        equivalence.append({"completed_epoch": epoch + 1, "sample_frames": eq["sample_frames"], "sample_candidates": eq["sample_candidates"],
                           "nonempty_sample_frames": eq["nonempty_sample_frames"], "coverage": eq["coverage"],
                           "empty_only_video_frame_comparisons": eq["empty_only_video_frame_comparisons"],
                           "comparisons": eq["comparisons"], "sha256": sha(dest / "reuse_equivalence.json")})
    check(len({(r["checkpoint_sha256"], r["threshold"]) for r in rows}) == 170, "Selection matrix incomplete/duplicated")
    check(sum(s["counts"]["raw_candidates"] for s in epoch_summaries) == 66274, "Raw candidate total differs")
    reuse = {"status": "complete", "split": "dev", "scope": "all raw candidate identities + bounded actual float16 direct/reuse samples, all 17 thresholds",
             "full_candidate_pairs": 66274, "full_checkpoint_frames": 24210, "epochs": equivalence,
             "sample_frames": sum(e["sample_frames"] for e in equivalence), "sample_candidates": sum(e["sample_candidates"] for e in equivalence),
             "nonempty_sample_frames": sum(e["nonempty_sample_frames"] for e in equivalence),
             "direct_reuse_comparisons": sum(e["comparisons"] for e in equivalence),
             "empty_only_video_frame_comparisons": sum(e["empty_only_video_frame_comparisons"] for e in equivalence), "frozen": False}
    dump_new(context.out / "reuse_equivalence_report.json", reuse)
    scores = {"schema": "p2-p5-dev-scores-v1", "status": "complete", "split": "dev", "metric_source": "P5",
        "p1_commit": context.inventory["p1_commit"], "p5_commit": P5_COMMIT,
        "selection_protocol_sha256": sha(context.protocol_path), "evaluator_source_sha256": sha(context.p5_path),
        "output_adapter_sha256": sha(context.t0_path), "prediction_config_sha256": context.index["prediction_config_sha256"],
        "dev_manifest_sha256": context.inventory["dev_manifest_sha256"],
        "reuse_equivalence_report_sha256": sha(context.out / "reuse_equivalence_report.json"), "rows": rows,
        "frozen": False, "input_schema": "paired original-resolution P5 RLE + exact float16 working binary area",
        "scope": "all 2421 annotated dev images; no final-val; no complete video cache"}
    dump_new(context.out / "scores.json", scores)
    selection_path = context.p2root / "p2/selection.py"
    selector = load_module("p4_original_selection", selection_path)
    selected = selector.select(scores, context.inventory, sha(context.protocol_path))
    selected.update({"frozen": False, "user_review_required_before_freeze": True,
        "score_matrix_sha256": sha(context.out / "scores.json"), "checkpoint_inventory_sha256": sha(context.inventory_path),
        "selection_module_sha256": sha(selection_path), "output_adapter_sha256": sha(context.t0_path),
        "protocol_path": str(context.protocol_path)})
    dump_new(context.out / "selected_not_frozen.json", selected)
    ranked = rank_rows(rows)
    check(ranked[0]["checkpoint_sha256"] == selected["winner"]["checkpoint_sha256"] and ranked[0]["threshold"] == selected["winner"]["threshold"], "Selection/ranking discrepancy")
    dump_new(context.out / "ranking.json", ranked)
    columns = ["rank", "completed_epoch", "checkpoint_sha256", "threshold", "tp", "fp", "fn", "precision", "recall", "f1", "f2", "false_alarm_rate", "n_predictions", "n_frames"]
    with (context.out / "ranking.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: row[k] for k in columns} for row in ranked)
    totals = Counter()
    for summary in epoch_summaries:
        totals.update(summary["counts"])
    final = {"status": "P2_DEV_SELECTED_AWAITING_USER_REVIEW_NOT_FROZEN", "p5_commit": P5_COMMIT,
        "counts": dict(totals), "scoring_combinations": len(rows), "winner": selected["winner"], "reuse": reuse,
        "area_rule": "Each whole T0 instance, >0.5 working pixels, retain if 1000*area >= work_h*work_w",
        "ordering": "descending float32 instance confidence, ascending original candidate_index on ties; score=1 after ordering",
        "preserved": ["all original weights/libraries/pilot/validation records", "original inference/NMS", "frozen data splits"],
        "not_executed": ["training", "freeze", "final-val evaluation", "full video cache export", "upload"],
        "common_input_sha256": context.common_hashes}
    dump_new(context.out / "final_summary.json", final)
    dump_new(context.out / "final_output_hashes.json", {p.relative_to(context.out).as_posix(): sha(p)
        for p in sorted(context.out.rglob("*")) if p.is_file() and p.name != "final_output_hashes.json"})
    print(canonical({"status": final["status"], "winner": selected["winner"], "counts": dict(totals)}), flush=True)


def install_guard(root):
    def guard(event, args):
        if event == "import" and args[0].split(".")[0] in {"torch", "ultralytics"}:
            raise RuntimeError("Model imports forbidden in CPU scoring")
        if event.startswith(("socket.", "subprocess.")) or event == "os.system":
            raise RuntimeError("Network/subprocess forbidden in CPU scoring")
        if event == "open":
            path, mode, flags = args
            write = ((mode and any(c in mode for c in "wax+")) or (flags and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if write and isinstance(path, (str, bytes, os.PathLike)):
                check(Path(path).resolve().is_relative_to(root / "scores"), f"Write outside scores: {path}")
    sys.addaudithook(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--p2-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--epochs", type=int, nargs="+")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    check(args.epochs is not None or args.finalize, "Specify --epochs and/or --finalize")
    if args.epochs is not None:
        check(len(set(args.epochs)) == len(args.epochs) and all(0 <= e < 10 for e in args.epochs), "Invalid/repeated epoch")
    install_guard(args.output_root.resolve())
    context = Context(args.output_root, args.diagnostic_root, args.p2_root)
    environment = {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "numpy_path": np.__file__,
        "opencv": cv2.__version__, "pycocotools": importlib.metadata.version("pycocotools"), "command": sys.argv,
        "model_imported": any(k.split(".")[0] in {"torch", "ultralytics"} for k in sys.modules), "script_sha256": sha(__file__)}
    envpath = context.out / ("environment_finalize.json" if args.finalize and args.epochs is None else "environment_epochs_" + "_".join(map(str, args.epochs)) + ".json")
    if not envpath.exists():
        dump_new(envpath, environment)
    if args.epochs is not None:
        for epoch in args.epochs:
            score_epoch(context, context.index["checkpoints"][epoch])
    if args.finalize:
        finalize(context)


if __name__ == "__main__":
    main()
