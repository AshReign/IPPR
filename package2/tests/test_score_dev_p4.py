"""Score-runner boundary tests, independent of any full-dev model scoring."""
from copy import deepcopy
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import pytest
from pycocotools import mask as mu

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tested_p4_score_runner", SOURCE / "tools/score_dev_p4.py")
score = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = score
spec.loader.exec_module(score)


def fixture():
    binary = np.asfortranarray(np.ones((30, 20), dtype=np.uint8))
    rle = mu.encode(binary)
    row = {"frame": {"image_id": 14, "image_sha256": "image_sha", "video_id": "dev_vid", "frame_idx": 7,
                     "height": 30, "width": 20},
           "transform": {"work_shape_hw": [15, 10]},
           "candidates": [{"candidate_index": 0, "class_id": 0, "conf": np.float32(.1),
                           "box": np.array([0, 0, 20, 30], dtype=np.float32)}]}
    pair = {"ordinal": 3, "image_id": 14, "source_image_sha256": "image_sha", "video_id": "dev_vid", "source_frame_idx": 7,
            "identity_checked": True, "box_conf_byte_equal": True, "original_native_rle_exact_equal": True,
            "work_payload_sha256": "payload", "work_shape_hw": [15, 10], "original_shape_hw": [30, 20], "candidate_count": 1}
    af = [3, 14, 15, 10, 1, "source", "payload", 1]
    br = [[3, 0, 30, 20, zlib.compress(rle["counts"]), 600, "work_sha"]]
    ar = [[3, 0, 150, "work_sha"]]
    return row, pair, af, br, ar


def test_bind_preserves_true_source_identity_and_parallel_work_area():
    args = fixture()
    out = score.bind_frame(*args)
    assert out["ordinal"] == 3 and out["frame"]["image_id"] == 14
    assert out["areas"] == [150] and out["work_hw"] == (15, 10)
    assert out["rles"][0]["size"] == [30, 20]
    assert out["candidates"][0]["image_id"] == 14
    assert "image_id" not in args[0]["candidates"][0]


@pytest.mark.parametrize("mutation", ["image_id", "ordinal", "video", "frame", "image_sha", "payload", "shape", "original_shape",
    "identity_flag", "area_flag", "candidate_count", "candidate_index", "area_ordinal", "work_sha", "rle_shape",
    "rle_area", "negative_area", "oversized_area", "prefilter", "class"])
def test_bind_fails_closed_on_corrupted_identity_or_statistics(mutation):
    row, pair, af, br, ar = fixture()
    if mutation == "image_id": af[1] = 15
    elif mutation == "ordinal": af[0] = 4
    elif mutation == "video": pair["video_id"] = "val_video"
    elif mutation == "frame": pair["source_frame_idx"] = 8
    elif mutation == "image_sha": pair["source_image_sha256"] = "changed"
    elif mutation == "payload": af[6] = "changed"
    elif mutation == "shape": af[2] = 14
    elif mutation == "original_shape": pair["original_shape_hw"] = [31, 20]
    elif mutation == "identity_flag": pair["box_conf_byte_equal"] = False
    elif mutation == "area_flag": af[7] = 0
    elif mutation == "candidate_count": af[4] = 2
    elif mutation == "candidate_index": ar[0][1] = 1
    elif mutation == "area_ordinal": ar[0][0] = 4
    elif mutation == "work_sha": ar[0][3] = "changed"
    elif mutation == "rle_shape": br[0][2] = 31
    elif mutation == "rle_area": br[0][5] = 599
    elif mutation == "negative_area": ar[0][2] = -1
    elif mutation == "oversized_area": ar[0][2] = 151
    elif mutation == "prefilter": row["candidates"][0]["conf"] = np.nextafter(np.float32(.1), np.float32(0))
    elif mutation == "class": row["candidates"][0]["class_id"] = 1
    with pytest.raises(ValueError):
        score.bind_frame(row, pair, af, br, ar)


def test_bind_retains_zero_candidate_frame():
    row, pair, af, br, ar = fixture()
    row["candidates"] = []
    pair["candidate_count"] = af[4] = 0
    out = score.bind_frame(row, pair, af, [], [])
    assert out["frame"]["image_id"] == 14
    assert out["candidates"] == out["areas"] == out["rles"] == []


def test_protocol_refuses_val_expanded_grid_and_wrong_comparison():
    p = json.loads((SOURCE / "configs/selection_p4_protocol.json").read_text())
    score.protocol_check(p)
    for field, value in [("split", "val"), ("confidence_thresholds", score.THRESHOLDS + ["0.95"]),
                         ("comparison", "conf > threshold"), ("ranking", list(reversed(score.RANKING)))]:
        changed = {**p, field: value}
        with pytest.raises(ValueError):
            score.protocol_check(changed)


@pytest.mark.parametrize("field,value", [("area_fraction", .01), ("area_scale", "original resolution"),
    ("area_stage", "after resize"), ("components", "split components"), ("coco_score", .9),
    ("coco_score", True), ("p5_min_score", .1), ("p5_iou_threshold", .6),
    ("binarization", "work_probability >= 0.5"), ("ordering", "ascending confidence"),
    ("category_mapping", "class_id remains 0"), ("input", "native binary RLE resized down"),
    ("encode_mask", "nearest-neighbor resize"), ("method", "T1")])
def test_protocol_rejects_material_output_rule_change(field, value):
    p = json.loads((SOURCE / "configs/selection_p4_protocol.json").read_text())
    p["output_adapter"][field] = value
    with pytest.raises(ValueError, match="adapter rule"):
        score.protocol_check(p)


@pytest.mark.parametrize("field", ["p5_commit", "p5_evaluator_sha256"])
def test_protocol_rejects_changed_p5_pin(field):
    p = json.loads((SOURCE / "configs/selection_p4_protocol.json").read_text())
    p[field] = "changed"
    with pytest.raises(ValueError, match="P5 pin"):
        score.protocol_check(p)


def test_rank_uses_predeclared_full_precision_ties():
    baseline = {"f2": "0.5", "recall": "0.4", "precision": "0.3", "completed_epoch": 1, "threshold": ".1"}
    rows = [baseline, {**baseline, "completed_epoch": 2}, {**baseline, "threshold": ".2"},
            {**baseline, "f2": "0.50000000000000000001", "completed_epoch": 10}]
    ordered = score.rank_rows(rows)
    assert ordered[0]["completed_epoch"] == 10
    assert [(r["completed_epoch"], r["threshold"]) for r in ordered[1:]] == [(1, ".1"), (1, ".2"), (2, ".1")]
    assert all("rank" not in row for row in rows)


def test_sample_coverage_separates_verified_empty_video_from_real_masks():
    records = [{"ordinal": 0, "frame": {"video_id": "A"}, "candidates": []},
               {"ordinal": 1, "frame": {"video_id": "A"}, "candidates": [1]},
               {"ordinal": 2, "frame": {"video_id": "B"}, "candidates": []},
               {"ordinal": 3, "frame": {"video_id": "B"}, "candidates": []}]
    result = score.sample_coverage(records, [1])
    assert result["nonempty_sample_videos"] == ["A"]
    assert result["verified_empty_only_video_ordinals"] == {"B": [2, 3]}
    assert result["covered_videos"] == ["A", "B"]
    with pytest.raises(ValueError, match="lacks a nonempty"):
        score.sample_coverage(records, [0, 2])


def test_gzip_csv_flushes_complete_bytes_before_hashing(tmp_path):
    path = tmp_path / "audit.csv.gz"
    writer = score.GzipCsv(path, ["candidate_index", "work_area"])
    writer.row({"candidate_index": 3, "work_area": 230})
    writer.close()
    before = score.sha(path)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [{"candidate_index": "3", "work_area": "230"}]
    assert score.sha(path) == before


def test_completed_output_hash_check_rejects_change_and_extra_file(tmp_path):
    (tmp_path / "summary.json").write_text('{"status":"complete"}')
    score.dump_new(tmp_path / "output_hashes.json", score.file_hashes(tmp_path))
    score.verify_output_hashes(tmp_path)
    (tmp_path / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="changed"):
        score.verify_output_hashes(tmp_path)
