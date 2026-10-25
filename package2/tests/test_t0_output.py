"""CPU protocol tests; these do not replace full-dev identity or P5 validation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mu

SOURCE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


t0 = load("tested_p2_t0_output", SOURCE / "p2/t0_output.py")
p5 = load("tested_fixed_p5", SOURCE / "p5_evaluator.py")


def candidate(index=0, conf=0.6, image_id=1):
    return {"candidate_index": index, "class_id": 0, "image_id": image_id,
            "conf": np.float32(conf), "box": [0, 0, 10, 10]}


def rle(mask):
    value = mu.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    value["counts"] = value["counts"].decode("ascii")
    return value


def test_strict_binary_boundary_survives_float16_readback():
    half = np.float16(0.5)
    mask = np.array([[np.nextafter(half, np.float16(0)), half,
                      np.nextafter(half, np.float16(1)), np.float16(1)]], dtype="<f2")
    restored = np.frombuffer(mask.tobytes(), dtype="<f2").reshape(mask.shape)
    assert t0.work_binary_area(restored) == 2


def test_work_area_inclusive_integer_and_fractional_cutoff():
    assert not t0.area_keep(9, (100, 100))
    assert t0.area_keep(10, (100, 100))
    assert t0.area_keep(11, (100, 100))
    assert not t0.area_keep(230, (360, 640))
    assert t0.area_keep(231, (360, 640))
    assert not t0.area_keep(0, (1, 1))


@pytest.mark.parametrize("percentage", range(10, 91, 5))
def test_all17_confidence_boundaries_use_float32_inclusive(percentage):
    threshold = percentage / 100
    level = np.float32(threshold)
    scores = [np.nextafter(level, np.float32(-np.inf)), level,
              np.nextafter(level, np.float32(np.inf))]
    candidates = [candidate(i, score) for i, score in enumerate(scores)]
    outputs = t0.from_encoded(candidates, [rle([[1]])] * 3, [1] * 3, (1, 1), threshold)
    assert [p["candidate_index"] for p in outputs] == [2, 1]
    assert all(p["score"] == 1.0 for p in outputs)
    assert [p["instance_conf"] for p in outputs] == [float(scores[2]), float(scores[1])]


def test_python_double_refilter_would_drop_exact_float32_point_seven():
    score = np.float32(0.7)
    assert float(score) < 0.7  # P4-style Python float second filter would drop this.
    outputs = t0.from_encoded([candidate(conf=score)], [rle([[1]])], [1], (1, 1), 0.7)
    assert len(outputs) == 1


def test_sort_descending_conf_and_original_index_ties_without_mutation():
    candidates = [candidate(8, .5), candidate(3, .8), candidate(2, .8), candidate(1, .2)]
    masks = [rle([[1]]) for _ in candidates]
    before_candidates, before_masks = deepcopy(candidates), deepcopy(masks)
    outputs = t0.from_encoded(candidates, masks, [1] * 4, (1, 1), .1)
    assert [p["candidate_index"] for p in outputs] == [2, 3, 8, 1]
    assert candidates == before_candidates and masks == before_masks
    outputs[0]["segmentation"]["size"][0] = 9
    assert masks == before_masks


def test_disconnected_regions_keep_single_full_instance_and_area_is_before_resize():
    mask = np.zeros((100, 100), dtype=np.float16)
    mask[2:3, 2:7] = 1
    mask[90:91, 90:95] = 1
    assert cv2.connectedComponents((mask > .5).astype(np.uint8), connectivity=8)[0] == 3
    output = t0.from_work([candidate()], [mask], (200, 200), .1, p5.encode_mask)
    assert len(output) == 1
    expected = p5.encode_mask(mask, 200, 200)
    assert output[0]["segmentation"] == expected
    assert t0.work_binary_area(mask) == 10  # each 5-pixel component alone would fail.
    mask[90, 94] = 0
    assert t0.from_work([candidate()], [mask], (200, 200), .1, p5.encode_mask) == []


def test_area_uses_work_count_even_when_original_rle_would_fail_cutoff():
    # Fixture separates semantic inputs; original RLE area cannot substitute for work area.
    outputs = t0.from_encoded([candidate()], [rle(np.zeros((100, 100)))], [1], (10, 100), .1)
    assert len(outputs) == 1
    assert int(mu.area(outputs[0]["segmentation"])) == 0


def test_portrait_noninteger_resize_has_correct_hw_and_matches_cached_path():
    masks = np.zeros((2, 17, 9), dtype=np.float16)
    masks[0, 2:13, 2:6] = np.float16(.51)
    masks[1, 10:14, 6:8] = 1
    candidates = [candidate(9, .3), candidate(4, .7)]
    encoded = [p5.encode_mask(mask, 103, 55) for mask in masks]
    direct = t0.from_work(candidates, masks, (103, 55), .1, p5.encode_mask)
    reuse = t0.from_encoded(candidates, encoded, [t0.work_binary_area(x) for x in masks], (17, 9), .1)
    assert direct == reuse
    assert all(p["segmentation"]["size"] == [103, 55] for p in direct)
    assert all(mu.decode(p["segmentation"]).shape == (103, 55) for p in direct)


def test_empty_candidates_and_empty_masks():
    assert t0.from_work([], [], (360, 640), .1, p5.encode_mask) == []
    assert t0.from_encoded([], [], [], (360, 640), .1) == []
    assert t0.from_work([candidate()], [np.zeros((360, 640), np.float16)], (720, 1280), .1, p5.encode_mask) == []


def test_actual_p5_preserves_score1_list_order_and_zero_prediction_frame():
    gt_mask = rle(np.ones((10, 10)))
    records = [candidate(5, .4), candidate(2, .8)]
    preds = t0.from_encoded(records, [gt_mask, gt_mask], [100, 100], (10, 10), .1)
    frames = pd.DataFrame({"image_id": [1, 2], "video_id": ["dev1", "dev2"],
                           "frame_idx": [0, 0], "t_sec": [0., 0.], "has_rip": [True, False]})
    gt = p5.GroundTruth(frames, {1: (10, 10), 2: (10, 10)}, {1: [gt_mask], 2: []})
    scored = p5.evaluate(preds, gt, min_score=0)
    assert [p["candidate_index"] for p in preds] == [2, 5]
    assert scored[["tp", "fp", "fn"]].values.tolist() == [[1, 1, 0], [0, 0, 0]]
    assert scored.image_id.tolist() == [1, 2]


@pytest.mark.parametrize("bad_mask", [np.array([[np.nan]], np.float32),
    np.array([[1.1]], np.float32), np.array([[-.1]], np.float32),
    np.array([[1]], np.uint8), np.zeros((1, 0), np.float16), np.zeros((1, 1, 1), np.float16)])
def test_reject_invalid_work_fields(bad_mask):
    with pytest.raises(ValueError):
        t0.work_binary_area(bad_mask)


@pytest.mark.parametrize("bad_candidate", [{"class_id": 1}, {"conf": np.nan},
    {"candidate_index": -1}, {"image_id": None}])
def test_reject_invalid_candidate_identity(bad_candidate):
    cand = {**candidate(), **bad_candidate}
    with pytest.raises(ValueError):
        t0.from_encoded([cand], [rle([[1]])], [1], (1, 1), .1)


def test_reject_inconsistent_parallel_identity_and_geometry():
    with pytest.raises(ValueError, match="Repeated"):
        t0.from_encoded([candidate(), candidate()], [rle([[1]])] * 2, [1, 1], (1, 1), .1)
    with pytest.raises(ValueError, match="mix image"):
        t0.from_encoded([candidate(0), candidate(1, image_id=2)], [rle([[1]])] * 2, [1, 1], (1, 1), .1)
    with pytest.raises(ValueError, match="counts differ"):
        t0.from_encoded([candidate()], [], [1], (1, 1), .1)
    with pytest.raises(ValueError, match="exceeds"):
        t0.from_encoded([candidate()], [rle([[1]])], [2], (1, 1), .1)
    with pytest.raises(ValueError, match="original RLE dimensions"):
        t0.from_encoded([candidate(0), candidate(1)], [rle([[1]]), rle([[1, 0]])], [1, 1], (1, 1), .1)
