"""Tests for the interim matching rule, the selection rule and the full runner."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pycocotools import mask as mask_utils

from eval.interim_scoring import Counts, match_image, metrics, parse_file_name
from scripts.make_synthetic_cache import make_dataset
from scripts.select_configs import freeze
from scripts.tune_dev import main as tune_main
from temporal.search import load_search_space, t0_configs, t1_configs, t2_configs
from temporal.selection import select

SPACE = Path(__file__).resolve().parents[1] / "configs" / "search_space.yaml"


def rle(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def square(x0, size=10, shape=(40, 40)):
    m = np.zeros(shape, bool)
    m[5:5 + size, x0:x0 + size] = True
    return m


# ---------------------------------------------------------------- matching rule
def test_matching_perfect_miss_and_false_positive():
    a, b = square(2), square(25)
    assert match_image([rle(a)], [rle(a)]) == (1, 0, 0)          # perfect
    assert match_image([], [rle(a)]) == (0, 0, 1)                # miss
    assert match_image([rle(a)], []) == (0, 1, 0)                # FP on a no-rip frame
    assert match_image([rle(a), rle(b)], [rle(a)]) == (1, 1, 0)  # one extra prediction


def test_matching_duplicate_predictions_on_one_rip():
    a = square(2)
    assert match_image([rle(a), rle(a)], [rle(a)]) == (1, 1, 0)


def test_matching_below_iou_threshold_is_fp_and_fn():
    shifted = square(8)                                           # IoU = 0.25 with square(2)
    assert match_image([rle(shifted)], [rle(square(2))]) == (0, 1, 1)


def test_metrics_and_false_alarm_rate():
    m = metrics(Counts(tp=8, fp=2, fn=2, norip_frames=10, norip_frames_with_pred=3))
    assert m["precision"] == pytest.approx(0.8) and m["recall"] == pytest.approx(0.8)
    assert m["f2"] == pytest.approx(0.8) and m["false_alarm_rate"] == pytest.approx(0.3)


def test_parse_file_name():
    assert parse_file_name("RipVIS-143_01416.jpg") == ("RipVIS-143", 1416)
    assert parse_file_name("RipVIS-NR-026_00600.jpg") == ("RipVIS-NR-026", 600)
    assert parse_file_name("RipVIS-035_0005.jpg") == ("RipVIS-035", 5)      # not zero-padded
    assert parse_file_name("train/images/RipVIS-31_00006.jpg") == ("RipVIS-031", 6)


# ---------------------------------------------------------------- search space
def test_search_space_sizes_and_determinism():
    space = load_search_space(str(SPACE))
    assert len(t0_configs(space)) == 17
    assert len(t1_configs(space)) == 76
    a, b = t2_configs(space), t2_configs(space)
    assert len(a) == 20 and a == b                                # fixed seed
    for c in a:
        p = c["params"]
        assert 0.1 <= p["tau"] <= 3.0 and p["low"] < p["high"]
        assert p["iou_thr"] == 0.3


# ---------------------------------------------------------------- selection rule
def _df(rows):
    return pd.DataFrame(rows, columns=["config_id", "condition", "recall", "f2"])


def _dfp(rows):
    """Rows with a JSON params column, as written by the runner."""
    return pd.DataFrame([[c, k, r, f, json.dumps(p)] for c, k, r, f, p in rows],
                        columns=["config_id", "condition", "recall", "f2", "params"])


def test_selection_prefers_f2_among_configs_reaching_r0():
    df = _df([["T0", "T0", 0.60, 0.55],
              ["a", "T1", 0.70, 0.60], ["b", "T1", 0.55, 0.80], ["c", "T1", 0.65, 0.62]])
    assert select(df)["T1"]["config_id"] == "c"      # b has higher F2 but lower recall


def test_selection_breaks_ties_by_recall_then_id():
    df = _df([["T0", "T0", 0.5, 0.5],
              ["b", "T1", 0.7, 0.6], ["a", "T1", 0.6, 0.6], ["c", "T1", 0.7, 0.6]])
    assert select(df)["T1"]["config_id"] == "b"


def test_t0_threshold_is_tuned_and_sets_r0():
    df = _dfp([["T0_a", "T0", 0.80, 0.60, {"conf_thr": 0.10}],
               ["T0_b", "T0", 0.70, 0.65, {"conf_thr": 0.30}],
               ["T0_c", "T0", 0.75, 0.65, {"conf_thr": 0.20}],
               ["x", "T1", 0.74, 0.90, {"window_s": 1.0, "theta": 0.5}],
               ["y", "T1", 0.76, 0.70, {"window_s": 1.0, "theta": 0.5}]])
    chosen = select(df)
    assert chosen["T0"]["config_id"] == "T0_c"      # F2 tie -> higher recall
    assert chosen["T1"]["r0"] == 0.75 and chosen["T1"]["config_id"] == "y"


def test_selection_tie_break_shorter_window_then_lower_theta():
    df = _dfp([["T0", "T0", 0.5, 0.5, {"conf_thr": 0.1}],
               ["a", "T1", 0.7, 0.6, {"window_s": 2.0, "theta": 0.3}],
               ["b", "T1", 0.7, 0.6, {"window_s": 1.0, "theta": 0.5}],
               ["c", "T1", 0.7, 0.6, {"window_s": 1.0, "theta": 0.4}]])
    assert select(df)["T1"]["config_id"] == "c"


def test_selection_falls_back_to_highest_recall_and_flags():
    df = _df([["T0", "T0", 0.9, 0.7], ["a", "T1", 0.6, 0.8], ["b", "T1", 0.7, 0.5]])
    chosen = select(df)["T1"]
    assert chosen["config_id"] == "b" and chosen["flag"]


# ---------------------------------------------------------------- end to end
def test_runner_and_freeze_on_synthetic_data(tmp_path):
    paths = make_dataset(tmp_path / "syn", seed=1)
    space = load_search_space(str(SPACE))
    space["t1"] = {"windows_s": [0.5, 1.0], "theta": {"start": 0.3, "stop": 0.7, "step": 0.2}}
    space["t2"]["n_configs"] = 3
    space["t0"] = {"conf": {"start": 0.1, "stop": 0.5, "step": 0.2}}   # 3 thresholds
    small = tmp_path / "space.yaml"
    small.write_text(yaml.safe_dump(space))

    out = tmp_path / "run"
    tune_main(["--cache-dir", str(paths["cache_dir"]), "--gt-json", str(paths["gt_json"]),
               "--manifest", str(paths["manifest"]),
               "--search-space", str(small), "--tca-module", "tests.fake_tca",
               "--workers", "2", "--out-dir", str(out)])

    pooled = pd.read_csv(out / "dev_search.csv")
    assert len(pooled) == 3 + 2 * 3 + 3 and (pooled["n_videos"] == 4).all()
    per_video = pd.read_csv(out / "dev_search_per_video.csv")
    assert len(per_video) == len(pooled) * 4
    t0 = pooled[pooled["condition"] == "T0"].sort_values("f2").iloc[-1]
    assert t0["norip_frames"] > 0                    # the no-rip video is scored
    meta = json.loads((out / "dev_search_meta.json").read_text())
    assert meta["machine"]["cpu_count"]              # tuning machine is recorded

    frozen = tmp_path / "frozen"
    freeze(out / "dev_search.csv", frozen)
    t1 = yaml.safe_load((frozen / "t1.yaml").read_text())
    assert t1["condition"] == "T1" and "window_s" in t1["params"]
    assert len(t1["source_csv_sha256"]) == 64
    with pytest.raises(FileExistsError):             # never re-tune after freezing
        freeze(out / "dev_search.csv", frozen)

    # Temporal smoothing should beat raw predictions on this noisy synthetic data.
    best_t1 = json.loads(json.dumps(t1["dev_metrics"]))
    assert best_t1["f2"] >= t0["f2"]
