"""Unit tests for T0, T1 and the shared helpers on hand-built sequences."""

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from temporal.cache_io import Frame, Instance, VideoCache
from temporal.common import passes_area_filter
from temporal.t0 import run_t0
from temporal.t1 import components, iter_window_means, run_t1, run_t1_theta_sweep

WORK = ORIG = (40, 60)


def blob(cx=30, cy=20, r=8, size=WORK) -> np.ndarray:
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.float16)


def video(per_frame: list[list[tuple[np.ndarray, float]]], fps: float = 10.0,
          annotate_all: bool = True) -> VideoCache:
    """Cache from a list of frames, each a list of (soft_mask, conf)."""
    frames = [Frame(i, i / fps,
                    [Instance(np.zeros(4, np.float32), c, m) for m, c in insts],
                    image_id=(i + 1) if annotate_all else None)
              for i, insts in enumerate(per_frame)]
    return VideoCache("RipVIS-999", fps, ORIG, WORK, frames)


def decoded(results: list[dict], image_id: int) -> list[np.ndarray]:
    return [mask_utils.decode(r["segmentation"]).astype(bool)
            for r in results if r["image_id"] == image_id]


# ---------------------------------------------------------------- T0
def test_t0_keeps_only_confident_instances():
    cache = video([[(blob(15), 0.9), (blob(45), 0.2)]])
    out = decoded(run_t0(cache, {"conf_thr": 0.5}), 1)
    assert len(out) == 1 and out[0][20, 15] and not out[0][20, 45]


def test_t0_output_is_coco_result_at_original_resolution():
    cache = VideoCache("RipVIS-999", 10, (80, 120), WORK,
                       [Frame(0, 0.0, [Instance(np.zeros(4), 0.9, blob())], image_id=7)])
    (res,) = run_t0(cache, {"conf_thr": 0.5})
    assert res["image_id"] == 7 and res["score"] == 1.0 and res["category_id"] == 1
    assert res["segmentation"]["size"] == [80, 120]


# ---------------------------------------------------------------- T1
def test_t1_suppresses_one_frame_false_positive():
    # A rip present in every frame plus a blob that appears in frame 5 only.
    frames = [[(blob(15), 0.9)] for _ in range(10)]
    frames[5].append((blob(45, r=6), 0.9))
    out = decoded(run_t1(video(frames), {"window_s": 0.5, "theta": 0.5}), 6)
    assert len(out) == 1 and out[0][20, 15] and not out[0][20, 45]


def test_t1_bridges_a_short_miss():
    frames = [[(blob(), 0.9)] for _ in range(10)]
    frames[6] = []                                  # the segmenter misses frame 6
    out = decoded(run_t1(video(frames), {"window_s": 0.5, "theta": 0.5}), 7)
    assert len(out) == 1                            # still reported at the missed frame


def test_t1_is_causal():
    base = [[(blob(), 0.9)] for _ in range(10)]
    changed = [list(f) for f in base]
    for i in range(6, 10):                          # alter only the future
        changed[i] = [(blob(10, r=5), 0.9)]
    a = run_t1(video(base), {"window_s": 1.0, "theta": 0.5})
    b = run_t1(video(changed), {"window_s": 1.0, "theta": 0.5})
    early = lambda res: [r["segmentation"]["counts"] for r in res if r["image_id"] <= 6]
    assert early(a) == early(b)


def test_t1_first_frame_uses_available_history():
    out = decoded(run_t1(video([[(blob(), 0.9)]]), {"window_s": 4.0, "theta": 0.5}), 1)
    assert len(out) == 1


def test_t1_window_in_seconds_is_frame_rate_invariant():
    # The same scene at 10 and 20 FPS: rip visible except during 0.4-0.6 s.
    def scene(fps):
        n = int(2 * fps)
        return video([[] if 0.4 <= i / fps < 0.6 else [(blob(), 0.9)]
                      for i in range(n)], fps=fps)
    means = {}
    for fps in (10.0, 20.0):
        means[fps] = {round(f.timestamp, 3): m.mean()
                      for f, m in iter_window_means(scene(fps), 0.5)}
    shared = sorted(set(means[10.0]) & set(means[20.0]))
    assert shared
    for t in shared:
        assert means[10.0][t] == pytest.approx(means[20.0][t], rel=0.15)


def test_t1_uses_8_connectivity():
    m = np.zeros((10, 10), np.float32)
    m[2:5, 2:5] = 1
    m[5:8, 5:8] = 1                                  # touches only diagonally
    assert len(components(m, 0.5)) == 1


def test_t1_sweep_matches_individual_runs():
    frames = [[(blob(), 0.9)] if i % 3 else [] for i in range(12)]
    cache = video(frames)
    sweep = run_t1_theta_sweep(cache, 0.5, [0.3, 0.6])
    for theta in (0.3, 0.6):
        single = run_t1(cache, {"window_s": 0.5, "theta": theta})
        assert [r["segmentation"]["counts"] for r in single] == \
               [r["segmentation"]["counts"] for r in sweep[theta]]


def test_only_annotated_frames_are_output():
    frames = [Frame(i, i / 10, [Instance(np.zeros(4), 0.9, blob())],
                    image_id=100 if i == 4 else None) for i in range(8)]
    cache = VideoCache("RipVIS-999", 10, ORIG, WORK, frames)
    for res in (run_t0(cache, {"conf_thr": 0.5}),
                run_t1(cache, {"window_s": 1, "theta": 0.5})):
        assert {r["image_id"] for r in res} == {100}


# ---------------------------------------------------------------- shared filter
def test_area_filter_drops_tiny_components():
    tiny = np.zeros((100, 100), bool)
    tiny[0, 0:5] = True                              # 5 px = 0.05 % of the frame
    assert not passes_area_filter(tiny, 0.001)
    tiny[0:2, 0:10] = True                           # 20 px = 0.2 %
    assert passes_area_filter(tiny, 0.001)


def test_area_filter_is_applied_at_original_resolution():
    from temporal.common import encode_result
    work = np.zeros((10, 10), bool)
    work[0, 0] = True                                # 1 % of the working frame
    assert encode_result(work, 1, (100, 100), 0.001) is not None
    assert encode_result(work, 1, (100, 100), 0.02) is None     # ~1 % < 2 %
    entry = encode_result(work, 1, (100, 100), 0.0)
    assert entry["segmentation"]["size"] == [100, 100]          # original resolution
