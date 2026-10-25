import math

import pytest

from eval.temporal_metrics import summarise, track_stats


def seq(pattern, dt=0.2, gt=None):
    """Build frames from a string like '11001': 1 = matched, 0 = missed; gt mask optional."""
    gt = gt or "1" * len(pattern)
    return [(i * dt, g == "1", m == "1") for i, (m, g) in enumerate(zip(pattern, gt))]


def test_always_matched():
    s = track_stats("v", seq("11111"))
    assert s.detected and s.events == 0 and s.delay_s == 0


def test_one_gap():
    assert track_stats("v", seq("10011")).events == 1


def test_two_gaps():
    assert track_stats("v", seq("1010101")).events == 3
    assert track_stats("v", seq("1001001")).events == 2


def test_gap_at_end_is_not_a_fragmentation():
    assert track_stats("v", seq("11100")).events == 0


def test_misses_before_first_detection_are_delay_not_fragmentation():
    s = track_stats("v", seq("0000001101"))
    assert s.events == 1
    assert s.delay_s == pytest.approx(1.2)


def test_never_detected():
    s = track_stats("v", seq("0000"))
    assert not s.detected and s.delay_s is None


def test_frames_without_ground_truth_are_ignored():
    # middle two frames have no rip in the ground truth, so they are not a gap
    s = track_stats("v", seq("1001", gt="1001"))
    assert s.events == 0


def test_span_in_minutes():
    s = track_stats("v", [(0.0, True, True), (30.0, True, True), (90.0, True, True)])
    assert s.span_min == pytest.approx(1.5)


def test_unsorted_input_is_sorted():
    frames = list(reversed(seq("10011")))
    assert track_stats("v", frames).events == 1


def test_summary():
    stats = [
        track_stats("a", [(0, True, True), (60, True, False), (120, True, True)]),  # 1 event, 2 min
        track_stats("b", [(0, True, False), (30, True, True), (60, True, True)]),   # 0 events, 1 min, delay 30
        track_stats("c", [(0, True, False), (10, True, False)]),                    # never detected
    ]
    out = summarise(stats)
    assert out["n_tracks"] == 3 and out["n_detected"] == 2
    assert out["frag_raw"] == 1
    assert out["frag_per_track"] == pytest.approx(0.5)
    assert out["frag_per_min"] == pytest.approx(1 / 3)
    assert out["median_delay_s"] == pytest.approx(15)


def test_summary_empty():
    out = summarise([])
    assert out["n_detected"] == 0 and math.isnan(out["frag_per_min"])
