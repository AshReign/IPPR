import numpy as np
import pandas as pd
import pytest

from eval.evaluator import GroundTruth, encode_mask, evaluate, summarise, temporal_frames

H, W = 20, 20


def box(r0, r1, c0, c1, h=H, w=W):
    m = np.zeros((h, w), bool)
    m[r0:r1, c0:c1] = True
    return m


def make_gt(frames):
    """frames: list of (image_id, [gt masks], has_rip). One video, 0.2 s apart."""
    rows = [("v", i, i * 0.2, img, len(ms), rip) for i, (img, ms, rip) in enumerate(frames)]
    df = pd.DataFrame(rows, columns=["video_id", "frame_idx", "t_sec", "image_id", "n_gt", "has_rip"])
    return GroundTruth(df, {img: (H, W) for img, _, _ in frames},
                       {img: [encode_mask(m, H, W) for m in ms] for img, ms, _ in frames})


def pred(img, mask, score=1.0):
    return {"image_id": img, "category_id": 1, "segmentation": encode_mask(mask, H, W), "score": score}


def score(preds, frames, **kw):
    per_frame = evaluate(preds, make_gt(frames), **kw)
    return per_frame, summarise(per_frame)


RIP = box(5, 15, 5, 15)


def test_perfect_prediction():
    _, s = score([pred(1, RIP)], [(1, [RIP], True)])
    assert (s["tp"], s["fp"], s["fn"]) == (1, 0, 0)
    assert s["precision"] == s["recall"] == s["f2"] == 1.0


def test_miss():
    _, s = score([], [(1, [RIP], True)])
    assert (s["tp"], s["fp"], s["fn"]) == (0, 0, 1)
    assert s["precision"] == 1.0 and s["recall"] == 0.0 and s["f2"] == 0.0


def test_false_positive():
    _, s = score([pred(1, box(0, 3, 0, 3))], [(1, [RIP], True)])
    assert (s["tp"], s["fp"], s["fn"]) == (0, 1, 1)


def test_iou_just_below_threshold_is_fp():
    half = box(5, 15, 5, 10)          # IoU 0.5 exactly with RIP: a TP
    less = box(5, 15, 5, 9)           # IoU 0.4: an FP
    assert score([pred(1, half)], [(1, [RIP], True)])[1]["tp"] == 1
    assert score([pred(1, less)], [(1, [RIP], True)])[1]["fp"] == 1


def test_empty_ground_truth_and_false_alarm_rate():
    frames = [(1, [], False), (2, [], False), (3, [RIP], True)]
    per_frame, s = score([pred(1, RIP)], frames)
    assert (s["tp"], s["fp"], s["fn"]) == (0, 1, 1)
    assert s["false_alarm_rate"] == 0.5 and s["n_no_rip_frames"] == 2
    assert per_frame.n_pred.tolist() == [1, 0, 0]


def test_no_predictions_and_no_ground_truth():
    _, s = score([], [(1, [], False)])
    assert s["precision"] == s["recall"] == 1.0 and s["false_alarm_rate"] == 0.0


def test_multiple_instances():
    a, b = box(0, 5, 0, 5), box(10, 20, 10, 20)
    _, s = score([pred(1, b), pred(1, a)], [(1, [a, b], True)])
    assert (s["tp"], s["fp"], s["fn"]) == (2, 0, 0)


def test_duplicate_predictions_on_one_rip():
    _, s = score([pred(1, RIP), pred(1, box(5, 15, 5, 14))], [(1, [RIP], True)])
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 0)


def test_taken_best_match_is_fp_even_if_another_gt_fits():
    # The official rule is greedy: a prediction whose best GT is already matched is an FP,
    # even when it would pass on another GT. pred2 has IoU 9/11 with both GTs (a tie),
    # so it picks the first GT, which pred1 already took.
    g1, g2 = box(0, 20, 0, 10), box(0, 20, 2, 12)
    _, s = score([pred(1, g1), pred(1, box(0, 20, 1, 11))], [(1, [g1, g2], True)])
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1)


def test_file_order_not_score_order():
    # Same two predictions, opposite file order: the first one in the file claims the GT.
    good, dup = box(5, 15, 5, 15), box(5, 15, 5, 13)
    a = score([pred(1, good, 0.2), pred(1, dup, 0.9)], [(1, [RIP], True)])[1]
    b = score([pred(1, dup, 0.9), pred(1, good, 0.2)], [(1, [RIP], True)])[1]
    assert (a["tp"], a["fp"]) == (b["tp"], b["fp"]) == (1, 1)


def test_min_score_drops_predictions():
    _, s = score([pred(1, RIP, 0.05)], [(1, [RIP], True)], min_score=0.1)
    assert (s["tp"], s["fn"]) == (0, 1)


def test_uncompressed_rle_is_accepted():
    flat = RIP.ravel(order="F").astype(np.uint8)
    edges = np.flatnonzero(np.diff(np.r_[0, flat, 0]))
    counts = np.diff(np.r_[0, edges, flat.size]).tolist()
    p = {"image_id": 1, "segmentation": {"size": [H, W], "counts": counts}, "score": 1.0}
    assert score([p], [(1, [RIP], True)])[1]["tp"] == 1


def test_wrong_size_and_unknown_image_raise():
    small = {"image_id": 1, "segmentation": encode_mask(RIP, 10, 10)}
    with pytest.raises(ValueError, match="size"):
        score([small], [(1, [RIP], True)])
    with pytest.raises(ValueError, match="not in this split"):
        score([pred(9, RIP)], [(1, [RIP], True)])


def test_encode_mask_upsamples_to_original_size():
    from pycocotools import mask as mask_utils
    work = box(2, 6, 2, 6, h=10, w=10)              # 4x4 block at half resolution
    full = mask_utils.decode(encode_mask(work, 20, 20))
    assert full.shape == (20, 20)
    assert full.sum() == 64 and full[4:12, 4:12].all()


def test_temporal_frames_feed_track_stats():
    frames = [(1, [RIP], True), (2, [RIP], True), (3, [RIP], True)]
    per_frame, _ = score([pred(1, RIP), pred(3, RIP)], frames)
    assert temporal_frames(per_frame, "v") == [(0.0, True, True), (0.2, True, False), (0.4, True, True)]
