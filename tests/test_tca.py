import numpy as np
import pytest
from temporal.tca import TCA, calculate_iou, encode_rle, run_tca


def mask(x=4, y=4, size=4, shape=(32, 32)):
    a = np.zeros(shape, np.float32)
    a[y:y+size, x:x+size] = 1
    return a


def config(**kwargs):
    d = dict(sigma=0, low=0.2, high=0.4, tau=0.1, present_dur=0.2,
             absence_dur=0.5, margin=3)
    d.update(kwargs)
    return TCA(**d)


def test_iou():
    a = mask()
    assert calculate_iou(a, a) == 1
    assert calculate_iou(a, mask(15)) == 0


def test_first_frame_not_output():
    t = config()
    assert t.update([mask()], 0) == []


def test_one_frame_false_positive():
    t = config()
    t.update([mask()], 0)
    assert t.update([], 0.1) == []


def test_track_id():
    t = config()
    t.update([mask()], 0)
    out = t.update([mask()], 0.3)
    assert len(out) == 1 and out[0][0] == 1


def test_two_tracks():
    t = config()
    t.update([mask(), mask(20)], 0)
    out = t.update([mask(), mask(20)], 0.3)
    assert {x[0] for x in out} == {1, 2}


def test_short_miss():
    t = config()
    t.update([mask()], 0)
    t.update([mask()], 0.3)
    out = t.update([], 0.4)
    assert len(out) == 1 and out[0][0] == 1


def test_timeout():
    t = config(timeout=0.5)
    t.update([mask()], 0)
    t.update([mask()], 0.3)
    assert t.update([], 1) == []


def test_expand_bbox():
    t = config(iou_threshold=0.1)
    t.update([mask()], 0)
    t.update([mask()], 0.3)
    old = t.tracks[0].box
    t.update([mask(6)], 0.4)
    assert t.tracks[0].box[3] > old[3]
    assert len(t.tracks) == 1


def test_local_storage():
    t = config()
    t.update([mask()], 0)
    assert t.tracks[0].heatmap.size < mask().size


def test_fps_equivalence():
    def simulate(dt):
        t = config(present_dur=0, tau=0.5)
        for i in range(round(1 / dt) + 1):
            t.update([mask()], i * dt)
        return t.tracks[0].heatmap.copy()
    assert np.allclose(simulate(1/30), simulate(1/60), atol=1e-5)


def test_rle():
    m = mask() > 0.5
    r = encode_rle(m)
    values = np.repeat(np.arange(len(r["counts"])) % 2, r["counts"])
    assert np.array_equal(values.reshape(m.shape, order="F"), m)


def test_run_tca_annotated_only():
    frames = [{"t_sec": 0, "detections": [mask()], "annotated": True, "image_id": 1},
              {"t_sec": .3, "detections": [mask()], "annotated": False, "image_id": 2},
              {"t_sec": .4, "detections": [mask()], "annotated": True, "image_id": 3}]
    out = run_tca(frames, dict(sigma=0, tau=.1))
    assert len(out) == 1 and out[0]["image_id"] == 3


def test_reset_between_videos():
    frames = [{"t_sec": 0, "detections": [mask()], "annotated": False},
              {"t_sec": .3, "detections": [mask()], "annotated": True, "image_id": 7}]
    assert run_tca(frames, dict(sigma=0, tau=.1)) == run_tca(frames, dict(sigma=0, tau=.1))


def test_bad_timestamps():
    t = config()
    t.update([mask()], 1)
    with pytest.raises(ValueError):
        t.update([mask()], 1)
