"""Causal temporal consistency aggregation with local per-track heatmaps."""
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from skimage.filters import apply_hysteresis_threshold


def calculate_iou(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Mask shapes differ")
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _bbox(mask, margin):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    h, w = mask.shape
    return (max(0, int(ys.min()) - margin),
            min(h, int(ys.max()) + margin + 1),
            max(0, int(xs.min()) - margin),
            min(w, int(xs.max()) + margin + 1))


def _union(a, b):
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]),
            min(a[2], b[2]), max(a[3], b[3]))


def _crop(a, box):
    y0, y1, x0, x1 = box
    return a[y0:y1, x0:x1]


def _local_iou(track, detection):
    """IoU without reconstructing a full-frame track mask."""
    det = detection > 0.5
    area = int(det.sum())
    mask = track.output if track.output.any() else track.last_mask
    y0, y1, x0, x1 = track.box
    intersection = int(np.logical_and(mask, det[y0:y1, x0:x1]).sum())
    union = int(mask.sum()) + area - intersection
    return intersection / union if union else 0.0


class Track:
    def __init__(self, track_id, soft, timestamp, margin):
        self.track_id = track_id
        self.shape = soft.shape
        self.margin = margin
        self.box = _bbox(soft > 0.5, margin)
        self.heatmap = np.zeros(_crop(soft, self.box).shape, np.float32)
        self.present_timer = np.zeros_like(self.heatmap)
        self.absent_timer = np.zeros_like(self.heatmap)
        self.output = np.zeros_like(self.heatmap, dtype=bool)
        self.last_mask = _crop(soft > 0.5, self.box).copy()
        self.last_seen = float(timestamp)
        self.last_update = float(timestamp)
        self.active = True

    def expand(self, soft):
        new_box = _bbox(soft > 0.5, self.margin)
        target = _union(self.box, new_box)
        if target == self.box:
            return
        oy0, oy1, ox0, ox1 = self.box
        ny0, ny1, nx0, nx1 = target
        sl = (slice(oy0 - ny0, oy1 - ny0), slice(ox0 - nx0, ox1 - nx0))
        shape = (ny1 - ny0, nx1 - nx0)
        for name in ("heatmap", "present_timer", "absent_timer", "output", "last_mask"):
            old = getattr(self, name)
            new = np.zeros(shape, dtype=old.dtype)
            new[sl] = old
            setattr(self, name, new)
        self.box = target

    def full_mask(self):
        mask = np.zeros(self.shape, dtype=bool)
        y0, y1, x0, x1 = self.box
        mask[y0:y1, x0:x1] = self.output
        return mask


class TCA:
    def __init__(self, iou_threshold=0.3, timeout=2.0, tau=1.0,
                 present_dur=0.2, absence_dur=0.5, sigma=2.0,
                 low=0.2, high=0.4, margin=8):
        if tau <= 0 or timeout < 0 or present_dur < 0 or absence_dur < 0:
            raise ValueError("Invalid time parameter")
        if sigma < 0 or margin < 0 or int(margin) != margin:
            raise ValueError("Invalid sigma or margin")
        if not 0 <= iou_threshold <= 1 or not 0 <= low <= high <= 1:
            raise ValueError("Invalid threshold")
        self.iou_threshold = iou_threshold
        self.timeout = timeout
        self.tau = tau
        self.present_dur = present_dur
        self.absence_dur = absence_dur
        self.sigma = sigma
        self.low = low
        self.high = high
        self.margin = int(margin)
        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.last_timestamp = None
        self.shape = None

    def _advance(self, track, soft, timestamp):
        dt = timestamp - track.last_update
        alpha = -math.expm1(-dt / self.tau)
        local = _crop(soft, track.box) if soft is not None else None
        present = local > 0.5 if local is not None else np.zeros_like(track.heatmap, bool)
        track.present_timer[present] += dt
        track.present_timer[~present] = 0
        track.absent_timer[~present] += dt
        track.absent_timer[present] = 0
        grow = present & (track.present_timer >= self.present_dur)
        fade = ~present & (track.absent_timer >= self.absence_dur)
        if local is not None:
            track.heatmap[grow] = ((1 - alpha) * track.heatmap[grow] +
                                   alpha * local[grow])
        track.heatmap[fade] *= 1 - alpha
        smooth = gaussian_filter(track.heatmap, self.sigma) if self.sigma else track.heatmap
        track.output = apply_hysteresis_threshold(smooth, self.low, self.high)
        track.last_update = timestamp

    def update(self, detections, timestamp):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("Invalid timestamp")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Timestamps must strictly increase")
        detections = [np.asarray(d, np.float32) for d in detections]
        for d in detections:
            if d.ndim != 2 or not np.isfinite(d).all() or np.any((d < 0) | (d > 1)):
                raise ValueError("Detections must be 2D soft masks in [0,1]")
            if self.shape is not None and d.shape != self.shape:
                raise ValueError("Image dimensions changed")
        if self.shape is None and detections:
            self.shape = detections[0].shape
        if self.shape is not None and any(d.shape != self.shape for d in detections):
            raise ValueError("Image dimensions changed")
        self.last_timestamp = timestamp
        for t in self.tracks:
            if t.active and timestamp - t.last_seen > self.timeout:
                t.active = False
        active = [t for t in self.tracks if t.active]
        matched = {}
        if active and detections:
            scores = np.array([[_local_iou(t, d) for d in detections] for t in active])
            rows, cols = linear_sum_assignment(-scores)
            for i, j in zip(rows, cols):
                if scores[i, j] >= self.iou_threshold and np.any(detections[j] > 0.5):
                    matched[active[i].track_id] = j
        used = set(matched.values())
        for t in active:
            j = matched.get(t.track_id)
            if j is not None:
                t.expand(detections[j])
                t.last_mask = _crop(detections[j] > 0.5, t.box).copy()
                t.last_seen = timestamp
            self._advance(t, detections[j] if j is not None else None, timestamp)
        for j, d in enumerate(detections):
            if j not in used and np.any(d > 0.5):
                self.tracks.append(Track(self.next_id, d, timestamp, self.margin))
                self.next_id += 1
        return [(t.track_id, t.full_mask()) for t in self.tracks
                if t.active and t.output.any()]


def encode_rle(mask):
    """COCO uncompressed RLE in column-major order."""
    pixels = np.asarray(mask, np.uint8).ravel(order="F")
    counts = []
    prev, run = 0, 0
    for value in pixels:
        if value != prev:
            counts.append(run)
            run, prev = 0, int(value)
        run += 1
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def run_tca(video_cache, params=None):
    """Frames: {'t_sec': seconds, 'detections': [2D float masks],
    'annotated': bool, 'image_id': int}. Include all processed timestamps.
    Returns COCO-like uncompressed RLE results for annotated frames only.
    One invocation corresponds to one video; state is reset automatically.
    """
    tca = TCA(**(params or {}))
    results = []
    for frame in video_cache:
        outputs = tca.update(frame["detections"], frame["t_sec"])
        if frame.get("annotated", False):
            for track_id, mask in outputs:
                results.append({"image_id": frame["image_id"], "category_id": 1,
                                "segmentation": encode_rle(mask), "score": 1.0,
                                "track_id": track_id})
    return results
