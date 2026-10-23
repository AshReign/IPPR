"""T1: causal sliding-window averaging (Package 4, step 2).

For every frame t:
  1. merge all cached instances into one map with a pixel-wise max of their
     soft masks (a frame with no instances gives an all-zero map);
  2. keep a ring buffer of the maps whose timestamps fall in (t - w, t];
  3. average the buffer and threshold at theta;
  4. split the result into connected components (8-connectivity); each
     component is one output instance.

The window is defined in seconds, so a given ``window_s`` covers the same
duration at 24 and 60 FPS. Only current and past frames are used (causal),
and the buffer starts empty for every video, so the first frames simply
average whatever is available.

Unlike TCA, T1 has no notion of instance identity: neighbouring rips whose
averaged maps touch merge into one component. Comparing T2 with T1 therefore
isolates what instance tracking adds beyond smoothing.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

import numpy as np
from skimage.measure import label

from .cache_io import VideoCache
from .common import DEFAULT_MIN_AREA_FRAC, masks_to_results, passes_area_filter


def merged_map(frame, work_size: tuple[int, int]) -> np.ndarray:
    """Pixel-wise max of all instance soft masks in one frame (float32)."""
    out = np.zeros(work_size, dtype=np.float32)
    for inst in frame.instances:
        np.maximum(out, inst.soft_mask.astype(np.float32), out=out)
    return out


def iter_window_means(cache: VideoCache, window_s: float
                      ) -> Iterator[tuple[object, np.ndarray]]:
    """Yield (annotated frame, causal window mean) in frame order.

    A running sum keeps each step O(pixels) regardless of the window length.
    Every frame enters the buffer, annotated or not, so the window really
    spans ``window_s`` seconds of video.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    buffer: deque[tuple[float, np.ndarray]] = deque()
    running = np.zeros(cache.work_size, dtype=np.float64)

    for frame in cache.frames:
        m = merged_map(frame, cache.work_size)
        buffer.append((frame.timestamp, m))
        running += m
        # Drop maps that fall outside (t - w, t]. A small tolerance stops
        # floating-point timestamps from dropping a frame exactly on the edge.
        while buffer and buffer[0][0] <= frame.timestamp - window_s + 1e-9:
            _, old = buffer.popleft()
            running -= old
        if frame.annotated:
            yield frame, (running / len(buffer)).astype(np.float32)


def components(mean_map: np.ndarray, theta: float,
               min_area_frac: float) -> list[np.ndarray]:
    """Threshold a window mean and split it into 8-connected components."""
    labels = label(mean_map >= theta, connectivity=2)  # connectivity=2 -> 8-neighbour
    masks = [labels == k for k in range(1, labels.max() + 1)]
    return [m for m in masks if passes_area_filter(m, min_area_frac)]


def run_t1(cache: VideoCache, params: dict) -> list[dict]:
    """Return COCO results for the annotated frames of one video.

    params:
        window_s       window length in seconds
        theta          threshold on the averaged map
        min_area_frac  shared output filter (default 0.001)
    """
    window_s = float(params["window_s"])
    theta = float(params["theta"])
    min_area = float(params.get("min_area_frac", DEFAULT_MIN_AREA_FRAC))

    results: list[dict] = []
    for frame, mean in iter_window_means(cache, window_s):
        results.extend(masks_to_results(components(mean, theta, min_area),
                                        cache, frame.image_id))
    return results


def run_t1_theta_sweep(cache: VideoCache, window_s: float, thetas: list[float],
                       min_area_frac: float = DEFAULT_MIN_AREA_FRAC
                       ) -> dict[float, list[dict]]:
    """Run T1 for several thresholds at once.

    The window means depend only on ``window_s``, so they are computed once and
    re-thresholded for every theta. This gives identical results to calling
    ``run_t1`` per theta, at a fraction of the cost during the dev search.
    """
    out: dict[float, list[dict]] = {t: [] for t in thetas}
    for frame, mean in iter_window_means(cache, window_s):
        for theta in thetas:
            out[theta].extend(masks_to_results(
                components(mean, theta, min_area_frac), cache, frame.image_id))
    return out
