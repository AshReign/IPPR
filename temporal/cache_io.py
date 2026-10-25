"""Prediction-cache data structures shared by T0, T1 and T2.

The team convention (A2 Technical Task Packages, "Shared conventions") is one
cache file per video holding, for every frame, ``frame_idx`` and ``timestamp``,
and for every instance its ``box`` (xyxy, original coordinates), ``conf`` and
``soft_mask`` (float16, long side 640 px).

Two on-disk formats are supported, both through ``load_video_cache``:

* ``<video_id>.sqlite`` -- Package 2's cache (schema ``p2-video-cache-sqlite-v1``),
  read with Package 2's own ``p2.cache_io.VideoCacheReader``. Frames are decoded
  lazily, one at a time, because a whole video of float16 soft masks can take
  several hundred MB in memory.
* ``<video_id>.pkl`` -- the pickle written by ``save_video_cache``, used for the
  synthetic test data.

Package 2's README confirms that each ``soft_mask`` is a full working canvas
(long side 640 px, no letterbox padding, zero outside the scaled box), so a
working-resolution mask can be upsampled straight to the original resolution.
``timestamp`` is Package 2's PTS-based time in seconds and is used as given.
"""

from __future__ import annotations

import pickle
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Instance:
    """One predicted rip-current instance in one frame."""

    box: np.ndarray        # (4,) float32, xyxy in original-image pixels
    conf: float            # instance confidence from the segmenter
    soft_mask: np.ndarray  # (h, w) float16 in [0, 1], working resolution


@dataclass
class Frame:
    """All predictions for one decoded video frame."""

    frame_idx: int
    timestamp: float                     # seconds, t = frame_idx / fps
    instances: list[Instance] = field(default_factory=list)
    image_id: int | None = None          # COCO image id if annotated, else None

    @property
    def annotated(self) -> bool:
        return self.image_id is not None


@dataclass
class VideoCache:
    """Frozen-model predictions for every frame of one video."""

    video_id: str                 # e.g. "RipVIS-143" or "RipVIS-NR-004"
    fps: float
    orig_size: tuple[int, int]    # (H, W) of the original video
    work_size: tuple[int, int]    # (h, w) of the cached soft masks
    frames: Sequence[Frame]       # sorted by frame_idx (a list, or lazy for SQLite)
    image_ids: dict[int, int] = field(default_factory=dict)  # frame_idx -> COCO id

    def annotated_frames(self) -> list[Frame]:
        return [f for f in self.frames if f.annotated]

    def attach_image_ids(self, frame_to_image: dict[int, int]) -> None:
        """Mark annotated frames using a {frame_idx: COCO image_id} mapping.

        The mapping comes from the ground-truth COCO file (or P1's
        frames_index.csv), joined on frame_idx, so scoring always uses exactly
        the frames that carry annotations.
        """
        self.image_ids = dict(frame_to_image)
        if isinstance(self.frames, list):          # eager frames: set in place
            for f in self.frames:
                f.image_id = self.image_ids.get(f.frame_idx)

    def close(self) -> None:
        """Release the SQLite connection, if any."""
        if isinstance(self.frames, _SqliteFrames):
            self.frames.close()


class _SqliteFrames(Sequence):
    """Read-only, lazily decoded frames of one Package 2 SQLite cache.

    Each access decodes one frame, so memory stays at one frame's masks even
    for long 4K videos. ``image_id`` is looked up in the owning cache's
    ``image_ids`` at access time, so ``attach_image_ids`` works the same way
    as for eager caches.
    """

    def __init__(self, reader, owner: "VideoCache | None" = None):
        self.reader = reader
        self.owner = owner
        rows = reader.db.execute("SELECT frame_idx FROM frames ORDER BY frame_idx")
        self.frame_ids = [int(r[0]) for r in rows]

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        raw = self.reader.read(self.frame_ids[i])
        instances = [Instance(raw["boxes"][k], float(raw["conf"][k]), raw["soft_mask"][k])
                     for k in range(len(raw["conf"]))]
        image_id = self.owner.image_ids.get(raw["frame_idx"]) if self.owner else None
        return Frame(int(raw["frame_idx"]), float(raw["timestamp"]), instances, image_id)

    def close(self) -> None:
        self.reader.close()


def load_p2_sqlite(path: str | Path, reader_dir: str | Path) -> VideoCache:
    """Open a Package 2 SQLite cache with Package 2's reader (kept as delivered)."""
    reader_dir = str(Path(reader_dir).resolve())
    if reader_dir not in sys.path:
        sys.path.insert(0, reader_dir)
    from p2.cache_io import VideoCacheReader   # Package 2's reader, not modified

    reader = VideoCacheReader(path)
    md = reader.metadata
    frames = _SqliteFrames(reader)
    cache = VideoCache(video_id=md["source_identity"]["video_id"],
                       fps=float("nan"),   # timing uses PTS timestamps, never fps
                       orig_size=tuple(md["original_shape_hw"]),
                       work_size=tuple(md["work_shape_hw"]),
                       frames=frames)
    frames.owner = cache
    return cache


def save_video_cache(cache: VideoCache, path: str | Path) -> None:
    """Write one video's cache (pickle; swap here if Package 2 picks another format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_video_cache(path: str | Path, p2_reader_dir: str | Path | None = None) -> VideoCache:
    """Read one video's cache: a Package 2 ``.sqlite`` or a test ``.pkl``."""
    if str(path).endswith(".sqlite"):
        if p2_reader_dir is None:
            raise ValueError("reading a .sqlite cache needs p2_reader_dir "
                             "(the 'reader' folder delivered by Package 2)")
        return load_p2_sqlite(path, p2_reader_dir)
    with open(path, "rb") as fh:
        cache = pickle.load(fh)
    if not isinstance(cache, VideoCache):
        raise TypeError(f"{path} does not contain a VideoCache")
    cache.frames.sort(key=lambda f: f.frame_idx)
    return cache
