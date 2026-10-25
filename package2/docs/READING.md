# Reading the frozen caches

Run the examples from the extracted or checked-out `package2/` directory. Download and extract caches to a local SSD first. Set your own relative or absolute cache path; original producer-machine paths in provenance metadata are not required to read a relocated SQLite file.

## Choose the correct path

| Entry point | Purpose | Model/GPU required? | Calls inverse restoration? |
|---|---|---|---|
| `reader/read_raw_cache.py` | Inspect aligned raw arrays and timeline | No | No |
| `reader/read_final_t0.py` | Frozen T0 → shared P5 `encode_mask` → COCO predictions | No | No |
| `reader/restoration_v4/read_v4.py` | Optional restoration of one untouched original-model working field | No model execution; CPU PyTorch/SciPy dependencies | Explicit v4 only |

For P3/P4, the raw `soft_mask` is the reusable working probability field. Keep its shape, candidate identity, box/confidence alignment and timestamp. A field changed by temporal averaging or tracking must follow the agreed downstream output protocol; it no longer satisfies v4's inverse assumptions.

## Raw inspection and Python API

Only Python, NumPy and standard-library SQLite/zlib are needed for raw reads; `requirements-raw.txt` pins this minimal dependency. The full `requirements-reader.txt` also includes the T0 dependencies.

```bash
python reader/read_raw_cache.py --cache cache/dev/RipVIS-031.sqlite --frame-idx 0
```

```python
from p2.cache_io import VideoCacheReader

cache = VideoCacheReader("cache/dev/RipVIS-031.sqlite")
try:
    assert cache.complete
    frame = cache.read(0)
    probabilities = frame["soft_mask"]  # float16, (N, Hwork, Wwork)
    boxes = frame["boxes"]             # float32, (N, 4), original xyxy
    confidence = frame["conf"]         # float32, (N,)
    timestamp = frame["timestamp"]     # real PTS-derived seconds
    pts = frame["pts"]                 # original integer ticks
    candidate_indices = range(len(confidence))
finally:
    cache.close()
```

Read frames in `range(len(cache))` for the complete video. Zero-candidate frames remain rows and must not be removed from a temporal sequence. An instance with an empty `soft_mask > 0.5` is still a retained raw candidate. Do not silently filter or reorder arrays in a shared raw-cache adapter.

## Frozen T0 output

```bash
python reader/read_final_t0.py --p2-root reader --cache cache/dev/RipVIS-031.sqlite --frame-idx 0 --image-id 2904
```

The reader verifies pinned module hashes, frozen checkpoint identity and cache geometry. It applies `conf >= float32(0.10)`, per-instance working area, stable confidence order, and the exact pinned shared encoder. Output includes `video_id`, `frame_idx`, `timestamp`, `pts`, `image_id`, raw count and the ordered COCO prediction list. An optional `--output new_result.json` creates a new JSON file and refuses to overwrite an existing one.

Use the actual P1 COCO `image_id` for each annotated frame, from `upstream/data/manifests/frames_index.csv` by `(video_id, frame_idx)`. `2904` belongs only to the example shown. The full cache contains unannotated frames too; do not treat `frame_idx` as a COCO image ID. Keep every annotated image in the evaluator's image set, including those with zero predictions. Pass predictions to the fixed P5 implementation in the emitted order with `min_score=0.0`; do not re-sort equal `score=1.0` values or reapply confidence filtering in Python double precision.

## Optional v4 original-field restoration

```bash
python reader/restoration_v4/read_v4.py --cache cache/dev/RipVIS-097.sqlite --frame 822 --candidate 2 --output v4_example.json
```

This opt-in tool requires the existing compatible CPU PyTorch/torchvision, NumPy, SciPy, Ultralytics and OpenCV environment. Dependency versions are listed beside it. It does not load model weights, perform inference, access GT, or calculate an accuracy score. It restores one original candidate to limit memory and records its diagnostics. The command's output must be a new file. It may create a small `reader_revision_runtime/` settings directory beside that output.

The tool relocates its reference to the adjacent byte-identical `soft_masks_v3.py` only after checking all pinned hashes. The v4 module itself is unchanged from the qualified implementation. Normal v3 reconstructions are preserved; only the saturated-probability fallback can be replaced by the interval solve. Failed/unsupported solves retain an explicitly reported approximate fallback.

Read the diagnostics and the [validation limits](VALIDATION.md). A successful CLI invocation is not a new accuracy-gate PASS. This is not lossless reconstruction, not the default T0 path, and never an inverse for modified T1/TCA maps. The old `p2.soft_masks.restore_work_masks` entry point remains the historical v3 implementation with the recorded actual-video failure; it has not silently become v4.

## Installation and checks

Reader-only users can install `requirements-reader.txt` into their own environment. Full model reproduction uses `requirements-core.txt` plus the explicitly matched PyTorch 2.6.0 / torchvision 0.21.0 wheels; installing a different CUDA build is not evidence of identical production numerics. Test dependencies are in `requirements-test.txt`. The original validated production runtime is recorded in [release_summary.json](../reports/release_summary.json).

With the full compatible environment, the existing CPU unit tests can be run without model inference:

```bash
python -m pytest -p no:cacheprovider tests -q
```

The English source payload passed 140 existing tests. Its raw/T0 consumer smoke check read RipVIS-031 frame 0, preserving the SQLite hash, and returned two raw candidates and one retained T0 prediction. This is a publication portability check, not a repeat of the model-quality gates. See [publication_checks.json](../reports/publication_checks.json).

The pinned P1 source and metadata live under `upstream/`; they contain no dataset images, videos or polygon-label dataset. Historical training, validation, selection and export CLI tools are preserved for reproduction, but a cache-only ZIP does not include all their datasets, checkpoints or original gate reports. Those tools require explicit external inputs and the documented original-runtime conditions. Reading existing caches needs none of those historical experiment inputs.

Use `P2_PYTHON` to select an existing compatible interpreter for the portable shell runners, or invoke that interpreter directly. See the source CLI `--help` and [training/selection notes](TRAINING_SELECTION.md) before starting a new experiment.
