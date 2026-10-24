# Package 2: frozen model and full-video caches

Release: **IPPR_P2_Formal_EN_20261024_r1**. This is the English teammate handoff of the completed P2 delivery. The frozen model, inference settings and SQLite bytes have not changed for publication.

**Use `epoch5.pt` (completed epoch 6) and T0 confidence `0.10`.** The dev selection used 10 checkpoints × 17 thresholds on all 2,421 annotated dev JPEGs: **P = 0.310697214, R = 0.491525424, F2 = 0.440276545**. These are JPEG selection metrics, not fresh scores of the MP4 caches or final-val.

## Download and layout

Download the **dev package first** for P3/P4 integration, then final-val when required. Download to a **local SSD** and extract before reading; do not run directly from a mounted Google Drive folder or inside a ZIP.

| Item | Contents |
|---|---|
| `IPPR_P2_Formal_EN_20261024_r1_dev.zip` | 17 full dev caches plus the common English source, reader, configuration and summary files |
| `IPPR_P2_Formal_EN_20261024_r1_final-val.zip` | 30 full final-val caches plus the same common files |
| `epoch5.pt` | The single frozen checkpoint; place it at `package2/weights/epoch5.pt` if model execution is needed |
| `WEIGHTS_SHA256.json` | Checkpoint size and SHA256 |

Each ZIP extracts into `package2/`. Extract both into the same parent directory. The common files are identical; split manifests use separate names. The frozen SQLite `split` value remains `val`, while the release directory is named `final-val` for clarity.

```text
package2/
  README.md
  cache/dev/RipVIS-031.sqlite
  cache/final-val/...
  weights/epoch5.pt                 # optional for cache consumers
  reader/read_raw_cache.py
  reader/read_final_t0.py
  reader/restoration_v4/read_v4.py   # optional; not the T0 path
  configs/frozen_release.json
  configs/final.yaml
  manifests/DEV_PACKAGE.json
  manifests/FINAL_VAL_PACKAGE.json
  reports/cache_inventory.csv
```

Git contains small source, tests, documentation, frozen descriptors and summaries. The selected checkpoint and SQLite caches are distributed through the existing shared Drive folder, not Git. Dataset images, source videos and polygon labels are not included in these handoff packages. Compare download size and SHA256 with the release checksums before use. Archive integrity and remote download verification are separate checks.

## Start with the default readers

The commands below run from the extracted `package2/` directory, using a Python environment with the listed dependencies. Paths are relative and can be replaced with your local cache root. No GPU, checkpoint or model inference is needed to read existing caches.

```bash
# Install reader dependencies in your own environment if needed.
python -m pip install -r requirements-reader.txt

# Inspect raw arrays and timestamp metadata; do not filter candidates.
python reader/read_raw_cache.py --cache cache/dev/RipVIS-031.sqlite --frame-idx 0

# Apply the frozen T0 output path, including the shared P5 encode_mask.
python reader/read_final_t0.py --p2-root reader --cache cache/dev/RipVIS-031.sqlite --frame-idx 0 --image-id 2904
```

`image_id=2904` is the P1 dev annotation mapping for this specific example. For other annotated frames, use `upstream/data/manifests/frames_index.csv` to map `(video_id, frame_idx)` to the real COCO image ID. Full-video frames do not all have annotation IDs. Do not invent IDs for evaluation or omit annotated frames with no predictions.

**The raw and T0 readers do not call v4 or the legacy inverse.** The raw reader requires only Python/NumPy; the T0 reader additionally uses pinned OpenCV, pandas and pycocotools through P5. The optional v4 tool restores only untouched original-model working probability fields. It is not lossless and must not be used for T1/TCA-modified maps. See [reader choices](docs/READING.md).

## Frozen cache and T0 rules

- Cache schema remains **`p2-video-cache-sqlite-v1`**: one file per full video, including zero-candidate frames.
- Raw pool: **`conf >= float32(0.1)`** after the fixed NMS; aligned original-image `xyxy` boxes, float32 confidence and full unpadded float16 working probabilities. Empty working masks and original candidate order are retained. No area filter is applied to stored raw candidates.
- T0: confidence filtering uses float32 inclusive equality; binarize each whole instance at `> 0.5`; retain working area when `1000 × area >= Hwork × Wwork`; do not split disconnected regions. Sort by descending float32 confidence, then original candidate index on ties.
- Encode through pinned P5: binary float32 → bilinear resize to original `(width, height)` → `>= 0.5` uint8 → Fortran-order COCO RLE. COCO `score=1.0` is written after confidence ordering; retain list order when evaluating with P5 `min_score=0`, IoU threshold `0.5`.
- Use actual PTS-derived `timestamp` for temporal processing. Neither a fixed FPS nor annotation sampling rate defines the full-video timeline.

## Coverage and separate validation results

| Split | Videos | Full frames | Raw candidates | SQLite bytes |
|---|---:|---:|---:|---:|
| dev | 17 | 23,691 | 74,419 | 1,419,472,896 |
| final-val | 30 | 29,006 | 59,754 | 1,867,612,160 |
| Total | 47 | 52,697 | 134,173 | 3,287,085,056 |

The total SQLite size is **3.061 GiB**, excluding weights and common files. Every row/payload, source hash, timeline and frozen identity passed full readback verification. That structural result is not fresh independent inference of all 52,697 frames.

| Validation path | Recorded result and scope |
|---|---|
| Native extraction | PASS: 200 dev JPEGs, 538 nonempty native instances, minimum IoU 1.0; 43 native-empty candidates also checked |
| Original v3 inverse on actual MP4 cache | **FAIL retained**: 1 of 149 candidates in 50 dev frames, minimum IoU 0.970761298150 |
| Optional v4 inverse, original base50 | PASS: 149 candidates, minimum IoU 0.993829960984; 12 approximate fallbacks remain |
| Optional v4 inverse, disjoint holdout50 | PASS: 149 candidates, minimum IoU 0.993668272540; 6 approximate fallbacks remain |
| Shared P4/P5 encode path | Separate PASS on 50 actual dev MP4 frames / 149 raw candidates; exact working-byte, identity, order and T0-output equality; 122 final T0 predictions |

The v4 gate remains **IoU >= 0.98 for every candidate**, without mean-based exceptions. It does not establish universal native recovery or validate real T1/TCA temporal outputs. Source audit warnings are also retained: RipVIS-100 has a maximum P1 arithmetic-time versus actual-PTS difference of about **0.030055604235 seconds**, and seven source anchors are ambiguous between adjacent/static frames. See [validation and known issues](docs/VALIDATION.md).

## Documentation and provenance

- [Reading and dependency choices](docs/READING.md)
- [Current SQLite schema](docs/CACHE_SCHEMA.md)
- [Frozen inference and output configuration](docs/CONFIGURATION.md)
- [Training and completed dev selection](docs/TRAINING_SELECTION.md)
- [Validation scope and known issues](docs/VALIDATION.md)
- [Original freeze versus team publication version](docs/PROVENANCE.md)
- [File index](docs/FILE_INDEX.md)
- [Compact machine-readable summary](reports/release_summary.json), [170-row dev ranking](reports/dev_selection_ranking.csv), [47-cache inventory](reports/cache_inventory.csv)

The original local frozen source snapshot commit is `c221d204dae9d24188c72dd0fa7bd8e8e7313c94`. It is not the new team GitHub publication commit. The latter is identified by the team checkout's `git rev-parse HEAD`.
