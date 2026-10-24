# SQLite v1 cache schema

The publication retains the existing **`p2-video-cache-sqlite-v1`** format and original SQLite bytes. It adds English documentation and new archive checksums; it does not migrate the cache format or change the frozen model.

Each video has one SQLite file and every decoded frame has a row, including zero-candidate frames. `frame_idx` is contiguous from 0 in presentation-order decoding. Original integer `pts` and exact rational `time_base` are retained; `timestamp=(pts-first_pts)*time_base` is real relative presentation time in seconds. Annotation sampling at 1 or 5 Hz and a nominal/average video FPS do not synthesize this timeline.

## Frame payload

| SQLite column | Representation |
|---|---|
| `frame_idx` | Integer primary key, zero-based decoded frame index |
| `timestamp` | Finite, strictly increasing relative PTS time; first frame 0 |
| `pts` | Original integer presentation timestamp ticks |
| `instance_count` | Number of retained raw NMS candidates |
| `boxes` | Little-endian float32 bytes, shape `(N,4)`, original-image `xyxy` |
| `confidences` | Little-endian float32 bytes, shape `(N,)` |
| `soft_masks` | zlib level-1 compressed little-endian float16 bytes, shape `(N,Hwork,Wwork)` |
| `raw_sha256` | SHA256 of uncompressed box, confidence and probability bytes concatenated in that order |

The Python reader returns keys `boxes`, `conf`, `soft_mask`, `frame_idx`, `timestamp` and `pts`. Candidate index is the corresponding row position from 0 to N-1. All predictions have P2 class 0. Zero-candidate arrays have shapes `(0,4)`, `(0,)` and `(0,Hwork,Wwork)`.

The stored candidate pool is `confidence >= float32(0.1)` after the fixed NMS. Its strict library comparison is compensated by one float32 ULP followed by the explicit inclusive comparison. Original NMS order is retained. There is no raw area filter, component splitting or empty-mask deletion. Do not silently remove an instance simply because `soft_mask > 0.5` is empty.

## Probability generation and geometry

The working image is unpadded and preserves original aspect ratio, with long side 640 and the other side rounded from the original dimensions. The existing v3 transform metadata records prototype dimensions, crop, box scaling and interpolation geometry.

1. Float32 mask coefficients multiply float32 prototypes to form logits.
2. Prototype-space padding is removed using original aspect ratio.
3. Logits are bilinearly interpolated to working size with `align_corners=False`.
4. Sigmoid is applied, boxes are scaled to working coordinates for cropping, and probabilities are stored as float16. Outside-box probabilities are zero. The recorded one-ULP rule preserves positive-logit `>0.5` decisions through float16 storage.

These are pre-binarization probabilities, not binary masks cast to floating point. Original and working dimensions are `(height,width)`; box coordinates remain original-image `xyxy`. Read the stored transform rather than assuming a particular padding or aspect ratio.

## Metadata and relocation

The `metadata` table stores JSON `spec`, a canonical-JSON SHA256 `signature`, and a separate completion flag. It includes original/working dimensions and transforms, source video ID/split/hash, expected decoded count, original time base and first PTS, P1 manifest hashes, checkpoint and implementation hashes, inference/NMS configuration, CUDA runtime and frozen gate provenance.

The original absolute paths within that immutable metadata are historical producer locations. They are not the path a consumer must open. Pass the relocated SQLite path to `VideoCacheReader`; retain the metadata and signature unchanged. The new `reports/cache_inventory.csv` maps each video to a relative release path and carries the unchanged SQLite SHA256.

## Reading and integrity

Use the [raw and T0 reader examples](READING.md). The raw reader uses only NumPy and standard-library SQLite/zlib; T0 additionally uses pinned P5 dependencies. Both default readers bypass all inverse restoration.

The original export used sequential PyAV decoding and bounded memory, SQLite DELETE journaling and `synchronous=FULL`, with at most 16 new rows per transaction. Resume verifies source/model/config identity, contiguous prefix rows, PTS and the last payload before predicting new frames. Full completion requires decoded-count equality, SQLite integrity, and every payload digest/timestamp read back. These production checks were followed by a separate complete 47-video readback; publication additionally verifies archive members and checksums.

The `raw_sha256` protects row payload bytes. It does not itself prove model accuracy. Whole-file SHA256 protects the supplied SQLite bytes, including metadata. The shared encode comparison, native extraction and inverse quality gates are separate evidence, described in [VALIDATION.md](VALIDATION.md).

## Optional inverse and known limits

The historical `p2.soft_masks.restore_work_masks` is v3 and has a retained actual-MP4 validation failure. Do not treat its earlier JPEG codec PASS as a guarantee for this full-video cache. The optional published v4 entry point is `reader/restoration_v4/read_v4.py`; its base50 and disjoint holdout50 satisfy the unchanged per-instance IoU >=0.98 gate, with explicit approximate fallbacks.

No inverse is lossless or universally exact. It applies only to untouched original-model working fields with their original boxes and geometry. It does not apply to T1/TCA-modified maps. T0 instead uses working binarization, per-instance area filtering and the shared P5 `encode_mask` directly.

RipVIS-100 retains its original strict source-audit FAIL because P1 arithmetic times differ from actual PTS by up to 0.030055604235 seconds. Its source integrity, decoded count and actual PTS checks pass. The cache uses real PTS; neither P1 nor the audit FAIL was rewritten. Seven limited anchor checks remain ambiguous between adjacent/static frames. These known limits are retained rather than hidden by the structural cache PASS.
