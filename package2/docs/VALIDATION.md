# Validation results and known limits

These results describe the completed frozen P2 delivery. Publication packaging, CPU reader checks and upload checks do not rerun or replace the model experiments below. Each PASS belongs only to its recorded path, sample and implementation.

## Separate gates

| Check | Input / sample | Result |
|---|---|---|
| Native soft-mask extraction | 200 dev JPEGs; 538 nonempty native instances plus 43 native-empty candidates | PASS; minimum binary IoU 1.0, required every-instance IoU >=0.99 |
| Original v3 JPEG codec roundtrip | 50 JPEG sample frames; 131 candidates | PASS; minimum IoU 0.994039824807; 4 approximate fallbacks |
| Original v3 actual-video inverse | 50 fixed actual dev MP4 frames; 149 candidates | **FAIL retained**; one failing instance, minimum IoU 0.970761298150; 21 fallbacks; production eligibility remains false |
| Optional v4 base50 regression | The same precommitted original 50 actual MP4 frames; 149 candidates, 135 native-nonempty | PASS; minimum IoU 0.993829960984; 9 interval solves and 12 remaining approximate fallbacks |
| Optional v4 holdout50 | Precommitted seed43 sample, disjoint from base50; 149 candidates, 138 native-nonempty | PASS; minimum IoU 0.993668272540; 6 interval solves and 6 remaining approximate fallbacks |
| Shared P4/P5 encode | Original base50 actual MP4 frames; 149 raw candidates, 135 nonempty working masks, 122 retained T0 predictions | Separate PASS; boxes/confidence/order, float16 bytes, geometry/NMS and T0 RLE lists agree exactly; 50 frozen-threshold comparisons, no failures |
| Complete structural delivery | All 47 videos, 52,697 rows, 134,173 candidates | PASS for content hashes, payload readback, PTS, frozen identity, source references and staged dev-before-val authorization |

The failed v3 instance is RipVIS-097, decoded frame 822, original candidate index 2. The independent v4 repair attained binary IoU 1.0 for that instance, but this does not rewrite the original v3 failure. Both v4 samples require **IoU >=0.98 for every raw candidate**, including native-empty candidates; no average or tolerance exception hides a failure.

v4 is a separately versioned, optional reader. It leaves the producer, raw cache bytes, model, NMS and T0/P5 scoring path unchanged. It fits quantization-consistent bilinear logits only for the v3 saturated-probability fallback, with rank/resource/feasibility checks. Unsuitable cases retain explicit approximate fallback. Missing-edge extrapolation, quantization and fallback remain limits; no lossless probability reconstruction or universal exact native mask recovery is claimed.

The actual v4 trials use qualification specification v2. A preliminary specification was superseded before any real trial solely to pin an updated validator source hash; sample plan, restoration algorithm and gate did not change. Both versions and the supersession record remain preserved in the sealed evidence. Native binary reference RLE and pixel hashes from both actual trials were archived separately, directly encoded without P5, enabling later CPU evidence review.

## What these results do not establish

- Earlier `diagnostic_only` CPU/GPU records remain diagnostic and are not relabelled as production evidence.
- Native extraction PASS is not proof of shared encode equivalence, and shared encode PASS is not proof of native-mask equality.
- Complete row/payload/PTS readback is not new independent inference of every full-video frame.
- The shared encoder can consume a suitably coordinated downstream map, but real T1 fusion and TCA-modified temporal outputs were not validated by these T0 tests.
- No inverse v2/v3/v4 should be applied to T1/TCA-modified fields under original-model transform assumptions.
- JPEG-based dev P/R/F2 cannot be assigned to the MP4 cache without a distinct evaluation.
- No final-val labels were evaluated or used to tune this release.

## Timeline and alignment

Every cache preserves actual integer PTS and `timestamp=(pts-first_pts)*time_base`. Frame index is the zero-based presentation-order decoded index, not a time unit. Annotation sampling rate (1 or 5 Hz) and video FPS (which varies) are distinct quantities.

RipVIS-100 retains its original strict source-audit **FAIL**. P1 arithmetic `t_sec` differs from actual PTS-derived time by up to **0.030055604235 seconds**. Its source integrity, actual PTS monotonicity and decoded-frame-count checks pass; cache timestamps use real PTS. P1 files and the FAIL were not edited. Downstream temporal implementations must explicitly account for this known P1-time versus cache-time distinction.

Seven limited source anchor checks remain ambiguous between adjacent or static frames, all at the sampled frame 0:

| Split | Videos |
|---|---|
| dev | RipVIS-116, RipVIS-NR-030 |
| final-val | RipVIS-015, RipVIS-053, RipVIS-NR-020, RipVIS-NR-022, RipVIS-NR-026 |

The anchor checks do not establish full-frame pixel equality between lossy annotated JPEGs and MP4 decode. Actual-cache validation used matching decoded MP4 pixels rather than transferring JPEG qualification to videos.

## Measured throughput

| Scope | Prediction + CPU transfer FPS | Export + full readback FPS |
|---|---:|---:|
| dev, 23,691 new frames / 17 videos | 44.027 | 19.514 |
| final-val, 29,006 new frames / 30 videos | 43.155 | 18.905 |
| combined, 52,697 new frames / 47 videos | 43.542 | 19.174 |

Prediction/transfer includes first-call warmup and CPU transfer; it is not pure GPU kernel throughput. Export time additionally includes sequential decoding, model initialization, SQLite writes and full readback. Pre-export source hashing, outer orchestration and independent qualification are excluded. Neither timer includes shared T0 `encode_mask` or P5 scoring, so these are not full T0-evaluation-pipeline FPS or a P6 benchmark. Exact measured values and coverage are in [release_summary.json](../reports/release_summary.json).

## Publication portability checks

The English source payload passed **140 existing CPU unit tests**. The raw and T0 consumer commands read RipVIS-031 frame 0 / image ID 2904 and returned **2 raw candidates / 1 T0 prediction**; the SQLite SHA256 was unchanged. No v4, model inference, GT loading or dependency installation was performed by these checks. [publication_checks.json](../reports/publication_checks.json) records their scope and source-log hashes. This finite reader check is separate from the retained native, inverse and shared-encoder gates above.

Original evidence hashes are listed in [evidence_index.json](../reports/evidence_index.json). Publication archive CRC/SHA checks, remote sizes and any remote download verification have their own release receipt; they are separate from these model-quality gates.
