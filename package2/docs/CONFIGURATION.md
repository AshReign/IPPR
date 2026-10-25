# Frozen configuration

`configs/frozen_release.json` and `configs/final.yaml` are portable **publication descriptors**, not byte-for-byte copies of the original sealed freeze artifact. They describe the same approved model and processing rules while using replaceable local paths. Their release hashes must not be substituted for the original freeze SHA256. See [provenance](PROVENANCE.md).

## Model and inference

| Setting | Frozen value |
|---|---|
| Model | YOLO11n-seg, `epoch5.pt`, completed epoch 6 |
| Checkpoint SHA256 | `278ba795436c1299621b68a6f775a4af8b284afcab9c242384abff877a6906ca` |
| Model input | `imgsz=640`, square letterbox, `rect=False` |
| Inference | batch 1, float32, `half=False`, `retina_masks=True`, augmentation off |
| Candidate prefilter | inclusive `conf >= float32(0.1)` |
| NMS | torchvision NMS, IoU 0.7, max 300 detections, class-aware, `multi_label=False` |
| Other NMS bounds | `max_nms=30000`, `max_wh=7680`, `max_time_img=30.0`, non-rotated, non-end-to-end |
| Stored probabilities | full unpadded working frame, long side 640, float16 |
| T0 confidence | inclusive float32 `0.10` |

The exact recorded CUDA runtime uses torch 2.6.0+cu124, CUDA runtime 12.4, cuDNN 90100, Ultralytics 8.3.197 and RTX 4060 Ti. Matmul TF32 is false, cuDNN TF32 true, cuDNN deterministic false, cuDNN benchmark false. These inference settings are separate from the original training configuration, where AMP and `deterministic=True` were used. Reproducing the configuration on another device does not automatically reproduce the original numerical qualification.

## Raw storage versus final T0

Raw caches keep every NMS candidate in the common inclusive 0.1 pool, including native/working empty-mask candidates and zero-candidate frames. They apply no area, component or final-output filter. Candidate indices are positions in aligned arrays and remain the original NMS order.

The frozen T0 path is:

1. Compare candidate confidence with the threshold in float32, including equality.
2. Binarize each whole working instance with `probability > 0.5`.
3. Retain an instance when `1000 * working_binary_area >= work_height * work_width`.
4. Keep disconnected regions as one prediction. Do not split or merge T0 model instances.
5. Sort by descending float32 confidence; equal values use original candidate index ascending.
6. Call pinned P5 `encode_mask`: `>0.5` to binary float32, OpenCV `INTER_LINEAR` to original `(width, height)`, `>=0.5` to uint8, then Fortran-layout compressed COCO RLE.
7. Map P2 class 0 to COCO category 1. Write COCO `score=1.0` after ordering. Preserve `instance_conf` and `candidate_index` for audit. Evaluate using P5 `min_score=0.0`, IoU threshold 0.5, and the same list order.

The accepted Ngoc/P4 implementation applies the 0.1% area filter at **working resolution**. Original-resolution area filtering is not assumed exactly equivalent and must not be silently substituted. The default T0 reader never invokes the optional v4 inverse.

## Numeric boundary and selection policy

The P2 selection threshold comparison remains float32 inclusive equality. In a prior synthetic check at float32 0.70 (`0.699999988079071`), P2 retained an equal candidate whereas the inspected P4 Python-double comparison did not. No candidate in the checked 66,274-candidate dev pool lay on those exact boundary values, and 115 actual float16 samples / 340 candidates matched the pinned P4 T0 COCO lists at all 17 thresholds. This finite-sample agreement is not proof that the two numeric comparisons are universally identical. P4 itself was not modified.

The completed P2 selection used confidence 0.10 through 0.90 in steps of 0.05. Rank by full-precision F2 descending, recall descending, precision descending, completed epoch ascending, then threshold ascending. These are P2 checkpoint/confidence rules, not the P4 temporal-configuration selection policy. `configs/selection_protocol.json` and `configs/selection_p4_protocol.json` preserve the historical protocol snapshots; an old status field in those files is not the current release status.

## Time and metadata

Use `timestamp=(pts-first_pts)*time_base` from the original decoded video. `frame_idx` is a zero-based presentation-order decoded index. Metadata dimensions use `(height, width)`; OpenCV resize receives `(width, height)`. `frames_index.csv` contains annotated-frame mappings only and is not a replacement for the full cache timeline.

Original producer paths inside immutable SQLite metadata are provenance strings. Consumers supply the actual local SQLite path; they must not rewrite metadata or its signature when moving a cache. The publication folder name `final-val` does not change the stored split value `val`.
