# Evaluator (Package 5)

`evaluator.py` scores predictions with the official RipVIS rule (`compute_pr_f1_f2.py`) and
gives identical P/R/F1/F2 (see `report/logs/agreement_pilot.json`), plus per-frame TP/FP/FN.

```python
from eval.evaluator import load_gt, encode_mask, evaluate, summarise

gt = load_gt("dev")                     # load once, reuse for every config
h, w = gt.shapes[image_id]              # original resolution
preds = [{"image_id": image_id, "category_id": 1, "score": 1.0,
          "segmentation": encode_mask(mask, h, w)}]   # mask: final binary mask, any resolution
per_frame = evaluate(preds, gt)         # one row per annotated frame: tp, fp, fn, ...
print(summarise(per_frame))             # precision, recall, f1, f2, false_alarm_rate
```

Command line: `python -m eval.evaluator --split dev --pred preds.json`

- `image_id`: join the cache to `data/manifests/frames_index.csv` on `(video_id, frame_idx)`.
- Upsampling: every condition (T0, T1, T2) passes its final mask through `encode_mask()`.
- Order matters: within a frame, predictions are matched in list order, not by score.
  T0 should list instances by descending conf (the cache's NMS order).
- Threshold sweep (P2): build preds once at conf 0.1 with `score=conf`, then call
  `evaluate(preds, gt, min_score=thr)` for each threshold.
- Each GT annotation is one instance, so a rip drawn as several annotations counts several times.
- Single-rip temporal metrics: `temporal_frames(per_frame, video_id)` feeds
  `eval.temporal_metrics.track_stats`.
