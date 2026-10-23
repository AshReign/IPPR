# Package 4 — T0, T1 and dev tuning

## Files
| Path | Purpose |
|---|---|
| `temporal/cache_io.py` | Cache data structures; reads P2's `<video_id>.sqlite` lazily through P2's own reader, and test `.pkl` caches |
| `temporal/common.py` | Shared 0.1 % area filter, upsampling to original resolution, COCO/RLE encoding |
| `temporal/t0.py` | T0: confidence threshold + soft mask > 0.5 |
| `temporal/t1.py` | T1: pixel-wise max merge, causal (t − w, t] window mean, threshold, 8-connected components |
| `temporal/search.py` | 76 T1 configs and 20 seeded T2 configs from `configs/search_space.yaml` |
| `temporal/selection.py` | Selection rule (recall ≥ R0, max F2, tie → recall → config_id; fallback flagged) |
| `eval/interim_scoring.py` | **Interim** evaluator (official matching rule) until Package 5's evaluator is ready |
| `scripts/tune_dev.py` | Runs every config on the dev videos, one process per video, writes CSVs |
| `scripts/select_configs.py` | Applies the rule to the CSV and freezes `configs/frozen/{t0,t1,t2}.yaml` |
| `scripts/make_synthetic_cache.py` | Synthetic videos + COCO GT for testing before the real cache exists |
| `tests/` | 21 unit and end-to-end tests (`fake_tca.py` is a stand-in, not TCA) |

## Run
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/tune_dev.py --cache-dir P2_CACHE_DIR --p2-reader P2_CACHE_DIR/reader \
    --gt-json RIPVIS/train/coco_annotations/train.json --manifest data/manifests/videos.csv \
    --t0-conf T0_THRESHOLD --out-dir runs/dev_search          # add --conditions T0 T1 before TCA exists
.venv/bin/python scripts/select_configs.py --search-csv runs/dev_search/dev_search.csv
```

## Assumptions to confirm with the team
1. Cache: P2's SQLite schema `p2-video-cache-sqlite-v1` (full working canvas, long side 640, no padding, PTS timestamps). Checked end to end on P2's 3-video pilot.
5. All conditions are upsampled the same way (binary mask at working resolution -> bilinear -> 0.5). P2's v2 inverse is only valid for raw model probabilities, so P5 must confirm how T0 is scored.
2. `run_tca(video_cache, params)` lives in `temporal/tca.py` and accepts the parameter names in `configs/search_space.yaml` (Package 3).
3. `eval/interim_scoring.py` must give the same TP/FP/FN as Package 5's evaluator before the search is frozen.
4. T1 merges all cached instances (pre-filter 0.1) without the T0 confidence threshold, as written in the task package.
