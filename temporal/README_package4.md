# Package 4 — T0, T1 and dev tuning

## Files
| Path | Purpose |
|---|---|
| `temporal/cache_io.py` | Cache data structures; reads P2's `<video_id>.sqlite` lazily through P2's own reader, and test `.pkl` caches |
| `temporal/common.py` | Shared 0.1 % area filter, upsampling to original resolution, COCO/RLE encoding |
| `temporal/t0.py` | T0: confidence threshold + soft mask > 0.5 |
| `temporal/t1.py` | T1: pixel-wise max merge, causal (t − w, t] window mean, threshold, 8-connected components |
| `temporal/t2_adapter.py` | T2: feeds P3's `run_tca` the same candidate pool, then applies the shared area filter and upsampling |
| `temporal/search.py` | 17 T0 thresholds, 76 T1 configs and 20 seeded T2 configs from `configs/search_space.yaml` |
| `temporal/selection.py` | Selection rule: T0 max F2 sets R0; T1/T2 recall ≥ R0, max F2, tie → recall → shorter window → lower θ → config_id; fallback flagged |
| `temporal/scoring.py` | Thin layer over P5's `eval/evaluator.py`: per-video GT, per-video counts, pooled metrics (P5 conventions) |
| `eval/interim_scoring.py` | Old interim scorer, kept only for its file-name parser and unit tests |
| `scripts/tune_dev.py` | Runs every config on the dev videos, one process per video, writes CSVs and the machine's CPU/RAM |
| `scripts/select_configs.py` | Applies the rule to the CSV and freezes `configs/frozen/{t0,t1,t2}.yaml` |
| `scripts/make_synthetic_cache.py` | Synthetic videos + COCO GT for testing before the real cache exists |
| `tests/` | Unit and end-to-end tests (`fake_tca.py` mimics P3's interface, it is not TCA) |

## Run
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/tune_dev.py --cache-dir P2_CACHE_DIR --p2-reader P2_CACHE_DIR/reader \
    --ripvis-root RIPVIS_ROOT --manifest data/manifests/videos.csv \
    --out-dir runs/dev_search          # T0 threshold is tuned on the grid; run_tca from temporal/tca.py
.venv/bin/python scripts/select_configs.py --search-csv runs/dev_search/dev_search.csv
```

## Decisions (Ash, 2026-10-23)
- Every condition gets the same candidate pool (conf >= 0.1) and tunes its own decision threshold. T1 and T2 merge all candidates; T0 tunes `conf_thr` on dev.
- Ties: highest F2 among configs with recall >= R0, then higher recall, then shorter window, then lower θ.
- Tuning and final scoring run on one named machine with the cache copied to local SSD; `dev_search_meta.json` records CPU and RAM.

## Still to confirm
1. Cache: P2's SQLite schema `p2-video-cache-sqlite-v1` (full working canvas, long side 640, no padding, PTS timestamps). Checked end to end on P2's 3-video pilot; the full-cache format is not final yet.
2. TCA: tested with P3's `run_tca` from branch `package3-tca` on the pilot (113 configs, 3 videos, ~2 min, 10 cores). It returns uncompressed RLE at working resolution and does not apply the 0.1 % filter, so `t2_adapter.py` does both. The runner expects it at `temporal/tca.py` once P3 is merged.
3. Scoring uses P5's evaluator (`load_gt("dev")`, `evaluate`, `encode_mask`); `image_id` comes from `data/manifests/frames_index.csv`. On the pilot, T0 at conf 0.1 without the area filter gives exactly P5's agreement log (TP 44, FP 445, FN 126, false-alarm rate 0.978).
4. Output order: T0 lists instances by descending conf (P5 matches in list order); T1 lists components in label order; T2 in track order.
