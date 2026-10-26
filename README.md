# Temporal post-processing for rip current segmentation

A YOLO instance segmentation model finds rip currents in each frame of a beach video
(RipVIS dataset). Frame-by-frame output flickers: rips appear and disappear between
neighbouring frames. This project tests whether causal temporal post-processing makes
the video output better, by comparing three methods on the same model output:

- **T0**: per-frame baseline. Keep candidates above a confidence threshold, binarise each mask at 0.5.
- **T1**: causal sliding-window average. Average the merged soft masks over the last `w` seconds, threshold at θ.
- **T2**: TCA (temporal consistency aggregation). Match candidates to tracks by mask IoU, accumulate
  per-track confidence over time, and use hysteresis and presence/absence durations to switch tracks on and off.

All three methods read the same cached model output (`conf >= 0.1`), are tuned on 17 dev
videos, frozen, and then scored once on 30 val videos with the official RipVIS matching rule (IoU >= 0.5).

## Results

Val split, 30 videos, 3,784 ground-truth rip instances:

| Method | Precision | Recall | F2 | False-alarm rate | Fragments / min | End-to-end FPS |
|---|---:|---:|---:|---:|---:|---:|
| T0 | 0.308 | 0.489 | **0.438** | 0.400 | 3.39 | 29.2 |
| T1 | 0.329 | 0.455 | 0.423 | 0.458 | 2.75 | 27.9 |
| T2 | 0.215 | 0.496 | 0.393 | 0.356 | **1.16** | 16.8 |

- Neither temporal method improves frame-level F2. Paired bootstrap over videos (1,000 resamples, 95% CI):
  T1 − T0 = −0.015 [−0.064, 0.012], T2 − T0 = −0.045 [−0.081, −0.004].
- T2 gives the most stable output: on the 12 single-rip val videos, once the rip has been found it drops out
  far less often (1.16 vs 3.39 gaps per minute). The cost is more false positive masks (6,867 vs 4,163)
  and a slower pipeline (16.8 vs 29.2 FPS).
- Fragmentation is measured on the single-rip videos only. FPS includes decoding, the model and
  post-processing, measured on one Apple Silicon machine (`mps`).

Full tables, per-video and per-frame counts: [report/logs/final_val](report/logs/final_val/README.md).

## Repository layout

| Path | Contents | Documentation |
|---|---|---|
| `data/` | Dataset manifests: video list, dev/val split, frame-to-image-ID index | [data/README.md](data/README.md) |
| `package2/` | Model training, the frozen checkpoint's video caches and their readers | [package2/README.md](package2/README.md) |
| `temporal/` | T0, T1, T2 (`tca.py`), the dev search space and selection rule | [temporal/README_package4.md](temporal/README_package4.md), [report/tca_spec.md](report/tca_spec.md) |
| `eval/` | Evaluator (official RipVIS rule), paired bootstrap, fragmentation and delay | [eval/README.md](eval/README.md) |
| `configs/` | Dev search space and the three frozen configurations (`configs/frozen/`) | |
| `scripts/` | Command-line entry points, listed below | |
| `report/logs/` | Saved outputs: dev search, runtime, final val results, evaluator agreement checks | |
| `tests/` | Unit and end-to-end tests | |

## Setup

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt       # then install torch for your machine
copy configs\paths.example.yaml configs\paths.yaml   # set ripvis_root and cache_root
pytest
```

## Running the pipeline

The committed manifests and frozen configs let you start at any step. Every command runs from the repo root.

**1. Data.** Download RipVIS to `ripvis_root`. The manifests in `data/manifests/` are already built;
`python -m data.build_manifests --help` rebuilds them.
```
python -m scripts.download_ripvis base
python -m scripts.download_ripvis devvideos
```

**2. Model output.** The frozen checkpoint and one SQLite cache per video (17 dev, 30 val) are released
separately, not in Git. See [package2/DOWNLOADS.md](package2/DOWNLOADS.md) and extract them to a local SSD.
Training and checkpoint selection are documented in `package2/docs/`.

**3. Dev search** (already done; the result is in `configs/frozen/`). Scores 17 T0, 76 T1 and 20 T2 configurations on dev,
then freezes the best of each. `select_configs.py` refuses to overwrite existing frozen configs.
```
python scripts/tune_dev.py --cache-dir <package2>/cache/dev --p2-reader <package2>/reader --out-dir runs/dev_search
python scripts/select_configs.py --search-csv runs/dev_search/dev_search.csv
```

**4. Run the frozen methods on val.** Writes `val_T0.json`, `val_T1.json`, `val_T2.json` (COCO results).
```
python scripts/run_frozen.py --split val --cache-dir <package2>/cache/final-val --p2-reader <package2>/reader --out-dir runs/val_frozen
```

**5. Runtime.** Needs the val MP4s and the checkpoint. Set `--device` to `cuda`, `mps` or `cpu`.
```
python scripts/measure_fps.py --videos-dir <val videos> --p2-root <package2> --device cuda --out-dir runs/fps_val
```

**6. Score.** Results table, paired bootstrap, per-video and per-frame counts.
```
python -m scripts.final_scoring --split val --pred T0=runs/val_frozen/val_T0.json --pred T1=runs/val_frozen/val_T1.json --pred T2=runs/val_frozen/val_T2.json --fps T0=29.2 --fps T1=27.9 --fps T2=16.8 --out runs/final_val
```
The committed results in `report/logs/final_val` came from this command. To check that the evaluator gives exactly
the same P/R/F1/F2 as the official `compute_pr_f1_f2.py` on these predictions:
```
python -m scripts.check_agreement --split val --pred T0=runs/val_frozen/val_T0.json --pred T1=runs/val_frozen/val_T1.json --pred T2=runs/val_frozen/val_T2.json --log runs/agreement_final_val.json
```

## Data and licence

RipVIS v1.8.4, CC BY-NC 4.0. The licence forbids redistribution, so no images, videos or annotations
are committed; only the manifests in `data/manifests/`.
