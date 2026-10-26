# Final val results (Package 5)

The one scoring run on the 30 val videos, after the T0/T1/T2 configs were frozen on dev.

Produced by:

```
python -m scripts.final_scoring --split val \
    --pred T0=val_T0.json --pred T1=val_T1.json --pred T2=val_T2.json \
    --fps T0=29.2 --fps T1=27.9 --fps T2=16.8 --out report/logs/final_val
```

- Predictions: Package 4 `scripts/run_frozen.py` (commit `051eb6c`), frozen configs `configs/frozen/t0|t1|t2.yaml`.
  The prediction files are not in the repo; their SHA-256 hashes are in `run_info.json`.
- FPS: end-to-end, from Package 4 `scripts/measure_fps.py` (`report/logs/fps_val`).
- Matching: the official `compute_pr_f1_f2.py` rule, IoU >= 0.5. The official script gives identical
  P/R/F1/F2 on all three files (`../agreement_final_val.json`).

| File | Contents |
|---|---|
| `results.csv`, `results.md` | One row per condition: P, R, F2, false-alarm rate, fragmentation per minute, median delay (detected / single-rip tracks), FPS, TP/FP/FN |
| `bootstrap.csv` | Paired bootstrap of the F2 difference (T1-T0, T2-T0, T2-T1): 1,000 resamples of val videos, seed 42, 95% percentile interval |
| `per_video.csv` | TP/FP/FN and F2 per video and condition |
| `per_frame_T0/T1/T2.csv` | TP/FP/FN per annotated frame |
| `run_info.json` | Code commit, prediction file hashes, seed, single-rip videos used for fragmentation and delay |

The ground-truth annotations are not included (the RipVIS licence forbids redistribution).
