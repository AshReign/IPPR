# Data manifests (Package 1)

Source: RipVIS v1.8.4 (Dumitriu et al., 2025a), CC BY-NC 4.0. Dataset files are never committed;
only the manifests in `data/manifests/` are.

## How to rebuild

```
python -m scripts.download_ripvis base
python -m data.build_manifests inventory
python -m data.build_manifests split          # frozen after the first run
python -m scripts.download_ripvis devvideos
python -m data.build_manifests meta
python -m data.build_manifests finalize
python -m data.build_manifests yolo-local     # each machine, before training
```

## Split decisions

- Seed: 42
- Dev: 17 whole training videos (7 multi-rip, 7 single-rip, 3 no-rip), none from the frame-mismatch list.
- Final val: official val minus the frame-mismatch videos (RipVIS-001, 007, 012, 014, 024, 039).
- Frame-mismatch list (RipVIS Readme, Known Limitation 2): 001, 003, 004, 006, 007, 011, 012, 014, 016,
  017, 018, 020, 021, 022, 024, 033, 034, 035, 036, 037, 039, 040.
- Additional 2,466 images: not used.

## Final counts

| Set | Videos (rip / no-rip) | Annotated frames | Masks |
|---|---|---|---|
| Train | TODO | TODO | TODO |
| Dev | TODO | TODO | TODO |
| Final val | TODO | TODO | TODO |
| Single-rip val | TODO | TODO | TODO |

## Dev videos

TODO: paste `dev_videos.txt`.

## Notes

- Same-event overlaps between dev and train (from the dataset info PDF): TODO or "none found".
- Disagreements between `max_rips_per_frame` and the documented rip counts: TODO or "none found".
- Videos with an annotated frame past the video end (`last_frame_ok = False`): TODO or "none".

## Files

| File | One row per | Used by |
|---|---|---|
| `videos.csv` | video (split, rip flags, fps, size, sampling rate) | everyone |
| `frames_all.csv` | annotated image (video, frame number, number of masks) | everyone |
| `dev_videos.txt`, `val_videos.txt` | video ID | model, tuning, evaluation |
| `yolo_train_images.txt`, `yolo_dev_images.txt` | image path relative to `ripvis_root` | model |
| `single_rip_val.txt` | video ID | temporal metrics |
| `frames_index.csv` | annotated dev/val frame, with `t_sec` | temporal methods, evaluation |
