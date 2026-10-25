# File index

All paths below are relative to the release's `package2/` root. The GitHub source checkout and the common payload in both cache ZIPs share the small English files; large payloads are supplied through Drive only.

| Path | Purpose |
|---|---|
| `README.md` | Teammate quick start, download layout and current frozen result |
| `docs/READING.md` | Raw/T0/optional-v4 reader selection and usage |
| `docs/CACHE_SCHEMA.md` | SQLite v1 row, probability, transform and metadata contract |
| `docs/CONFIGURATION.md` | Frozen inference, NMS, T0 filtering/order and numeric boundary rules |
| `docs/TRAINING_SELECTION.md` | Completed formal training, 170-combination dev selection and pilot context |
| `docs/VALIDATION.md` | Separate gates, retained failures, timeline limits and FPS scope |
| `docs/PROVENANCE.md` | Original frozen source identity versus new team publication identity |
| `configs/frozen_release.json`, `configs/final.yaml` | New portable frozen-parameter descriptors; not the original freeze file bytes |
| `p2/`, `tools/`, `tests/`, `examples/` | P2 source, reproducible CLI tools and CPU tests |
| `upstream/` | Pinned small P1 source and manifests, including the annotated-frame/image-ID mapping; no dataset images, videos or polygon-label dataset |
| `configs/source_release_map.json` | Source hashes and explicit path-only publication changes |
| `reader/read_raw_cache.py` | Raw inspection without P5/Torch/model imports or filtering |
| `reader/read_final_t0.py` | Frozen T0 output using pinned shared P5 encoder, no inverse |
| `reader/restoration_v4/` | Explicit optional original-model-field restorer, adjacent pinned reference and English instructions |
| `requirements-reader.txt` | Reader/T0 dependencies |
| `requirements-raw.txt` | Minimal Python/NumPy raw-reader dependency |
| `requirements-core.txt`, `requirements-test.txt` | Full P2 and CPU test dependencies; PyTorch wheel choice documented separately |
| `run_python.sh`, `run_tests.sh` | Portable interpreter/root launchers |
| `reports/release_summary.json` | Completed P2 results summary |
| `reports/training_summary.json` | Compact formal training settings and epoch measurements |
| `reports/publication_checks.json` | 140 existing CPU unit tests and one raw/T0 portability smoke check, separate from model-quality gates |
| `reports/dev_selection_ranking.csv` | All 170 ranked checkpoint/confidence combinations, original bytes preserved |
| `reports/cache_inventory.csv` | Relative locations, per-video counts/sizes and unchanged cache/source hashes |
| `reports/evidence_index.json` | Hash references to retained original reports and evidence |
| `cache/dev/*.sqlite` | 17 full dev videos; present in the dev ZIP, absent from Git |
| `cache/final-val/*.sqlite` | 30 full final-val videos; present in the final-val ZIP, absent from Git |
| `weights/epoch5.pt` | The separately downloaded frozen checkpoint; absent from Git |
| `manifests/DEV_PACKAGE.json` | Generated inventory/checksums for the dev ZIP payload |
| `manifests/FINAL_VAL_PACKAGE.json` | Generated inventory/checksums for the final-val ZIP payload |

The ZIP's own hash belongs outside that ZIP to avoid a circular checksum. Standalone `WEIGHTS_SHA256.json` identifies the unchanged checkpoint.
