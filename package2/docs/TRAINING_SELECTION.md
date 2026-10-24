# Training and completed dev selection

This is a summary of completed evidence, not an instruction to rerun training or change the frozen release.

## Data and formal training

The P1 split remains fixed: 95 training videos, 17 dev videos and 30 held-out final-val videos. Formal training loaded 12,195 train images and 2,421 dev images. The training YAML's `val` alias is the dev set, not final-val. Test data were not used for P2 selection.

Formal training started from original YOLO11n-seg pretrained weights (SHA256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`) with `resume=False`, 640 input, batch 8, maximum 30 epochs, patience 5, AMP, seed 42, `deterministic=True`, and a saved checkpoint each epoch. It stopped normally after 10 completed epochs under native early stopping. The run was not restarted for the subsequent diagnostics or publication.

The recorded formal-run monotonic duration was 2,729.802793 seconds, including final checkpoint hashing/persistence. This is not the same boundary as the per-epoch training and validation timers. Instrumentation counted 2,733 optimizer updates. The maximum training allocated peak was 1,474,313,216 bytes and reserved peak 2,057,306,112 bytes; these are allocator peaks, not whole-device VRAM. Runtime was Python 3.11.15, torch 2.6.0+cu124, Ultralytics 8.3.197, CUDA runtime 12.4 and RTX 4060 Ti under WSL2.

The native training `best.pt` corresponds to completed epoch 5 and was used for native early-stop fitness only. It is not the final P2 checkpoint. The final model is the original saved **`epoch5.pt`, completed epoch 6**, not a stripped or rewritten copy. Its SHA256 is `278ba795436c1299621b68a6f775a4af8b284afcab9c242384abff877a6906ca` and size is 17,628,845 bytes.

## Completed P5 dev selection

All 10 saved checkpoints were scored at 17 thresholds, 0.10 through 0.90 in steps of 0.05: **170 checkpoint/confidence combinations**, each covering the same 2,421 annotated dev JPEGs. P2 ranks full-precision F2 descending, then recall descending, precision descending, completed epoch ascending, and threshold ascending. This is not P4's temporal-parameter selection policy.

Selection used the confirmed Ngoc/P4 T0 path and fixed P5 evaluator commit `2fa7e07ab42cceb49ac149a391c05df0dcc63a1a`. Actual working float16 fields and verified work areas were reused with candidate identities/hashes checked; original-resolution binary RLE was not shrunk and upsampled to fake missing probabilities. Empty predictions remain part of the image set. Raw candidates keep original indices and float32 inclusive confidence semantics.

| Selected result | Value |
|---|---:|
| Completed epoch / checkpoint | 6 / `epoch5.pt` |
| T0 confidence | 0.10 |
| Precision | 0.3106972144387671 |
| Recall | 0.4915254237288136 |
| F2 | 0.4402765450553557 |
| TP / FP / FN | 1,885 / 4,182 / 1,950 |
| Predictions | 6,067 |
| Annotated frames | 2,421 |
| No-rip annotated frames / frame false alarms | 272 / 74 |

The complete 170-row ranking is [dev_selection_ranking.csv](../reports/dev_selection_ranking.csv), copied without changing its bytes. No new model selection or scoring was run to publish this handoff. User confirmation authorized the formal freeze of completed epoch 6 / confidence 0.10.

## JPEG selection versus full MP4 caches

The metrics above refer to annotated JPEG selection inputs. Full-video caches were generated from original decoded MP4 pixels using the same frozen checkpoint and production configuration. JPEG compression can change pixels and predictions, so identical prediction lists are not claimed across JPEG and MP4 inputs. Actual-cache checks compare cached outputs with new outputs from the **same decoded MP4 pixels**. No final-val labels were scored or used for tuning in this delivery.

Full 17-video dev caches were completed and qualified before the 30 final-val caches were exported. Native extraction, the old v3 failure, the new optional v4 samples, shared encode equality and complete structural readback have distinct records; see [VALIDATION.md](VALIDATION.md).

## Historical pilot

The original 320-image, two-epoch pilot produced only 24 optimizer updates. Checked pre-NMS confidences were below the fixed 0.1 threshold, supporting an undertrained pilot rather than a demonstrated export defect. Its empty predictions and failed historical gates remain preserved. The first useful integration handoff later used `epoch0.pt` / CPU / v2 on three dev videos (998 frames, 3,156 candidates), explicitly as a temporary model. Neither pilot is the frozen formal release.

All historical checkpoints, original logs, detailed per-candidate diagnostics and Chinese records remain in the sealed local evidence. They are not mixed into the essential teammate download bundles. The release includes compact training and validation summaries and their original source hashes; supplemental evidence can be distributed separately.
