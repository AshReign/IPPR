# Optional v4 restoration reader

The normal entry points are `reader/read_raw_cache.py` and
`reader/read_final_t0.py`. Neither calls this module. T0 uses the frozen working
mask, the confirmed area and confidence rules, and the pinned P5 `encode_mask`.

This optional CPU tool attempts to restore **unmodified original model working
probability fields** from their float16 cache representation. It also requires
the original box and transform. Never use it for T1 fused maps, TCA modified
maps, component masks, or arbitrary probability arrays. Restoration is not
lossless, and its fallback diagnostics must be retained.

From the extracted `package2` directory, after installing the dependencies in
`requirements-restoration.txt` and a compatible PyTorch/Torchvision pair:

```bash
python reader/restoration_v4/read_v4.py --cache cache/dev/RipVIS-097.sqlite --frame 822 --candidate 2 --output v4_example.json
```

The output must be a new file. The adapter uses CPU only, verifies the bundled
source hashes, reads one original candidate, and reports its fallback path.
It does not load a model or GT, compute a new native-mask IoU, or requalify the
restorer. Memory and run time depend on the candidate's original dimensions.

The adapter relocates the immutable restorer's `BASE_SOURCE` reference in
memory to the adjacent, byte-identical `soft_masks_v3.py`. Use `read_v4.py` for
portable access; the unchanged restorer itself retains its original reference
path. This is a relocation adapter, not a new reconstruction algorithm.

`QUALIFICATION.json` records the existing qualification and original evidence
hashes. The original v3 actual-video test remains
FAIL: 1 of 149 candidates fell below the unchanged 0.98 IoU gate (minimum
0.9707612981498331). The later optional v4 qualification passed two disjoint
50-frame sets, 149 candidates each, with minimum IoUs 0.9938299609835768 and
0.9936682725395733. These sample results do not establish universal recovery,
and do not replace the separate native-extraction or common-P5-path checks.
