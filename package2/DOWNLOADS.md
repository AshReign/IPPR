# Formal P2 downloads

Release: `IPPR_P2_Formal_EN_20261024_r1`.

[Team Google Drive folder](https://drive.google.com/drive/folders/1Y03emlqYEKkqN_GvMfVhHuhbwLYI8Px3)

This Git-only publication index is outside the 111-file common source payload
included in both cache ZIPs. It does not alter the frozen model, cache contents,
original source snapshot, or original sealed manifests. See [README.md](README.md)
for the full handoff and [docs/PROVENANCE.md](docs/PROVENANCE.md) for version scope.

Download the complete dev package first for P3/P4 work. Download to a local SSD,
then extract the ZIPs into the same parent directory; both use a `package2/` root.
Keep final-val separate from dev and do not use final-val labels for tuning.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `IPPR_P2_Formal_EN_20261024_r1_dev.zip` | 1420036179 | `b1c9709495e01e2f3fda6d33a6c8fdf47715eff9f3b0aa4f36402812cf007ed0` |
| `IPPR_P2_Formal_EN_20261024_r1_final-val.zip` | 1868178975 | `92cd3f2166d5b325bd2e3a2717d56cda94e6d3c6438c665b61ac340f50e39adf` |
| `epoch5.pt` | 17628845 | `278ba795436c1299621b68a6f775a4af8b284afcab9c242384abff877a6906ca` |

The folder also includes `IPPR_P2_Formal_EN_20261024_r1_dev_CHECKSUMS.json`,
`IPPR_P2_Formal_EN_20261024_r1_final-val_CHECKSUMS.json`, and `WEIGHTS_SHA256.json`.
Each ZIP contains a new publication manifest under `package2/manifests/`.
The dev ZIP contains 17 video caches; final-val contains 30, for 47 total.
Place the separate weight at `package2/weights/epoch5.pt` only if a later task
requires the model; reading cached data does not load the weight.

The selected model is completed epoch 6 (`epoch5.pt`), T0 confidence `0.10`.
JPEG dev selection P/R/F2: `0.3106972144387671 / 0.4915254237288136 /
0.4402765450553557`. These are selection results, not scores recomputed from
the MP4 caches.

From the extracted `package2/` directory, after installing
`requirements-reader.txt` in your own environment:

```bash
python reader/read_raw_cache.py --cache cache/dev/RipVIS-031.sqlite --frame-idx 0
python reader/read_final_t0.py --p2-root reader --cache cache/dev/RipVIS-031.sqlite --frame-idx 0 --image-id 2904
```

The raw reader preserves the SQLite v1 candidate pool (`conf >= 0.1`). T0 applies
the confirmed output rules and common P5 `encode_mask`. Neither default reader
calls v4. Optional v4 restoration is only for unmodified original model fields,
not T1/TCA modified maps, and is not lossless. Retain decoded PTS timing and the
documented RipVIS-100 time discrepancy.

Original frozen source snapshot commit:
`c221d204dae9d24188c72dd0fa7bd8e8e7313c94`. The team publication commit is a
separate identifier, visible in this branch's Git history. Source-to-publication
SHA mapping is in `configs/source_release_map.json`.
