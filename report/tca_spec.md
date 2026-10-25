# TCA Implementation Specification

## Overview
Temporal Confidence Aggregation (TCA) is a causal post-processing
method for improving the temporal consistency of rip current
segmentation predictions.

Implementation: `temporal/tca.py`

## Core Implementation Rules
- Input: Per-frame segmentation predictions from the shared cache.
- Association: Match detections to existing tracks using mask IoU
  and Hungarian assignment.
- Tracking: Maintain independent tracks for detected regions.
- Aggregation: Update track confidence using temporal evidence
  from consecutive frames.
- Smoothing: Apply Gaussian filtering to spatial confidence maps.
- Hysteresis: Use high and low thresholds to stabilise output masks.
- Persistence: Control activation and disappearance using presence
  and absence durations.
- Timeout: Remove inactive tracks after the configured timeout.
- Output: Temporally processed segmentation predictions.

## Frozen T2 Settings

| Parameter | Value |
|---|---:|
| Tau | 0.1221 s |
| Presence duration | 0.3142 s |
| Absence duration | 1.7877 s |
| Track timeout | 2.3746 s |
| High threshold | 0.5075 |
| Low threshold | 0.3460 |
| IoU threshold | 0.3000 |
| Gaussian sigma | 2 px |
| Minimum area fraction (evaluation) | 0.001 |

These are the frozen T2 settings used for the final validation
evaluation. They should not be changed after model selection.

## Implementation and Testing
- Main implementation: `temporal/tca.py`
- Previous implementation: `temporal/tca_before.py`
- Comparison script: `scripts/compare_tca.py`
- Unit tests: `tests/test_tca.py`

The comparison script was tested on 1,800 frames at 640x360
resolution and 30 FPS.

Previous implementation: 0.891 ms/frame.
Current implementation: 0.480 ms/frame.
Measured speedup: 1.85x.
Outputs identical in this comparison: True.

## Limitations
Temporal aggregation can improve prediction stability but may
preserve false positives and introduce additional processing delay.
Segmentation accuracy, temporal stability and runtime should
therefore be evaluated together.


## Frozen Configuration Reference
- Configuration file: `configs/frozen/t2.yaml`
- Condition: T2
- Configuration ID: T2_11
- Frozen date: 2026-10-24
- Selection source: `report/logs/dev_search/dev_search.csv`
- Development F2-score: 0.4018

The frozen values are passed explicitly to the evaluation pipeline
and differ from the default values in the TCA class.
