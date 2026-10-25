"""Experimental CPU-only inverse of work640's known prototype bilinear sampling.

Uses only stored float16 probabilities, box, shapes and prototype crop metadata.
No GT or native reference enters the reconstruction; native arrays are used solely
afterward to measure IoU. No production codec is changed by this diagnostic.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from p2.cache_io import VideoCacheReader


def sampling_matrix(output_indices, output_size, input_size):
    coordinates = (output_indices.double()+.5)*(input_size/output_size)-.5
    lower = coordinates.floor().long()
    fraction = coordinates-lower
    upper = lower+1
    lower = lower.clamp(0, input_size-1)
    upper = upper.clamp(0, input_size-1)
    matrix = torch.zeros((len(output_indices), input_size), dtype=torch.float64)
    rows = torch.arange(len(output_indices))
    matrix.index_put_((rows, lower), 1-fraction, accumulate=True)
    matrix.index_put_((rows, upper), fraction, accumulate=True)
    return matrix


def extrapolation_matrix(input_size, active_indices):
    """Map unobserved edge knots to a linear extrapolation of observed neighbours."""
    count = len(active_indices)
    matrix = torch.zeros((input_size, count), dtype=torch.float64)
    lo, hi = int(active_indices[0]), int(active_indices[-1])
    for index in range(input_size):
        if lo <= index <= hi:
            matrix[index, index-lo] = 1
        elif count == 1:
            matrix[index, 0] = 1
        elif index < lo:
            delta = lo-index
            matrix[index, 0], matrix[index, 1] = 1+delta, -delta
        else:
            delta = index-hi
            matrix[index, -1], matrix[index, -2] = 1+delta, -delta
    return matrix


def geometry_inverse(work, box, original_shape, crop_shape):
    """Estimate native logit knots via a separable linear least-squares inverse."""
    work = torch.as_tensor(work).double()
    box = torch.as_tensor(box).double()
    oh, ow = map(int, original_shape)
    ph, pw = map(int, crop_shape)
    wh, ww = work.shape
    scaled = box*torch.tensor([ww/ow, wh/oh]*2, dtype=torch.float64)
    # Match production's float32 scaled-box crop exactly, including integer ties.
    scaled32 = torch.as_tensor(box, dtype=torch.float32)*torch.tensor([ww/ow, wh/oh]*2)
    xs = torch.arange(ww)[(torch.arange(ww) >= scaled32[0]) & (torch.arange(ww) < scaled32[2])]
    ys = torch.arange(wh)[(torch.arange(wh) >= scaled32[1]) & (torch.arange(wh) < scaled32[3])]
    result = torch.zeros((oh, ow), dtype=torch.float32)
    if len(xs) == 0 or len(ys) == 0:
        return result, {"reason": "no interior working samples", "work_samples_hw": [len(ys), len(xs)]}
    ah = sampling_matrix(ys, wh, ph)
    aw = sampling_matrix(xs, ww, pw)
    active_h = torch.where(ah.abs().sum(0) > 0)[0]
    active_w = torch.where(aw.abs().sum(0) > 0)[0]
    ah_active, aw_active = ah[:, active_h], aw[:, active_w]
    observed = work[ys[:, None], xs[None, :]]
    # Saturated probabilities carry only sign/magnitude bounds; this diagnostic
    # uses a documented finite midpoint scale, not reference-dependent values.
    logits = torch.logit(observed.clamp(1e-7, 1-1e-7))
    vertical = torch.linalg.lstsq(ah_active, logits, driver="gelsd")
    horizontal = torch.linalg.lstsq(aw_active, vertical.solution.T, driver="gelsd")
    knots = horizontal.solution.T
    x_start, y_start = max(0, int(torch.ceil(box[0]))), max(0, int(torch.ceil(box[1])))
    x_end, y_end = min(ow, int(torch.ceil(box[2]))), min(oh, int(torch.ceil(box[3])))
    if x_end <= x_start or y_end <= y_start:
        return result, {"reason": "empty original box"}
    target_x, target_y = torch.arange(x_start, x_end), torch.arange(y_start, y_end)
    bh = sampling_matrix(target_y, oh, ph)
    bw = sampling_matrix(target_x, ow, pw)
    eh, ew = extrapolation_matrix(ph, active_h), extrapolation_matrix(pw, active_w)
    restored_logits = (bh@eh)@knots@(bw@ew).T
    result[y_start:y_end, x_start:x_end] = restored_logits.sigmoid().float()
    metadata = {"work_samples_hw": [len(ys), len(xs)],
        "estimated_knots_hw": [len(active_h), len(active_w)],
        "matrix_ranks_hw": [int(vertical.rank), int(horizontal.rank)],
        "saturated_observations": int(((observed<=0)|(observed>=1)).sum()),
        "work_probability_min": float(observed.min()), "work_probability_max": float(observed.max()),
        "fit_max_logit_residual": float((ah_active@knots@aw_active.T-logits).abs().max()),
        "missing_target_knots_h": int(((bh.abs().sum(0)>0)&(ah.abs().sum(0)==0)).sum()),
        "missing_target_knots_w": int(((bw.abs().sum(0)>0)&(aw.abs().sum(0)==0)).sum())}
    return result, metadata


def binary_iou(lhs, rhs):
    union = (lhs | rhs).sum().item()
    return (lhs & rhs).sum().item()/union if union else 1.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worst-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summary = json.loads((args.validation/"summary.json").read_text())
    failures = [row for row in summary["failures"] if isinstance(row["cache_iou"], (float, int)) and row["cache_iou"] < .98]
    failures.sort(key=lambda row: row["cache_iou"])
    if args.worst_only:
        failures = failures[:1]
    records = []
    for failed in failures:
        stem = f"{failed['video_id']}_{int(failed['frame_idx']):06d}"
        reader = VideoCacheReader(args.validation/"roundtrip_samples"/(stem+".sqlite"))
        try:
            cached = reader.read(0)
            meta = reader.metadata
        finally:
            reader.close()
        ci = failed["candidate_index"]
        crop = meta["prototype_crop_tlbr"]
        restored, details = geometry_inverse(cached["soft_mask"][ci], cached["boxes"][ci],
            meta["original_shape_hw"], (crop[2]-crop[0], crop[3]-crop[1]))
        # Native reference enters only here, after reconstruction is complete.
        with np.load(args.validation/"failures"/f"{stem}_{failed['instance_idx']}.npz", allow_pickle=False) as data:
            native = torch.from_numpy(data["soft_probabilities"]) > .5
        record = {**failed, "geometry_inverse_iou": binary_iou(restored>.5, native),
                  "native_area_pixels": int(native.sum()), "geometry": details}
        records.append(record)
        print(json.dumps({"case": len(records), "total": len(failures), "video": failed["video_id"],
            "frame": failed["frame_idx"], "candidate": ci, "iou": record["geometry_inverse_iou"]}), flush=True)
    report = {"diagnostic_only": True, "production_unchanged": True,
        "checkpoint_sha256": summary["checkpoint_sha256"], "cases": len(records),
        "failures_below_0_98": sum(row["geometry_inverse_iou"]<.98 for row in records),
        "min_iou": min((row["geometry_inverse_iou"] for row in records), default=None),
        "method": "separable least-squares recovery of native bilinear logit knots from known valid cropped work640 samples; missing edge knots linearly extrapolated; exact original box crop",
        "input_information": "float16 working probabilities, original box, original/work/prototype crop dimensions only; no auxiliary data or reference-dependent corrections",
        "rows": records}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key:value for key,value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
