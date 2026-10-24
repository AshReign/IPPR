"""CPU-only experimental fit excluding saturated observations; production untouched.

Stored fp16 probabilities, boxes and geometry are the only reconstruction inputs.
Saved uncompressed reference enters only after the candidate reconstruction.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch
import torch.nn.functional as F
from ultralytics.utils import ops

from p2.cache_io import VideoCacheReader
from p2.soft_masks import _sampling_matrix, _edge_extrapolation_matrix, restore_work_masks


def unsaturated_inverse(work, box, original_shape, crop_shape):
    started = time.perf_counter()
    work = torch.as_tensor(work).double()
    box = torch.as_tensor(box).double()
    oh, ow = map(int, original_shape)
    ph, pw = map(int, crop_shape)
    wh, ww = work.shape
    scaled = box.float()*torch.tensor([ww/ow, wh/oh]*2)
    xs = torch.arange(ww)[(torch.arange(ww)>=scaled[0]) & (torch.arange(ww)<scaled[2])]
    ys = torch.arange(wh)[(torch.arange(wh)>=scaled[1]) & (torch.arange(wh)<scaled[3])]
    observed = work[ys[:, None], xs[None, :]]
    finite = ((observed>0)&(observed<1)).numpy().reshape(-1)
    ah, aw = _sampling_matrix(ys, wh, ph), _sampling_matrix(xs, ww, pw)
    active_h, active_w = torch.where(ah.abs().sum(0)>0)[0], torch.where(aw.abs().sum(0)>0)[0]
    ah_active, aw_active = ah[:, active_h], aw[:, active_w]
    design = sparse.kron(sparse.csr_matrix(ah_active.numpy()), sparse.csr_matrix(aw_active.numpy()), format="csr")
    finite_design = design[finite]
    target = torch.logit(observed.reshape(-1)[torch.from_numpy(finite)]).numpy()
    # Sparse construction preserves four bilinear weights per observation. The
    # finite-observation rank/conditioning check occurs before any native scoring.
    gram = torch.from_numpy((finite_design.T@finite_design).toarray())
    eigenvalues = torch.linalg.eigvalsh(gram)
    detail = {"observations": int(observed.numel()), "finite_observations": int(finite.sum()),
              "saturated_zero": int((observed==0).sum()), "saturated_one": int((observed==1).sum()),
              "knots_hw": [len(active_h), len(active_w)], "gram_eigenvalue_min": float(eigenvalues[0]),
              "gram_eigenvalue_max": float(eigenvalues[-1]),
              "input_information": "stored float16 probabilities, original box and geometry only"}
    if eigenvalues[0] <= eigenvalues[-1]*1e-16:
        detail.update(status="insufficient_rank_or_condition", elapsed_seconds=time.perf_counter()-started)
        return None, detail
    detail["estimated_design_condition"] = float((eigenvalues[-1]/eigenvalues[0]).sqrt())
    rhs = torch.from_numpy(np.asarray(finite_design.T@target))
    solution = torch.cholesky_solve(rhs[:, None], torch.linalg.cholesky(gram))[:, 0]
    knots = solution.reshape(len(active_h), len(active_w))
    predicted_work_logits = ah_active@knots@aw_active.T
    lower_one = np.log((1.-2.**-12)/(2.**-12))
    upper_zero = np.log((2.**-25)/(1.-2.**-25))
    positive = predicted_work_logits[observed==1]
    negative = predicted_work_logits[observed==0]
    detail.update(saturated_one_logit_lower_bound=float(lower_one), saturated_zero_logit_upper_bound=float(upper_zero),
        positive_bound_violations=int((positive<lower_one).sum()),
        negative_bound_violations=int((negative>upper_zero).sum()),
        positive_bound_max_violation=float((lower_one-positive).clamp_min(0).max()) if len(positive) else 0.,
        negative_bound_max_violation=float((negative-upper_zero).clamp_min(0).max()) if len(negative) else 0.,
        saturated_fitted_logit_min=float(positive.min()) if len(positive) else None,
        finite_fit_max_logit_residual=float((predicted_work_logits.reshape(-1)[torch.from_numpy(finite)]-torch.from_numpy(target)).abs().max()))
    x0, y0 = max(0, int(torch.ceil(box[0]))), max(0, int(torch.ceil(box[1])))
    x1, y1 = min(ow, int(torch.ceil(box[2]))), min(oh, int(torch.ceil(box[3])))
    bh, bw = _sampling_matrix(torch.arange(y0,y1), oh, ph), _sampling_matrix(torch.arange(x0,x1), ow, pw)
    eh, ew = _edge_extrapolation_matrix(ph, active_h), _edge_extrapolation_matrix(pw, active_w)
    result = torch.zeros((oh,ow), dtype=torch.float32)
    result[y0:y1,x0:x1] = ((bh@eh)@knots@(bw@ew).T).sigmoid().float()
    detail.update(status="reconstructed_from_unsaturated_observations",
        missing_target_knots_h=int(((bh.abs().sum(0)>0)&(ah.abs().sum(0)==0)).sum()),
        missing_target_knots_w=int(((bw.abs().sum(0)>0)&(aw.abs().sum(0)==0)).sum()),
        elapsed_seconds=time.perf_counter()-started)
    return result, detail


def iou(a,b):
    union = int((a|b).sum())
    return int((a&b).sum())/union if union else 1.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summary = json.loads((args.validation/"summary.json").read_text())
    rows = []
    for failed in summary["failures"]:
        stem = f"{failed['video_id']}_{int(failed['frame_idx']):06d}"
        reader = VideoCacheReader(args.validation/"roundtrip_samples"/(stem+".sqlite"))
        try:
            cached, meta = reader.read(0), reader.metadata
        finally:
            reader.close()
        ci = failed["candidate_index"]
        work, box = cached["soft_mask"][ci], cached["boxes"][ci]
        crop = meta["prototype_crop_tlbr"]
        shape = tuple(meta["original_shape_hw"])
        proposed, detail = unsaturated_inverse(work, box, shape, (crop[2]-crop[0],crop[3]-crop[1]))
        legacy = restore_work_masks(work[None], shape, box[None])[0]
        wh, ww = work.shape
        oh, ow = shape
        scaled = torch.from_numpy(box)*torch.tensor([ww/ow,wh/oh]*2)
        support = ops.crop_mask(torch.ones((1,wh,ww)),scaled[None])
        support_up = F.interpolate(support[None], shape, mode="bilinear", align_corners=False)[0]
        work_up = F.interpolate(torch.from_numpy(work).float()[None,None], shape, mode="bilinear", align_corners=False)[0]
        normalized = ops.crop_mask(torch.where(support_up>0,work_up/support_up.clamp_min(1e-12),0),torch.from_numpy(box)[None])[0]
        # Reference is loaded only after every reconstruction variant is finished.
        with np.load(args.validation/"failures"/f"{stem}_{failed['instance_idx']}.npz",allow_pickle=False) as saved:
            native = torch.from_numpy(saved["soft_probabilities"])>.5
            preserved = torch.from_numpy(saved["restored_probabilities"])>.5
        errors = (legacy>.5)^native
        yy, xx = torch.meshgrid(torch.arange(oh),torch.arange(ow), indexing="ij")
        boundary = (xx<box[0]+ow/ww)|(xx>box[2]-ow/ww)|(yy<box[1]+oh/wh)|(yy>box[3]-oh/wh)
        saturated_up = F.interpolate(((torch.from_numpy(work)>=1)&support[0].bool()).float()[None,None],shape,mode="nearest")[0,0].bool()
        rows.append({**failed,"original_shape":shape,"work_shape":list(work.shape),"box":box.tolist(),
            "legacy_matches_saved":bool(torch.equal(legacy>.5,preserved)),"native_area_pixels":int(native.sum()),
            "legacy_iou":iou(legacy>.5,native),"support_normalized_iou":iou(normalized>.5,native),
            "unsaturated_inverse_iou":None if proposed is None else iou(proposed>.5,native),
            "legacy_binary_errors":int(errors.sum()),"legacy_errors_in_one_work_pixel_box_band":int((errors&boundary).sum()),
            "legacy_errors_in_saturated_region":int((errors&saturated_up).sum()),"inverse":detail})
    report={"diagnostic_only":True,"production_unchanged":True,"reference_used_only_for_scoring":True,
        "checkpoint_sha256":summary["checkpoint_sha256"],"rows":rows}
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
