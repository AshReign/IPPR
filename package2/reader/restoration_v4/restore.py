"""Versioned raw-float16 reader-side inverse, without changing the v3 producer.

Normal v3 results are returned unchanged. Only its saturated-probability fallback
may be replaced by a quantization-interval bilinear-logit fit. Native masks, GT,
checkpoints and images are not inputs. No recovery is promised for arbitrary or
modified T1/TCA fields. Saturated and unobserved logits are not recovered exactly.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import warnings

import numpy as np
import torch
from scipy import optimize, sparse
from scipy.sparse import csgraph, linalg as sparse_linalg

RESTORER_VERSION = "p2-raw-work-restorer-v4-quantization-interval-v1"
BASE_SOFT_MASK_SHA256 = "9021ae722b5e33437e3571c679808b1aae8d2afd90a387f6bfb5ecc9339c101f"
BASE_SOURCE = Path("C:/IPPR/package2/p2/soft_masks.py" if os.name == "nt" else "/mnt/c/IPPR/package2/p2/soft_masks.py")
SPEC = {
    "version": RESTORER_VERSION,
    "input": "unmodified v3 raw full unpadded working probabilities stored as float16; original xyxy box and original transform",
    "trigger": "v3 fallback reason saturated_probability_has_unbounded_logit_interval only",
    "normal_v3_path": "unchanged original function, result values preserved",
    "method": "sparse bilinear-logit interval minimax linear program; scipy HiGHS with one solver thread",
    "quantization": "float16 round-to-nearest-even cells restricted to representable float32 probabilities; P2 sign-preserving >0.5 correction included",
    "stored_half": "upper probability bound exactly 0.5",
    "stored_next_half": "quantization probability cell starts at float32 nextafter(0.5,+inf); underlying positive-logit preimage has open lower 0, relaxed closed 0 for LP with strict positive-logit postcheck",
    "float32_sigmoid_rounding": "expand allowed float32 probability endpoints by their half-ULP round-to-nearest envelopes before logit conversion; not a proof of exact GPU sigmoid implementation preimage",
    "saturated_zero_one": "one-sided logit intervals, no invented finite observation or probability clamp",
    "missing_edge_knots": "same geometric linear extrapolation as v3, explicitly approximate",
    "max_observations": 100000,
    "max_active_knots": 4096,
    "max_constraint_nonzeros": 1000000,
    "rank_rule": "unsaturated observation count >= unknowns; full structural column rank; positive nonsingular Gram LU pivots with relative minimum >=1e-12",
    "gram_pivot_relative_minimum": 1e-12,
    "lp_method": "highs",
    "lp_threads": 1,
    "lp_time_limit_seconds": 20.0,
    "lp_max_iterations": 100000,
    "lp_primal_feasibility_tolerance": 1e-7,
    "lp_dual_feasibility_tolerance": 1e-7,
    "maximum_accepted_logit_interval_violation": 2e-7,
    "maximum_accepted_minimax_slack": 2e-7,
    "working_binary_decisions": "all fitted observation logits >0 must equal stored probabilities >0.5; otherwise retain v3 fallback",
    "output": "float32 sigmoid probabilities, positive-logit >0.5 representation preserved; original box crop",
    "failure_policy": "retain original v3 fallback and explicit failure/limit/rank/feasibility diagnostic; no silent successful reconstruction",
    "losslessness_claimed": False,
    "native_or_gt_used": False,
    "iou_acceptance_threshold": "not decided here; independent original >=0.98 qualification remains required",
}
_BASE = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_MODULE_SHA = _sha(__file__)
_SPEC_SHA = hashlib.sha256(json.dumps(SPEC, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def restorer_manifest():
    if _sha(__file__) != _MODULE_SHA:
        raise RuntimeError("Restorer source changed after import")
    return {"version": RESTORER_VERSION, "sha256": _MODULE_SHA,
            "base_soft_mask_sha256": BASE_SOFT_MASK_SHA256, "spec": dict(SPEC), "spec_sha256": _SPEC_SHA}


def _base_module():
    global _BASE
    if _sha(BASE_SOURCE) != BASE_SOFT_MASK_SHA256:
        raise RuntimeError("Frozen v3 source SHA256 differs")
    if _BASE is None:
        spec = importlib.util.spec_from_file_location("_p2_v4_pinned_v3_source", BASE_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BASE = module
    return _BASE


def probability_intervals(stored):
    """Closed float32 endpoint cells for the sign-preserving fp16 encoder.

Odd fp16 values exclude midpoint ties (ties round to their even neighbours).
Float32 nextafter converts those open endpoints to exact allowed float32 values.
0/1 cells reach 0/1 and therefore remain one-sided after conversion to logits.
"""
    q = np.asarray(stored)
    if q.dtype != np.float16 or not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 1):
        raise ValueError("Quantization intervals require finite float16 probabilities")
    values = q.astype(np.float64)
    previous = np.nextafter(q, np.float16(-np.inf)).astype(np.float64)
    following = np.nextafter(q, np.float16(np.inf)).astype(np.float64)
    lower = np.maximum(0., (previous + values) / 2).astype(np.float32)
    upper = np.minimum(1., (following + values) / 2).astype(np.float32)
    odd = (q.view(np.uint16) & 1) != 0
    lower = np.where(odd, np.nextafter(lower, np.float32(np.inf)), lower)
    upper = np.where(odd, np.nextafter(upper, np.float32(-np.inf)), upper)
    upper = np.where(q == np.float16(.5), np.float32(.5), upper)
    lower = np.where(q == np.nextafter(np.float16(.5), np.float16(np.inf)),
                     np.nextafter(np.float32(.5), np.float32(np.inf)), lower)
    return lower.astype(np.float64), upper.astype(np.float64)


def _logit(probabilities):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(probabilities) - np.log1p(-probabilities)


def logit_intervals(stored):
    """Finite-precision sigmoid envelope, with the prior sign correction.

probability_intervals describes values AFTER float32 sigmoid/sign correction.
The logits may have a smaller positive magnitude than logit(nextafter32(.5)):
sign_preserving_sigmoid first moves a rounded .5 upward for ANY positive logit.
For that cell the open lower bound 0 is relaxed to closed 0 only in the LP; the
postcheck rejects zero/negative fitted logits at every stored positive pixel.
"""
    q = np.asarray(stored)
    low, high = probability_intervals(q)
    low32, high32 = low.astype(np.float32), high.astype(np.float32)
    prior = np.nextafter(low32, np.float32(-np.inf)).astype(np.float64)
    following = np.nextafter(high32, np.float32(np.inf)).astype(np.float64)
    real_low = np.maximum(0., (prior + low) / 2)
    real_high = np.minimum(1., (following + high) / 2)
    lower, upper = _logit(real_low), _logit(real_high)
    lower = np.where(q == np.nextafter(np.float16(.5), np.float16(np.inf)), 0., lower)
    upper = np.where(q == np.float16(.5), 0., upper)
    return lower, upper


def _rank_check(a, unsaturated):
    good = a[unsaturated].tocsc()
    unknowns = a.shape[1]
    detail = {"unsaturated_observations": int(unsaturated.sum()), "unknown_knots": unknowns,
              "method": "structural rank plus numerical sparse Gram LU pivots; not a precise SVD condition number"}
    if good.shape[0] < unknowns:
        return False, {**detail, "reason": "fewer_unsaturated_observations_than_unknown_knots"}
    structural = int(csgraph.structural_rank(good))
    detail["structural_rank"] = structural
    if structural < unknowns:
        return False, {**detail, "reason": "unsaturated_structural_rank_deficient"}
    gram = (good.T @ good).tocsc()
    try:
        lu = sparse_linalg.splu(gram, permc_spec="COLAMD", diag_pivot_thresh=0.)
    except RuntimeError as error:
        return False, {**detail, "reason": "unsaturated_gram_singular", "solver_message": str(error)}
    pivots = lu.U.diagonal()
    relative = float(pivots.min() / pivots.max()) if len(pivots) and pivots.max() > 0 else 0.
    detail.update(gram_pivot_minimum=float(pivots.min()), gram_pivot_maximum=float(pivots.max()),
                  gram_relative_pivot_minimum=relative)
    if not np.isfinite(pivots).all() or np.any(pivots <= 0) or relative < SPEC["gram_pivot_relative_minimum"]:
        return False, {**detail, "reason": "unsaturated_gram_pivots_rank_or_condition_limit"}
    return True, detail


def _try_interval_inverse(work, box, original_hw, crop_shape, base):
    """Return (probabilities_or_None, diagnostic); no reference mask parameter."""
    oh, ow = map(int, original_hw); wh, ww = work.shape; ph, pw = map(int, crop_shape)
    scaled = np.asarray(box, np.float32) * np.array([ww / ow, wh / oh, ww / ow, wh / oh], np.float32)
    xs = np.where((np.arange(ww) >= scaled[0]) & (np.arange(ww) < scaled[2]))[0]
    ys = np.where((np.arange(wh) >= scaled[1]) & (np.arange(wh) < scaled[3]))[0]
    detail = {"status": "retained_v3_fallback", "work_samples_hw": [len(ys), len(xs)]}
    def decline(reason, **extra):
        return None, {**detail, **extra, "reason": reason}
    n = len(xs) * len(ys)
    if not n:
        return decline("no_interior_work_samples")
    if n > SPEC["max_observations"]:
        return decline("observation_resource_limit", observations=n)
    ah = base._sampling_matrix(torch.from_numpy(ys), wh, ph)
    aw = base._sampling_matrix(torch.from_numpy(xs), ww, pw)
    active_h = torch.where(ah.abs().sum(0) > 0)[0]
    active_w = torch.where(aw.abs().sum(0) > 0)[0]
    unknowns = len(active_h) * len(active_w)
    detail.update(observations=n, active_knots_hw=[len(active_h), len(active_w)], unknown_knots=unknowns)
    if unknowns > SPEC["max_active_knots"]:
        return decline("knot_resource_limit")
    a = sparse.kron(sparse.csr_matrix(ah[:, active_h].numpy()), sparse.csr_matrix(aw[:, active_w].numpy()), format="csr")
    observed = work[np.ix_(ys, xs)].ravel()
    unsaturated = (observed > 0) & (observed < 1)
    sufficient, rank = _rank_check(a, unsaturated)
    detail["rank_assessment"] = rank
    if not sufficient:
        return decline("insufficient_unsaturated_rank")
    lower, upper = logit_intervals(observed)
    finite_low, finite_high = np.isfinite(lower), np.isfinite(upper)
    inequalities = sparse.vstack([a[finite_high], -a[finite_low]], format="csr")
    inequalities = sparse.hstack([inequalities, -np.ones((inequalities.shape[0], 1))], format="csr")
    if inequalities.nnz > SPEC["max_constraint_nonzeros"]:
        return decline("constraint_resource_limit", constraint_nonzeros=int(inequalities.nnz))
    rhs = np.concatenate([upper[finite_high], -lower[finite_low]])
    objective = np.zeros(unknowns + 1); objective[-1] = 1.
    options = {"time_limit": SPEC["lp_time_limit_seconds"], "threads": SPEC["lp_threads"],
               "maxiter": SPEC["lp_max_iterations"], "primal_feasibility_tolerance": SPEC["lp_primal_feasibility_tolerance"],
               "dual_feasibility_tolerance": SPEC["lp_dual_feasibility_tolerance"]}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = optimize.linprog(objective, A_ub=inequalities, b_ub=rhs,
                bounds=[(None, None)] * unknowns + [(0., None)], method=SPEC["lp_method"], options=options)
        detail.update(solver_status=int(result.status), solver_message=result.message,
                      solver_warnings=[str(x.message) for x in caught], solver_iterations=int(result.nit))
    except (ValueError, RuntimeError, MemoryError) as error:
        return decline("linear_program_exception", solver_message=repr(error))
    if not result.success or result.x is None or not np.isfinite(result.x).all():
        return decline("linear_program_not_optimal_or_nonfinite")
    logits = np.asarray(a @ result.x[:-1]).ravel()
    violation = float(max(0., np.max(logits[finite_high] - upper[finite_high]), np.max(lower[finite_low] - logits[finite_low])))
    slack = float(result.x[-1])
    decisions_equal = bool(np.array_equal(logits > 0, observed > .5))
    detail.update(minimax_slack=slack, max_logit_interval_violation=violation,
                  working_binary_decisions_equal=decisions_equal, constraint_nonzeros=int(inequalities.nnz))
    if slack > SPEC["maximum_accepted_minimax_slack"] or violation > SPEC["maximum_accepted_logit_interval_violation"]:
        return decline("quantization_intervals_not_numerically_feasible")
    if not decisions_equal:
        return decline("working_binary_decisions_would_change")
    x0, y0 = max(0, int(np.ceil(box[0]))), max(0, int(np.ceil(box[1])))
    x1, y1 = min(ow, int(np.ceil(box[2]))), min(oh, int(np.ceil(box[3])))
    if x1 <= x0 or y1 <= y0:
        return decline("empty_original_box")
    bh = base._sampling_matrix(torch.arange(y0, y1), oh, ph)
    bw = base._sampling_matrix(torch.arange(x0, x1), ow, pw)
    eh = base._edge_extrapolation_matrix(ph, active_h)
    ew = base._edge_extrapolation_matrix(pw, active_w)
    knots = result.x[:-1].reshape(len(active_h), len(active_w))
    original_logits = (bh @ eh).numpy() @ knots @ (bw @ ew).numpy().T
    if not np.isfinite(original_logits).all():
        return decline("nonfinite_original_logits")
    probabilities = torch.zeros((oh, ow), dtype=torch.float32)
    probabilities[y0:y1, x0:x1] = base.sign_preserving_sigmoid(torch.from_numpy(original_logits))
    if not torch.isfinite(probabilities).all():
        return decline("nonfinite_original_probabilities")
    detail.update(status="reconstructed_quantization_interval", reason="saturation_handled_by_quantization_interval_solver",
        missing_target_knots_h=int(((bh.abs().sum(0) > 0) & (ah.abs().sum(0) == 0)).sum()),
        missing_target_knots_w=int(((bw.abs().sum(0) > 0) & (aw.abs().sum(0) == 0)).sum()),
        missing_knots_policy="linear extrapolation, an assumption rather than identifiable recovery",
        knot_recovery_claimed=False, probability_losslessness_claimed=False,
        approximation="one quantization-consistent logit field, with allowed numerical tolerance and v3 missing-edge extrapolation")
    return probabilities, detail


def restore_work_masks(work_masks, original_shape, boxes, *, transform, return_diagnostics=False):
    """Drop-in read-side wrapper; raw float16 only and unchanged v3 producer only."""
    manifest = restorer_manifest(); base = _base_module()
    tensor = torch.as_tensor(work_masks)
    if tensor.dtype != torch.float16 or tensor.ndim != 3:
        raise ValueError("v4 requires N,H,W raw float16 stored probabilities")
    if not isinstance(transform, dict) or transform.get("format_version") != "p2-soft-mask-v3" or transform.get("inverse_applicability") != "unmodified_raw_model_fields_only":
        raise ValueError("v4 only accepts the unchanged v3 raw-model probability format, never T1/TCA modified maps")
    if torch.get_num_threads() > 2:
        raise RuntimeError("Configure at most two torch CPU threads before using this bounded restorer")
    result, original_details = base.restore_work_masks(work_masks, original_shape, boxes, transform=transform, return_diagnostics=True)
    crop = transform["prototype_crop_tlbr"]
    crop_shape = (crop[2] - crop[0], crop[3] - crop[1])
    work = tensor.detach().cpu().numpy()
    box_array = torch.as_tensor(boxes, dtype=torch.float32).detach().cpu().numpy()
    diagnostics = []
    for index, original in enumerate(original_details):
        detail = {**original, "restorer_version": RESTORER_VERSION, "restorer_sha256": manifest["sha256"],
                  "restorer_spec_sha256": manifest["spec_sha256"], "v3_diagnostic": dict(original),
                  "probability_losslessness_claimed": False}
        if original.get("status") == "fallback" and original.get("reason") == "saturated_probability_has_unbounded_logit_interval":
            improved, attempt = _try_interval_inverse(work[index], box_array[index], original_shape, crop_shape, base)
            detail["v4_attempt"] = attempt
            if improved is not None:
                result[index] = improved.to(result.device)
                detail.update(status="reconstructed_quantization_interval", reason=attempt["reason"],
                              knot_recovery_claimed=False, missing_knots_policy=attempt["missing_knots_policy"],
                              missing_target_knots_h=attempt["missing_target_knots_h"], missing_target_knots_w=attempt["missing_target_knots_w"])
        else:
            detail["v4_attempt"] = {"status": "not_attempted", "reason": "original_v3_path_not_saturation_fallback"}
        diagnostics.append(detail)
    if return_diagnostics:
        return result, diagnostics
    if any(x.get("status") == "fallback" for x in diagnostics):
        warnings.warn("Some masks retain the explicitly approximate v3 fallback; inspect diagnostics.", RuntimeWarning)
    return result
