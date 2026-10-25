"""Audited soft masks for Ultralytics 8.3.197, with inclusive confidence filtering.

Keep ``Results`` binary masks as an independent standard-output reference. Attached
``soft_masks`` are float32, original-resolution probabilities; ``work_soft_masks``
are float16, no-padding, aspect-preserving long-side-640 probabilities. Neither is
made by casting a binary mask. See reports/softmask_source_audit.md.
"""
from __future__ import annotations

import math
from typing import Sequence
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision  # Fix the locked library's conditional NMS backend explicitly.
import ultralytics
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, nms, ops

LOCKED_ULTRALYTICS = "8.3.197"
LEGACY_FORMAT_VERSION = "p2-soft-mask-v1"
GEOMETRY_V2_FORMAT_VERSION = "p2-soft-mask-v2"
FORMAT_VERSION = "p2-soft-mask-v3"
WORK_ENCODER = "native-aspect-prototype-logits-to-work-v1"
INVERSE_RCOND = 1e-12
INVERSE_MAX_CONDITION = 1e8
PREDICT_DEFAULTS = {
    "imgsz": 640,
    "retina_masks": True,
    "rect": False,
    "half": False,
    "conf": 0.1,
    "iou": 0.7,
    "max_det": 300,
    "agnostic_nms": False,
    "augment": False,
    "verbose": False,
    "save": False,
}
NMS_FIXED = {
    "backend": "torchvision.ops.nms",
    "multi_label": False,
    "max_nms": 30000,
    "max_wh": 7680,
    "max_time_img": 30.0,
    "rotated": False,
    "end2end": False,
    "confidence_dtype": "float32",
    "confidence_comparison": ">= float32(threshold)",
    "instance_order": "unchanged torchvision NMS output order, then stable boolean filtering",
}


def float32_threshold(threshold: float) -> np.float32:
    """Canonical representation for stored confidences and all protocol thresholds."""
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Confidence threshold must be finite and in (0, 1].")
    return np.float32(threshold)


def strict_nms_prefilter(threshold: float) -> float:
    """One float32 ULP below t makes the library's strict > implement float32 >= t."""
    return float(np.nextafter(float32_threshold(threshold), np.float32(-np.inf)))


def confidence_keep(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Use this same inclusive comparison for candidate and later T0 filtering."""
    return scores.float() >= float(float32_threshold(threshold))


def inclusive_nms(predictions: torch.Tensor, *, nc: int, threshold: float = 0.1,
                  iou: float = 0.7, classes=None, agnostic: bool = False,
                  max_det: int = 300) -> list[torch.Tensor]:
    """Locked standard NMS with explicit float32 inclusive confidence semantics.

    The input is cloned because library NMS modifies xywh coordinates in-place.
    Returning a fresh list makes repeated threshold-equivalence checks safe.
    """
    if not isinstance(predictions, torch.Tensor) or predictions.ndim != 3:
        raise ValueError("Expected YOLO raw prediction tensor with shape B,C,N.")
    # The locked NMS heuristically treats final dimension 6 as end-to-end. Real
    # YOLO11 head output has far more candidates; fail rather than misinterpret.
    if predictions.shape[-1] == 6:
        raise ValueError("Six-candidate raw input is ambiguous to locked Ultralytics NMS.")
    filtered = nms.non_max_suppression(
        predictions.float().clone(), conf_thres=strict_nms_prefilter(threshold),
        iou_thres=iou, classes=classes, agnostic=agnostic, multi_label=False,
        max_det=max_det, nc=nc, max_time_img=NMS_FIXED["max_time_img"],
        max_nms=NMS_FIXED["max_nms"], max_wh=NMS_FIXED["max_wh"],
        rotated=False, end2end=False, return_idxs=False,
    )
    return [pred[confidence_keep(pred[:, 4], threshold)] for pred in filtered]


def sign_preserving_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    """Compute true sigmoid, correcting only positive values rounded exactly to .5.

    Official binary output uses logit > 0, whereas float32 sigmoid can round a
    tiny positive logit to exactly .5. One ULP resolves that representational
    boundary without changing any value with an already-correct >.5 comparison.
    """
    probabilities = logits.float().sigmoid()
    half = probabilities.new_tensor(0.5)
    above = torch.nextafter(half, probabilities.new_tensor(float("inf")))
    return torch.where((logits > 0) & (probabilities <= half), above, probabilities)


def native_soft_masks(protos: torch.Tensor, coefficients: torch.Tensor,
                      boxes: torch.Tensor, original_shape: Sequence[int]) -> torch.Tensor:
    """Match process_mask_native geometry, preserving probability before binarization.

    Order is coefficients @ float32 protos -> remove prototype-space padding ->
    bilinear logit interpolation -> sigmoid -> original-coordinate box crop.
    Cropping after sigmoid is essential: sigmoid(cropped zero logit) would be .5
    outside every box instead of the required zero probability.
    """
    height, width = map(int, original_shape[:2])
    if not len(coefficients):
        return protos.new_empty((0, height, width), dtype=torch.float32)
    channels, mh, mw = protos.shape
    logits = (coefficients.float() @ protos.float().view(channels, -1)).view(-1, mh, mw)
    logits = ops.scale_masks(logits[None], (height, width))[0]
    return ops.crop_mask(sign_preserving_sigmoid(logits), boxes)


def work_shape(original_shape: Sequence[int], long_side: int = 640) -> tuple[int, int]:
    height, width = map(int, original_shape[:2])
    if min(height, width, long_side) <= 0:
        raise ValueError("Image dimensions and long_side must be positive.")
    scale = long_side / max(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def quantize_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """float16 storage with the >.5 bit preserved at each stored working pixel."""
    quantized = probabilities.to(torch.float16)
    half = quantized.new_tensor(0.5)
    above = torch.nextafter(half, quantized.new_tensor(float("inf")))
    return torch.where((probabilities > 0.5) & (quantized <= half), above, quantized)


def encode_work_masks(probabilities: torch.Tensor, long_side: int = 640) -> torch.Tensor:
    """Downsample true original-pixel probabilities, never binary masks; no padding."""
    if probabilities.ndim != 3:
        raise ValueError("Expected N,H,W probability tensor.")
    target = work_shape(probabilities.shape[-2:], long_side)
    if not len(probabilities):
        return probabilities.new_empty((0, *target), dtype=torch.float16)
    resized = F.interpolate(probabilities.float()[None], target, mode="bilinear", align_corners=False)[0]
    return quantize_probabilities(resized)


def direct_work_masks(protos: torch.Tensor, coefficients: torch.Tensor,
                      boxes: torch.Tensor, original_shape: Sequence[int]) -> torch.Tensor:
    """Production encoder: crop padding and interpolate logits directly to work640.

    Padding is computed from ORIGINAL aspect, exactly as native scale_masks does,
    not from rounded work dimensions. No original-resolution tensor is allocated.
    All post-NMS candidates are retained, including empty working masks.
    """
    oh, ow = map(int, original_shape[:2])
    wh, ww = work_shape((oh, ow))
    if not len(coefficients):
        return protos.new_empty((0, wh, ww), dtype=torch.float16)
    channels, ph, pw = protos.shape
    logits = (coefficients.float() @ protos.float().view(channels, -1)).view(-1, ph, pw)
    gain = min(ph / oh, pw / ow)
    pdw, pdh = (pw - ow * gain) / 2, (ph - oh * gain) / 2
    top, left = round(pdh - .1), round(pdw - .1)
    bottom, right = ph-round(pdh + .1), pw-round(pdw + .1)
    logits = F.interpolate(logits[None, :, top:bottom, left:right], (wh, ww),
                           mode="bilinear", align_corners=False)[0]
    working_boxes = boxes.float().clone()
    working_boxes[:, [0, 2]] *= ww / ow
    working_boxes[:, [1, 3]] *= wh / oh
    probabilities = ops.crop_mask(sign_preserving_sigmoid(logits), working_boxes)
    return quantize_probabilities(probabilities)


def _sampling_matrix(output_indices: torch.Tensor, output_size: int, input_size: int) -> torch.Tensor:
    coordinates = (output_indices.double()+.5)*(input_size/output_size)-.5
    lower = coordinates.floor().long()
    fraction = coordinates-lower
    upper = lower+1
    matrix = torch.zeros((len(output_indices), input_size), dtype=torch.float64)
    rows = torch.arange(len(output_indices))
    matrix.index_put_((rows, lower.clamp(0, input_size-1)), 1-fraction, accumulate=True)
    matrix.index_put_((rows, upper.clamp(0, input_size-1)), fraction, accumulate=True)
    return matrix


def _edge_extrapolation_matrix(input_size: int, active_indices: torch.Tensor) -> torch.Tensor:
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


def _legacy_restore_one(work: torch.Tensor, box: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    resized = F.interpolate(work.float()[None, None], shape, mode="bilinear", align_corners=False)[0]
    return ops.crop_mask(resized, box.float()[None])[0]


def _support_normalized_restore_one(work: torch.Tensor, box: torch.Tensor,
                                    shape: tuple[int, int]) -> tuple[torch.Tensor, dict]:
    """Probability-only fallback; box support includes valid zero probabilities."""
    work, box = work.float(), box.float()
    oh, ow = shape
    wh, ww = work.shape
    scaled_box = box*torch.tensor([ww/ow, wh/oh]*2, dtype=torch.float32)
    support = ops.crop_mask(torch.ones((1, wh, ww), dtype=torch.float32), scaled_box[None])
    support_up = F.interpolate(support[None], shape, mode="bilinear", align_corners=False)[0]
    numerator = F.interpolate(work[None, None], shape, mode="bilinear", align_corners=False)[0]
    # Never infer support from work>0: an interior p=0 is a valid observation.
    denominator = torch.where(support_up > 0, support_up, 1)
    normalized = torch.where(support_up > 0, numerator/denominator, 0)
    output = ops.crop_mask(normalized, box[None])[0]
    original_support = ops.crop_mask(torch.ones((1, oh, ow)), box[None]).bool()
    zero_count = int(((support_up <= 0)&original_support).sum())
    return output, {"fallback_probability_method": "bilinear_probability_divided_by_bilinear_known_box_support",
        "support_source": "scaled_original_xyxy_box_pixel_support_including_probability_zero",
        "work_support_pixels": int(support.sum()), "original_box_zero_support_pixels": zero_count,
        "has_original_box_zero_support": zero_count > 0,
        "zero_support_policy": "leave probability zero; no fabricated observations",
        "fallback_approximation": "probability interpolation; saturated logit knots are not recovered"}


def geometry_inverse_single(work: torch.Tensor | np.ndarray, box: torch.Tensor | np.ndarray,
                            original_shape: Sequence[int], crop_shape: Sequence[int], *,
                            saturation_fallback: str = "legacy_v1") -> tuple[torch.Tensor, dict]:
    """CPU float64 fit of raw-model bilinear logit knots; returns fp32 probabilities.

    Inputs are exclusively the stored working probabilities and geometric
    metadata. No native logit/prototype values, binary reference, or GT are used.
    Missing edge knots are linearly extrapolated, an explicit approximation.
    Missing observations, deficient rank and poor conditioning return a legacy
    probability fallback. Saturation may use the versioned known-support fallback;
    neither fallback claims recovery of saturated or unobserved logit knots.
    """
    work = torch.as_tensor(work).detach().cpu().double()
    box = torch.as_tensor(box).detach().cpu().double()
    oh, ow = map(int, original_shape[:2])
    ph, pw = map(int, crop_shape[:2])
    if saturation_fallback not in ("legacy_v1", "known_box_support"):
        raise ValueError("Unknown saturation fallback.")
    if work.ndim != 2 or box.shape != (4,) or min(oh, ow, ph, pw) <= 0:
        raise ValueError("Invalid working mask, box or prototype/original shape.")
    if not bool(torch.isfinite(work).all() and torch.isfinite(box).all()) or bool(((work < 0)|(work > 1)).any()):
        raise ValueError("Inverse inputs must be finite probabilities and box coordinates.")
    if not (0 <= box[0] <= box[2] <= ow and 0 <= box[1] <= box[3] <= oh):
        raise ValueError("Inverse requires clipped original-coordinate xyxy boxes.")
    wh, ww = work.shape
    scaled = box.float()*torch.tensor([ww/ow, wh/oh]*2, dtype=torch.float32)
    x_grid, y_grid = torch.arange(ww), torch.arange(wh)
    xs = x_grid[(x_grid >= scaled[0]) & (x_grid < scaled[2])]
    ys = y_grid[(y_grid >= scaled[1]) & (y_grid < scaled[3])]
    detail = {"method": "separable_bilinear_logit_inverse", "rcond": INVERSE_RCOND,
              "max_allowed_condition": INVERSE_MAX_CONDITION, "compute_device": "cpu",
              "compute_dtype": "float64", "work_samples_hw": [len(ys), len(xs)]}

    def fallback(reason):
        detail.update(status="fallback", reason=reason, knot_recovery_claimed=False)
        return _legacy_restore_one(work, box, (oh, ow)), detail

    if not len(xs) or not len(ys):
        return fallback("no_interior_work_samples")
    observed = work[ys[:, None], xs[None, :]]
    saturated = int(((observed <= 0)|(observed >= 1)).sum())
    detail["saturated_observations"] = saturated
    if saturated:
        # fp16 0/1 encode unbounded logit intervals. Do not invent finite logits.
        if saturation_fallback == "known_box_support":
            restored, fallback_detail = _support_normalized_restore_one(work, box, (oh, ow))
            detail.update(status="fallback", reason="saturated_probability_has_unbounded_logit_interval",
                          knot_recovery_claimed=False, **fallback_detail)
            return restored, detail
        return fallback("saturated_probability_has_unbounded_logit_interval")
    ah, aw = _sampling_matrix(ys, wh, ph), _sampling_matrix(xs, ww, pw)
    active_h, active_w = torch.where(ah.abs().sum(0) > 0)[0], torch.where(aw.abs().sum(0) > 0)[0]
    ah_active, aw_active = ah[:, active_h], aw[:, active_w]
    logits = torch.logit(observed)
    vertical = torch.linalg.lstsq(ah_active, logits, rcond=INVERSE_RCOND, driver="gelsd")
    horizontal = torch.linalg.lstsq(aw_active, vertical.solution.T, rcond=INVERSE_RCOND, driver="gelsd")
    singulars = [vertical.singular_values, horizontal.singular_values]
    conditions = [float(values[0]/values[-1]) if len(values) and values[-1] > 0 else math.inf for values in singulars]
    detail.update(estimated_knots_hw=[len(active_h), len(active_w)],
        matrix_ranks_hw=[int(vertical.rank), int(horizontal.rank)],
        singular_min_hw=[float(values[-1]) if len(values) else 0. for values in singulars],
        singular_max_hw=[float(values[0]) if len(values) else 0. for values in singulars],
        condition_estimates_hw=[value if math.isfinite(value) else None for value in conditions])
    if int(vertical.rank) != len(active_h) or int(horizontal.rank) != len(active_w):
        return fallback("insufficient_rank")
    if any(value > INVERSE_MAX_CONDITION for value in conditions):
        return fallback("ill_conditioned_sampling_matrix")
    knots = horizontal.solution.T
    x_start, y_start = max(0, int(torch.ceil(box[0]))), max(0, int(torch.ceil(box[1])))
    x_end, y_end = min(ow, int(torch.ceil(box[2]))), min(oh, int(torch.ceil(box[3])))
    result = torch.zeros((oh, ow), dtype=torch.float32)
    if x_end <= x_start or y_end <= y_start:
        detail.update(status="empty_original_box", knot_recovery_claimed=False)
        return result, detail
    bh = _sampling_matrix(torch.arange(y_start, y_end), oh, ph)
    bw = _sampling_matrix(torch.arange(x_start, x_end), ow, pw)
    eh, ew = _edge_extrapolation_matrix(ph, active_h), _edge_extrapolation_matrix(pw, active_w)
    restored_logits = (bh@eh)@knots@(bw@ew).T
    result[y_start:y_end, x_start:x_end] = restored_logits.sigmoid().float()
    detail.update(status="reconstructed", knot_recovery_claimed=True,
        fit_max_logit_residual=float((ah_active@knots@aw_active.T-logits).abs().max()),
        missing_target_knots_h=int(((bh.abs().sum(0)>0)&(ah.abs().sum(0)==0)).sum()),
        missing_target_knots_w=int(((bw.abs().sum(0)>0)&(aw.abs().sum(0)==0)).sum()),
        missing_knots_policy="linear extrapolation, an assumption rather than identifiable recovery")
    return result, detail


def restore_work_masks(work_masks: torch.Tensor | np.ndarray,
                       original_shape: Sequence[int], boxes: torch.Tensor | np.ndarray,
                       *, transform: dict | None = None, return_diagnostics: bool = False):
    """Versioned reconstruction of one frame's working masks to original pixels.

    v2/v3 require explicit raw encoder metadata and use a bounded per-instance CPU
    solve. It is NOT valid for TCA/T1/other modified probability fields. None or
    explicit v1 keeps legacy bilinear behavior for old artifacts. The >=.98 gate
    remains mandatory; the inverse is not a universal losslessness guarantee.
    v3 changes only saturation fallback to known-box-support normalization; v2
    retains the original legacy fallback exactly.
    """
    masks = torch.as_tensor(work_masks).float()
    height, width = map(int, original_shape[:2])
    if masks.ndim != 3:
        raise ValueError("Expected N,H,W working masks.")
    box_tensor = torch.as_tensor(boxes, dtype=torch.float32, device=masks.device)
    if tuple(box_tensor.shape) != (len(masks), 4):
        raise ValueError("boxes must have shape (N,4) and match instance order.")
    version = LEGACY_FORMAT_VERSION if transform is None else transform.get("format_version")
    if version not in (LEGACY_FORMAT_VERSION, GEOMETRY_V2_FORMAT_VERSION, FORMAT_VERSION):
        raise ValueError("Unsupported or missing soft-mask format version.")
    if version in (GEOMETRY_V2_FORMAT_VERSION, FORMAT_VERSION):
        if transform.get("work_encoder") != WORK_ENCODER or transform.get("inverse_applicability") != "unmodified_raw_model_fields_only":
            raise ValueError("Geometry inverse requires the versioned direct raw-model encoder; modified maps are unsupported.")
        if tuple(transform["original_shape_hw"]) != (height, width) or tuple(transform["work_shape_hw"]) != tuple(masks.shape[-2:]):
            raise ValueError("Inverse geometry metadata does not match the stored mask.")
        crop = transform["prototype_crop_tlbr"]
        crop_shape = crop[2]-crop[0], crop[3]-crop[1]
        output = torch.empty((len(masks), height, width), dtype=torch.float32)
        details = []
        cpu_masks, cpu_boxes = masks.detach().cpu(), box_tensor.detach().cpu()
        for index in range(len(masks)):
            output[index], detail = geometry_inverse_single(cpu_masks[index], cpu_boxes[index], (height, width), crop_shape,
                saturation_fallback="known_box_support" if version == FORMAT_VERSION else "legacy_v1")
            details.append(detail)
        if not return_diagnostics and any(item["status"] == "fallback" for item in details):
            warnings.warn("Some masks require an approximate probability fallback; request return_diagnostics=True to inspect reasons.", RuntimeWarning)
        output = output.to(masks.device)
        return (output, details) if return_diagnostics else output
    if not len(masks):
        empty = masks.new_empty((0, height, width))
        return (empty, []) if return_diagnostics else empty
    restored = F.interpolate(masks[None], (height, width), mode="bilinear", align_corners=False)[0]
    restored = ops.crop_mask(restored, box_tensor)
    return (restored, [{"status": "legacy_v1"} for _ in range(len(masks))]) if return_diagnostics else restored


def transform_metadata(original_shape, input_shape, proto_shape) -> dict:
    """Record exact transforms used by locked Ultralytics, including rounding."""
    oh, ow = map(int, original_shape[:2])
    ih, iw = map(int, input_shape[:2])
    ph, pw = map(int, proto_shape[:2])
    gain = min(ih / oh, iw / ow)
    resized_w, resized_h = round(ow * gain), round(oh * gain)
    dw, dh = (iw - resized_w) / 2, (ih - resized_h) / 2
    proto_gain = min(ph / oh, pw / ow)
    pdw, pdh = (pw - ow * proto_gain) / 2, (ph - oh * proto_gain) / 2
    return {
        "format_version": FORMAT_VERSION,
        "work_encoder": WORK_ENCODER,
        "inverse_applicability": "unmodified_raw_model_fields_only",
        "original_shape_hw": [oh, ow], "model_input_shape_hw": [ih, iw],
        "prototype_shape_hw": [ph, pw], "work_shape_hw": list(work_shape((oh, ow))),
        "work_long_side": 640, "work_has_padding": False,
        "letterbox_gain": gain, "letterbox_resized_shape_hw": [resized_h, resized_w],
        "letterbox_padding_ltrb": [round(dw - .1), round(dh - .1), round(dw + .1), round(dh + .1)],
        "scale_boxes_unpad_xy": [round((iw - ow * gain) / 2 - .1), round((ih - oh * gain) / 2 - .1)],
        "prototype_crop_tlbr": [round(pdh - .1), round(pdw - .1), ph-round(pdh + .1), pw-round(pdw + .1)],
        "interpolation": "torch bilinear, align_corners=False; logits before sigmoid",
        "working_encode": "prototype logits -> original-aspect padding crop -> bilinear to work_shape -> sigmoid -> scaled xyxy crop -> sign-preserving float16",
        "working_restore": "fp16 probabilities -> valid-box logit samples -> CPU float64 separable bilinear-knot least squares (gelsd,rcond1e-12) -> linear missing-edge-knot extrapolation -> direct original-grid sigmoid and exact xyxy crop; saturation uses known-box-support-normalized probability interpolation with explicit zero-support diagnostics; missing samples/rank/condition failures fall back to v1",
        "box_support": "integer pixel x>=x1,x<x2,y>=y1,y<y2; outside probability zero",
    }


class SoftMaskPredictor(SegmentationPredictor):
    """Standard Results plus aligned original and working probability tensors.

    Instantiate via YOLO(...).predict(predictor=SoftMaskPredictor,
    **PREDICT_DEFAULTS). Standard Results.cpu()/numpy()/slicing do not propagate
    custom fields: consume/copy these fields directly before transforming Results.
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if ultralytics.__version__ != LOCKED_ULTRALYTICS:
            raise RuntimeError(f"Source audited for {LOCKED_ULTRALYTICS}, installed {ultralytics.__version__}.")
        settings = {**PREDICT_DEFAULTS, **(overrides or {})}
        super().__init__(cfg, settings, _callbacks)
        if not self.args.retina_masks or self.args.half or self.args.rect or self.args.augment:
            raise ValueError("P2 requires retina_masks=True, half=False, rect=False, augment=False.")
        if self.args.imgsz not in (640, [640, 640], (640, 640)):
            raise ValueError("P2 inference input must remain imgsz=640.")

    def postprocess(self, preds, img, orig_imgs):
        if getattr(self.model, "end2end", False) or getattr(self, "_feats", None) is not None:
            raise ValueError("P2 only supports ordinary YOLO11 segmentation without embedding mode.")
        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        detected = inclusive_nms(
            preds[0], nc=len(self.model.names), threshold=self.args.conf,
            iou=self.args.iou, classes=self.args.classes,
            agnostic=self.args.agnostic_nms, max_det=self.args.max_det,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        return self.construct_results(detected, img, orig_imgs, protos)

    def construct_result(self, pred, img, orig_img, img_path, proto):
        # Standard native binary masks are produced by the actual locked library,
        # not by thresholding our new probability path. Filtering/order is shared.
        cache_result = CacheSoftMaskPredictor.construct_result(self, pred, img, orig_img, img_path, proto)
        pred = pred.clone()
        candidate_index = torch.arange(len(pred), device=pred.device)
        empty_candidate_index = candidate_index[:0]
        empty_soft_positive = torch.empty(0, dtype=torch.bool, device=pred.device)
        if len(pred):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            standard = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])
            probabilities = native_soft_masks(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])
            keep = standard.sum((-2, -1)) > 0
            empty_candidate_index = candidate_index[~keep]
            empty_soft_positive = (probabilities[~keep] > .5).flatten(1).any(1)
            pred, standard, probabilities = pred[keep], standard[keep], probabilities[keep]
            candidate_index = candidate_index[keep]
        else:
            standard = None
            probabilities = proto.new_empty((0, *orig_img.shape[:2]), dtype=torch.float32)
        result = Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=standard)
        result.soft_masks = probabilities
        result.work_soft_masks = cache_result.work_soft_masks[candidate_index]
        result.candidate_index = candidate_index
        result.p2_native_empty_candidate_index = empty_candidate_index
        result.p2_native_empty_soft_has_positive = empty_soft_positive
        result.p2_cache_candidates = cache_result
        result.p2_transform = transform_metadata(orig_img.shape, img.shape[2:], proto.shape[-2:])
        result.p2_nms = {**NMS_FIXED, "conf": float(float32_threshold(self.args.conf)),
                         "library_strict_prefilter": strict_nms_prefilter(self.args.conf),
                         "iou": self.args.iou, "max_det": self.args.max_det,
                         "agnostic": self.args.agnostic_nms, "classes": self.args.classes}
        return result


class CacheSoftMaskPredictor(SoftMaskPredictor):
    """Bounded working-scale production output, with every raw NMS candidate.

    ``masks`` is None: this class deliberately does not create official original-
    resolution binary masks. ``work_soft_masks`` is float16 N,Hw,Ww. Row position
    equals ``candidate_index`` and is retained in cache storage. Empty-mask and
    final-area filtering belong to the later final-output stage, not the cache.
    """
    def construct_result(self, pred, img, orig_img, img_path, proto):
        pred = pred.clone()
        if len(pred):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        work = direct_work_masks(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])
        result = Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=None)
        result.work_soft_masks = work
        result.candidate_index = torch.arange(len(pred), device=pred.device)
        result.p2_transform = transform_metadata(orig_img.shape, img.shape[2:], proto.shape[-2:])
        result.p2_transform["candidate_retention"] = "all post-NMS rows at recorded inclusive confidence threshold; no mask-empty/area filtering"
        result.p2_transform["candidate_index"] = "zero-based post-NMS row position, unchanged"
        result.p2_nms = {**NMS_FIXED, "conf": float(float32_threshold(self.args.conf)),
                         "library_strict_prefilter": strict_nms_prefilter(self.args.conf),
                         "iou": self.args.iou, "max_det": self.args.max_det,
                         "agnostic": self.args.agnostic_nms, "classes": self.args.classes}
        return result
