"""CPU tests for exact protocol boundaries and independently generated native masks."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.utils import ops

from p2.soft_masks import (CacheSoftMaskPredictor, SoftMaskPredictor, _support_normalized_restore_one,
    confidence_keep, direct_work_masks, encode_work_masks,
    geometry_inverse_single, inclusive_nms, native_soft_masks, quantize_probabilities, restore_work_masks,
    sign_preserving_sigmoid, strict_nms_prefilter, transform_metadata, work_shape)


class ConfidenceTests(unittest.TestCase):
    def test_all_search_boundaries_are_inclusive(self):
        for integer in range(10, 91, 5):
            threshold = np.float32(integer / 100)
            scores = torch.tensor([
                np.nextafter(threshold, np.float32(-np.inf)), threshold,
                np.nextafter(threshold, np.float32(np.inf)),
            ], dtype=torch.float32)
            self.assertEqual(confidence_keep(scores, float(threshold)).tolist(), [False, True, True])
            self.assertEqual((scores > strict_nms_prefilter(float(threshold))).tolist(), [False, True, True])
            raw = torch.zeros(1, 6, 3)  # 4 xywh + 1 class + 1 mask coefficient
            raw[0, 0] = torch.tensor([5, 25, 45])
            raw[0, 1:4] = 4
            raw[0, 4] = scores
            raw[0, 5] = torch.tensor([11, 22, 33])
            got = inclusive_nms(raw, nc=1, threshold=float(threshold))[0]
            self.assertEqual(got[:, 6].tolist(), [33, 22])

    def test_cached_candidates_equal_direct_threshold_nms(self):
        generator = torch.Generator().manual_seed(42)
        raw = torch.rand(1, 7, 207, generator=generator)
        raw[:, :2] *= 640
        raw[:, 2:4] = raw[:, 2:4] * 150 + 1
        # Include exact boundaries, adjacent floats, overlaps and tied confidences.
        for i, integer in enumerate(range(10, 91, 5)):
            t = np.float32(integer / 100)
            raw[0, 4, i*3:i*3+3] = torch.tensor([
                np.nextafter(t, np.float32(-np.inf)), t, np.nextafter(t, np.float32(np.inf))])
        raw[0, :4, 61] = raw[0, :4, 60]
        raw[0, 4, 61] = raw[0, 4, 60]
        original = raw.clone()
        cached = inclusive_nms(raw, nc=1, threshold=.1)[0]
        for integer in range(10, 91, 5):
            threshold = integer / 100
            expected = inclusive_nms(raw, nc=1, threshold=threshold)[0]
            selected = cached[confidence_keep(cached[:, 4], threshold)]
            torch.testing.assert_close(selected, expected, rtol=0, atol=0)
        torch.testing.assert_close(raw, original, rtol=0, atol=0)

    def test_invalid_thresholds_fail(self):
        for value in (0, -1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                strict_nms_prefilter(value)


class NativeGeometryTests(unittest.TestCase):
    def test_native_binary_exact_with_fractional_boxes_and_padding(self):
        generator = torch.Generator().manual_seed(91)
        for shape in ((79, 173), (173, 79), (133, 133), (103, 181)):
            prototypes = torch.randn(4, 40, 40, generator=generator)
            coefficients = torch.randn(3, 4, generator=generator)
            h, w = shape
            boxes = torch.tensor([[.2, 3.7, w-3.2, h-5.1], [0, 0, w, h], [w/3, h/3, w/2, h/2]])
            standard = ops.process_mask_native(prototypes, coefficients, boxes, shape).bool()
            probabilities = native_soft_masks(prototypes, coefficients, boxes, shape)
            self.assertTrue(torch.equal(probabilities > .5, standard))
            support = ops.crop_mask(torch.ones_like(probabilities), boxes).bool()
            self.assertTrue(torch.equal(probabilities[~support], torch.zeros_like(probabilities[~support])))
            self.assertTrue(bool(((probabilities > 0) & (probabilities < 1)).any()))

    def test_tiny_logits_keep_strict_positive_binary_sign(self):
        logits = torch.tensor([-1e-12, 0., 1e-12, -1., 1.], dtype=torch.float32)
        got = sign_preserving_sigmoid(logits)
        self.assertTrue(torch.equal(got > .5, logits > 0))
        self.assertGreater(float(got[2]), .5)
        self.assertEqual(float(got[1]), .5)

    def test_sigmoid_before_interpolation_is_not_equivalent(self):
        logits = torch.tensor([[[[-10., 1.]]]])
        correct = F.interpolate(logits, (1, 5), mode="bilinear", align_corners=False).sigmoid()
        wrong = F.interpolate(logits.sigmoid(), (1, 5), mode="bilinear", align_corners=False)
        self.assertFalse(torch.equal(correct > .5, wrong > .5))

    def test_standard_results_and_soft_instances_keep_same_order(self):
        predictor = SoftMaskPredictor.__new__(SoftMaskPredictor)
        predictor.args = SimpleNamespace(conf=.1, iou=.7, max_det=300, agnostic_nms=False, classes=None)
        predictor.model = SimpleNamespace(names={0: "rip"})
        proto = torch.ones(1, 16, 16)
        # Coefficients: positive, negative (empty standard mask), positive.
        pred = torch.tensor([[5, 5, 40, 50, .9, 0, 1], [8, 8, 50, 60, .8, 0, -1], [10, 20, 30, 40, .7, 0, 2]])
        img = torch.zeros(1, 3, 64, 64)
        original = np.zeros((37, 64, 3), dtype=np.uint8)
        result = predictor.construct_result(pred, img, original, "synthetic.jpg", proto)
        self.assertEqual(len(result.boxes), 2)
        torch.testing.assert_close(result.boxes.conf, torch.tensor([.9, .7]))
        self.assertTrue(torch.equal(result.soft_masks > .5, result.masks.data.bool()))
        self.assertEqual(result.work_soft_masks.shape, (2, 370, 640))
        self.assertEqual(result.soft_masks.dtype, torch.float32)
        self.assertEqual(result.work_soft_masks.dtype, torch.float16)
        self.assertEqual(result.candidate_index.tolist(), [0, 2])
        self.assertEqual(result.p2_cache_candidates.candidate_index.tolist(), [0, 1, 2])
        self.assertEqual(len(result.p2_cache_candidates.boxes), 3)
        self.assertEqual(float(result.p2_cache_candidates.work_soft_masks[1].max()) > .5, False)
        self.assertIsNone(result.p2_cache_candidates.masks)
        self.assertEqual(result.p2_native_empty_candidate_index.tolist(), [1])
        self.assertEqual(result.p2_native_empty_soft_has_positive.tolist(), [False])
        restored, details = restore_work_masks(result.p2_cache_candidates.work_soft_masks,
            original.shape[:2], result.p2_cache_candidates.boxes.xyxy,
            transform=result.p2_cache_candidates.p2_transform, return_diagnostics=True)
        self.assertFalse(bool((restored[1] > .5).any()))
        self.assertEqual(details[1]["status"], "reconstructed")

    def test_empty_frame_is_representable(self):
        predictor = SoftMaskPredictor.__new__(SoftMaskPredictor)
        predictor.args = SimpleNamespace(conf=.1, iou=.7, max_det=300, agnostic_nms=False, classes=None)
        predictor.model = SimpleNamespace(names={0: "rip"})
        result = predictor.construct_result(torch.empty(0, 7), torch.zeros(1, 3, 640, 640),
            np.zeros((91, 173, 3), dtype=np.uint8), "empty.jpg", torch.zeros(1, 160, 160))
        self.assertIsNone(result.masks)
        self.assertEqual(result.soft_masks.shape, (0, 91, 173))
        self.assertEqual(result.work_soft_masks.shape, (0, 337, 640))


class WorkingScaleTests(unittest.TestCase):
    def test_production_encoder_is_bounded_at_work_resolution(self):
        boxes = torch.tensor([[0., 0., 7680., 4320.]])
        probabilities = direct_work_masks(torch.ones(1, 160, 160), torch.ones(1, 1), boxes, (4320, 7680))
        self.assertEqual(probabilities.shape, (1, 360, 640))
        self.assertEqual(probabilities.dtype, torch.float16)
        self.assertTrue(bool((probabilities > .5).all()))

    def test_production_roundtrip_uses_same_box_and_padding_geometry(self):
        boxes = torch.tensor([[53.3, 79.4, 1503.2, 901.7]])
        prototypes, coefficients = torch.ones(1, 160, 160), torch.ones(1, 1)
        original = native_soft_masks(prototypes, coefficients, boxes, (1080, 1920))
        working = direct_work_masks(prototypes, coefficients, boxes, (1080, 1920))
        scaled_boxes = boxes * torch.tensor([640/1920, 360/1080, 640/1920, 360/1080])
        support = ops.crop_mask(torch.ones_like(working), scaled_boxes).bool()
        self.assertEqual(float(working[~support].max()), 0.)
        restored = restore_work_masks(working.numpy(), (1080, 1920), boxes.numpy())
        truth, prediction = original > .5, restored > .5
        self.assertGreaterEqual(float((truth & prediction).sum()/(truth | prediction).sum()), .98)

    def test_aspect_dimensions_and_geometry_metadata(self):
        self.assertEqual(work_shape((1080, 1920)), (360, 640))
        self.assertEqual(work_shape((1920, 1080)), (640, 360))
        meta = transform_metadata((1080, 1920), (640, 640), (160, 160))
        self.assertEqual(meta["letterbox_padding_ltrb"], [0, 140, 0, 140])
        self.assertEqual(meta["prototype_crop_tlbr"], [35, 0, 125, 160])

    def test_float16_storage_preserves_working_pixel_binary_sign(self):
        values = torch.tensor([0, .49999999, .5, .50000006, .5001, .8, 1.])
        quantized = quantize_probabilities(values)
        self.assertEqual(quantized.dtype, torch.float16)
        self.assertTrue(torch.equal(values > .5, quantized > .5))

    def test_roundtrip_rectangular_probability_and_zero_outside(self):
        boxes = torch.tensor([[53., 79., 503., 601.]])
        probability = ops.crop_mask(torch.ones(1, 720, 1280) * .8, boxes)
        working = encode_work_masks(probability).cpu().numpy()
        restored = restore_work_masks(working, (720, 1280), boxes.numpy())
        truth, predicted = probability > .5, restored > .5
        iou = (truth & predicted).sum() / (truth | predicted).sum()
        self.assertGreaterEqual(float(iou), .98)
        support = ops.crop_mask(torch.ones_like(restored), boxes).bool()
        self.assertEqual(float(restored[~support].max()), 0.)
        self.assertEqual(restore_work_masks(np.empty((0, 360, 640), dtype=np.float16),
            (720, 1280), np.empty((0, 4), dtype=np.float32)).shape, (0, 720, 1280))


class GeometryInverseTests(unittest.TestCase):
    def test_support_normalized_fallback_includes_valid_zero_probabilities(self):
        work = torch.tensor([[0., 1.], [0., 1.]])
        restored, details = _support_normalized_restore_one(work, torch.tensor([0.,0.,4.,4.]), (4,4))
        self.assertEqual(float(restored[1,1]), .25)
        self.assertEqual(float(restored[1,2]), .75)
        self.assertEqual(details["work_support_pixels"], 4)
        self.assertEqual(details["original_box_zero_support_pixels"], 0)
        for value in (0., .25, .5):
            restored, _ = _support_normalized_restore_one(torch.full((2,2),value),
                torch.tensor([0.,0.,4.,4.]), (4,4))
            self.assertFalse(bool((restored>.5).any()))

    def test_support_normalized_fallback_zero_support_is_explicit(self):
        restored, details = _support_normalized_restore_one(torch.zeros((2,2)),
            torch.tensor([1.,1.,1.1,1.1]), (4,4))
        self.assertFalse(bool((restored!=0).any()))
        self.assertTrue(details["has_original_box_zero_support"])
        self.assertEqual(details["original_box_zero_support_pixels"], 1)
        self.assertEqual(details["work_support_pixels"], 0)

    def test_v3_changes_only_saturation_fallback_and_preserves_v2_dispatch(self):
        shape, boxes = (241,389), torch.tensor([[31.2,17.3,351.4,223.8]])
        metadata = transform_metadata(shape,(640,640),(16,16))
        self.assertEqual(metadata["format_version"], "p2-soft-mask-v3")
        metadata_v2 = {**metadata,"format_version":"p2-soft-mask-v2"}
        saturated = direct_work_masks(torch.full((1,16,16),20.),torch.ones((1,1)),boxes,shape)
        v2, old_detail = restore_work_masks(saturated,shape,boxes,transform=metadata_v2,return_diagnostics=True)
        legacy = restore_work_masks(saturated,shape,boxes)
        self.assertTrue(torch.equal(v2,legacy))
        v3, new_detail = restore_work_masks(saturated,shape,boxes,transform=metadata,return_diagnostics=True)
        original_support = ops.crop_mask(torch.ones((1,*shape)),boxes).bool()
        self.assertTrue(torch.equal(v3>.5,original_support))
        self.assertEqual(new_detail[0]["status"],"fallback")
        self.assertFalse(new_detail[0]["knot_recovery_claimed"])
        self.assertIn("fallback_probability_method",new_detail[0])
        self.assertNotIn("fallback_probability_method",old_detail[0])
        nonsaturated = direct_work_masks(torch.full((1,16,16),.3),torch.ones((1,1)),boxes,shape)
        old = restore_work_masks(nonsaturated,shape,boxes,transform=metadata_v2)
        new = restore_work_masks(nonsaturated,shape,boxes,transform=metadata)
        self.assertTrue(torch.equal(old,new))

    def test_known_nonlinear_knots_survive_float16_probabilities(self):
        y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
        proto = (torch.sin(x*.7)+torch.cos(y*.5))[None]
        shape = (241, 389)
        boxes = torch.tensor([[31.2, 17.3, 351.4, 223.8]])
        working = direct_work_masks(proto, torch.ones(1, 1), boxes, shape)
        restored, details = restore_work_masks(working, shape, boxes,
            transform=transform_metadata(shape, (640, 640), (16, 16)), return_diagnostics=True)
        native = native_soft_masks(proto, torch.ones(1, 1), boxes, shape)
        reference, actual = native > .5, restored > .5
        self.assertGreater(float((reference & actual).sum()/(reference | actual).sum()), .99)
        self.assertEqual(details[0]["status"], "reconstructed")
        self.assertLess(max(details[0]["condition_estimates_hw"]), 1e8)
        self.assertEqual(details[0]["rcond"], 1e-12)

    def test_saturated_and_narrow_samples_are_explicit_fallbacks(self):
        for value in (0., 1.):
            restored, detail = geometry_inverse_single(torch.full((8, 8), value),
                [0., 0., 16., 16.], (16, 16), (2, 2))
            self.assertEqual(detail["status"], "fallback")
            self.assertEqual(detail["reason"], "saturated_probability_has_unbounded_logit_interval")
            self.assertFalse(detail["knot_recovery_claimed"])
            self.assertTrue(bool(torch.isfinite(restored).all()))
        restored, detail = geometry_inverse_single(torch.zeros((8, 8)),
            [1., 1., 1.1, 1.1], (16, 16), (2, 2))
        self.assertEqual(detail["reason"], "no_interior_work_samples")
        self.assertFalse(bool((restored > .5).any()))
        _, detail = geometry_inverse_single(torch.full((8, 8), .6),
            [6., 6., 8., 8.], (16, 16), (4, 4))
        self.assertEqual(detail["reason"], "insufficient_rank")

    def test_missing_edge_knots_are_an_explicit_approximation(self):
        # Work samples cannot observe the first knot, while native crop contains
        # pixels influenced by it. A nonlinear true edge is not identifiable.
        knots = torch.tensor([[[[-3., 2., -1., 2.], [-3., 2., -1., 2.],
                                [-3., 2., -1., 2.], [-3., 2., -1., 2.]]]])
        work = F.interpolate(knots, (16, 16), mode="bilinear", align_corners=False)[0, 0].sigmoid().half()
        box = torch.tensor([21., 0., 64., 64.])
        work = ops.crop_mask(work[None], (box*torch.tensor([.25]*4))[None])[0]
        _, detail = geometry_inverse_single(work, box, (64, 64), (4, 4))
        self.assertEqual(detail["status"], "reconstructed")
        self.assertEqual(detail["missing_target_knots_w"], 1)
        self.assertIn("assumption", detail["missing_knots_policy"])

    def test_v2_requires_explicit_matching_raw_encoder(self):
        masks = np.empty((0, 360, 640), dtype=np.float16)
        boxes = np.empty((0, 4), dtype=np.float32)
        metadata = transform_metadata((1080, 1920), (640, 640), (160, 160))
        restored, details = restore_work_masks(masks, (1080, 1920), boxes,
            transform=metadata, return_diagnostics=True)
        self.assertEqual(tuple(restored.shape), (0, 1080, 1920))
        self.assertEqual(details, [])
        for field, wrong in (("work_encoder", "downsample_original_probabilities"),
                             ("inverse_applicability", "temporally_aggregated"),
                             ("format_version", "unknown")):
            with self.assertRaises(ValueError):
                restore_work_masks(masks, (1080, 1920), boxes, transform={**metadata, field: wrong})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
