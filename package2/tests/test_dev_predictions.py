"""CPU-only native RLE/storage/provenance tests; no P5 scoring or GPU use."""
import copy
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest

import numpy as np
from pycocotools import mask as mask_utils
import torch
from ultralytics.utils import ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from collect_dev_predictions import (SelectionCandidateReader, SelectionCandidateWriter,
    FROZEN_MANIFEST_SHA256, P2_ROOT, PROOF_SCHEMA, REAL_SAMPLE_SEED, SYNTHETIC_NMS_NAMES, THRESHOLDS,
    epoch_checkpoints, make_predictor_class, native_original_rles, real_sample_ordinals,
    require_equal_runtime, sha256, validate_cuda_equivalence_report, verify_frozen_baseline)


def frame(index=0):
    return {"ordinal": index, "video_id": "dev-synthetic", "frame_idx": index * 6,
            "height": 13, "width": 19, "image_sha256": "fixture", "timestamp": index * .2,
            "p1_t_sec": index * .1999}


def candidate(index=0, confidence=.1, empty=False):
    mask = np.zeros((13, 19), np.uint8)
    if not empty:
        mask[2:9, 4:16] = 1
    encoded = mask_utils.encode(np.asfortranarray(mask))
    return {"candidate_index": index, "conf": np.float32(confidence), "box": np.array([0, 0, 19, 13], np.float32),
            "class_id": 0, "rle": {"size": list(encoded["size"]), "counts": encoded["counts"].decode()},
            "mask_area": int(mask_utils.area(encoded))}


class CandidateStorageTests(unittest.TestCase):
    def test_resume_empty_frames_empty_masks_and_float32_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch.sqlite"
            metadata = {"checkpoint_sha256": "test", "expected_frames": 3}
            writer = SelectionCandidateWriter(path, metadata)
            writer.append(frame(0), {}, [])
            writer.append(frame(1), {}, [candidate(empty=True), candidate(1, .55)])
            writer.close()
            writer = SelectionCandidateWriter(path, metadata)
            self.assertEqual(writer.next_frame, 2)
            writer.append(frame(2), {}, [candidate(confidence=.9)])
            writer.finish(3)
            writer.close()
            reader = SelectionCandidateReader(path)
            self.assertTrue(reader.complete)
            self.assertEqual(len(reader), 3)
            first, second, third = list(reader.iter_frames(.1))
            self.assertEqual(first["candidates"], [])
            self.assertEqual(second["candidates"][0]["mask_area"], 0)
            self.assertEqual(second["candidates"][0]["conf"].dtype, np.dtype("float32"))
            self.assertEqual(second["candidates"][1]["box"].dtype, np.dtype("float32"))
            self.assertEqual(mask_utils.decode(second["candidates"][0]["rle"]).sum(), 0)
            selected = list(reader.iter_frames(.55))
            self.assertEqual(selected[1]["candidates"][0]["candidate_index"], 1)
            self.assertEqual(len(selected[0]["candidates"]), 0)
            self.assertNotEqual(second["frame"]["timestamp"], second["frame"]["p1_t_sec"])
            self.assertEqual(len(list(reader.iter_frames(.9))[2]["candidates"]), 1)
            reader.close()

    def test_bad_frame_never_commits_and_cannot_finish_early(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch.sqlite"
            writer = SelectionCandidateWriter(path, {})
            bad = candidate(confidence=np.nextafter(np.float32(.1), np.float32(-np.inf)))
            with self.assertRaises(ValueError):
                writer.append(frame(), {}, [bad])
            self.assertEqual(writer.next_frame, 0)
            writer.append(frame(), {}, [])
            with self.assertRaises(ValueError):
                writer.finish(2)
            with self.assertRaises(ValueError):
                writer.append(frame(2), {}, [])
            writer.close()
            reader = SelectionCandidateReader(path)
            self.assertEqual(len(reader), 1)
            self.assertFalse(reader.complete)
            reader.close()

    def test_resume_rejects_changed_checkpoint_or_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch.sqlite"
            writer = SelectionCandidateWriter(path, {"checkpoint_sha256": "one", "config": {"imgsz": 640}})
            writer.close()
            with self.assertRaises(ValueError):
                SelectionCandidateWriter(path, {"checkpoint_sha256": "two", "config": {"imgsz": 640}})

    def test_database_failure_rolls_back_whole_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch.sqlite"
            writer = SelectionCandidateWriter(path, {"expected_frames": 1})
            writer.db.execute("""CREATE TRIGGER fail_second BEFORE INSERT ON candidates
                WHEN NEW.candidate_index = 1 BEGIN SELECT RAISE(FAIL, 'simulated disk write failure'); END;""")
            writer.db.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                writer.append(frame(), {}, [candidate(), candidate(1, .5)])
            self.assertEqual(writer.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0], 0)
            self.assertEqual(writer.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0], 0)
            self.assertEqual(writer.next_frame, 0)
            writer.db.execute("DROP TRIGGER fail_second")
            writer.db.commit()
            writer.append(frame(), {}, [])
            writer.close()

    def test_resume_rejects_false_completion_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch.sqlite"
            metadata = {"expected_frames": 3}
            writer = SelectionCandidateWriter(path, metadata)
            writer.append(frame(), {}, [])
            writer.close()
            with sqlite3.connect(path) as database:
                database.execute("UPDATE metadata SET value='true' WHERE key='complete'")
            with self.assertRaises(ValueError):
                SelectionCandidateWriter(path, metadata)

    def test_corrupted_mask_payload_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch.sqlite"
            writer = SelectionCandidateWriter(path, {})
            writer.append(frame(), {}, [candidate()])
            writer.finish(1)
            writer.close()
            with sqlite3.connect(path) as database:
                database.execute("UPDATE candidates SET mask_area=1")
            reader = SelectionCandidateReader(path)
            with self.assertRaises(ValueError):
                reader.read(0)
            reader.close()

    def test_native_checkpoint_enumeration_is_numeric_and_rejects_best(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("epoch10.pt", "epoch2.pt", "epoch0.pt", "best.pt"):
                (Path(directory) / name).touch()
            self.assertEqual([number for number, path in epoch_checkpoints(directory)], [0, 2, 10])
            with self.assertRaises(ValueError):
                epoch_checkpoints(explicit=[Path(directory) / "best.pt"])


class NativeOriginalRLETests(unittest.TestCase):
    def test_predictor_keeps_raw_candidate_order_and_zero_area_masks(self):
        from types import SimpleNamespace
        predictor_type = make_predictor_class()
        predictor = predictor_type.__new__(predictor_type)
        predictor.model = SimpleNamespace(names={0: "rip_current"})
        prediction = torch.tensor([[5, 5, 40, 50, .9, 0, 1],
                                   [8, 8, 50, 60, .8, 0, -1],
                                   [10, 20, 30, 40, .7, 0, 2]])
        result = predictor.construct_result(prediction, torch.zeros(1, 3, 64, 64),
                    np.zeros((37, 64, 3), np.uint8), "fixture.jpg", torch.ones(1, 16, 16))
        self.assertEqual(result.candidate_index.tolist(), [0, 1, 2])
        self.assertEqual(len(result.selection_candidates), 3)
        self.assertEqual(result.selection_candidates[1]["mask_area"], 0)
        self.assertGreater(result.selection_candidates[0]["mask_area"], 0)
        self.assertIsNone(result.masks)
        self.assertFalse(hasattr(result, "work_soft_masks"))
        for record in result.selection_candidates:
            self.assertEqual(record["rle"]["size"], [37, 64])

    def test_streamed_rles_equal_full_native_masks_with_empty_candidates(self):
        generator = torch.Generator().manual_seed(501)
        prototypes = torch.randn(4, 40, 40, generator=generator)
        coefficients = torch.randn(7, 4, generator=generator)
        for shape in ((79, 173), (173, 79), (133, 133)):
            h, w = shape
            boxes = torch.tensor([[.2, 3.7, w-3.2, h-5.1], [0, 0, w, h], [0, 0, 0, 0],
                                  [w/3, h/3, w/2, h/2], [0, 0, 1, 1], [1, 1, w-1, h-1], [10, 20, w-5, h-5]])
            official = ops.process_mask_native(prototypes, coefficients, boxes, shape).bool().numpy()
            records = native_original_rles(prototypes, coefficients, boxes, shape)
            self.assertEqual(len(records), len(coefficients))
            for index, record in enumerate(records):
                decoded = mask_utils.decode(record["rle"]).astype(bool)
                self.assertTrue(np.array_equal(decoded, official[index]))
                self.assertEqual(record["mask_area"], int(official[index].sum()))
            self.assertEqual(records[2]["mask_area"], 0)
        self.assertEqual(native_original_rles(prototypes, coefficients[:0], boxes[:0], (79, 173)), [])


def passing_proof_fixture():
    """Recorded-evidence fixture only; never claims a real GPU test was run."""
    inventory = {"signature": "source-signature", "frames": [
        {"image_rel": f"train/images/dev-{i}.jpg", "image_sha256": f"image-{i}", "height": 13, "width": 19}
        for i in range(10)]}
    runtime = {"device_type": "cuda", "device_index": 0, "device_name": "test GPU", "device_capability": [8, 9],
        "torch_version": "test-torch", "torch_cuda_version": "test-cuda", "torchvision_version": "test-tv",
        "cudnn_version": 0, "ultralytics_version": "test-ultra", "inference_dtype": "float32", "batch": 1,
        "matmul_allow_tf32": False, "cudnn_allow_tf32": True, "cudnn_deterministic": False, "cudnn_benchmark": True,
        "deterministic_algorithms": False, "float32_matmul_precision": "highest", "cublas_workspace_config": ":4096:8"}
    config = {"inference": {"imgsz": 640}, "nms": {"max_det": 300}, "device": "0",
              "torch": runtime["torch_version"], "ultralytics": runtime["ultralytics_version"],
              "source_sha256": {key: key for key in ("collector", "verifier", "soft_masks", "installed_ops", "installed_nms")}}
    def nms_case(name, raw_count):
        return {"name": name, "raw_candidates": raw_count, "retained_at_point1": 1, "pass": True, "input_unchanged": True,
                "threshold_results": [{"threshold": t, "direct_count": 1, "reused_count": 1,
                   "exact_box_conf_class_coefficients_and_order_equal": True} for t in THRESHOLDS]}
    def native_case(name, shape, instances):
        return {"name": name, "shape": shape, "instances": instances, "pass": True,
                "rows": [{"candidate_index": i, "area": 0, "native_pixels_equal": True} for i in range(instances)]}
    ordinals = real_sample_ordinals(10, 5)
    real = []
    for index in ordinals:
        source = inventory["frames"][index]
        real.append(dict(nms_case(source["image_rel"], 8400), source_ordinal=index, source_image_sha256=source["image_sha256"]))
    native = [native_case("synthetic_full_vs_sliced_native", list(shape), 17) for shape in ((79, 173), (173, 79), (128, 128))]
    native.extend(native_case(case["name"], [13, 19], 1) for case in real)
    report = {"schema": PROOF_SCHEMA, "kind": "p2_dev_candidate_reuse_equivalence", "status": "CUDA_WITH_REAL_OUTPUT_PASS",
              "cuda_requirement_met": True, "all_equal": True, "checkpoint_sha256": "checkpoint",
              "source_inventory_signature": inventory["signature"], "source_inventory_sha256": "inventory-sha",
              "source_sha256": copy.deepcopy(config["source_sha256"]), "inference": copy.deepcopy(config["inference"]),
              "nms": copy.deepcopy(config["nms"]), "runtime": runtime,
              "synthetic_nms_cases": [nms_case(name, count) for name, count in SYNTHETIC_NMS_NAMES.items()],
              "real_cases": real, "real_sampling": {"method": "numpy.default_rng.permutation-prefix", "seed": REAL_SAMPLE_SEED, "ordinals": ordinals},
              "real_nonempty_candidates": 5, "real_native_candidate_limit": 8, "native_rle_cases": native}
    return report, config, inventory


class EquivalenceGateTests(unittest.TestCase):
    def verify(self, report, config, inventory):
        return validate_cuda_equivalence_report(report, "checkpoint", config, inventory, "inventory-sha")

    def test_full_structural_fixture_passes_and_device_aliases_match(self):
        report, config, inventory = passing_proof_fixture()
        self.assertEqual(self.verify(report, config, inventory), report["runtime"])
        config["device"] = "cuda:0"
        self.verify(report, config, inventory)

    def test_pass_flags_cannot_replace_threshold_or_pixel_evidence(self):
        report, config, inventory = passing_proof_fixture()
        mutations = [
            lambda p: p["real_cases"][0]["threshold_results"].pop(),
            lambda p: p["synthetic_nms_cases"].pop(),
            lambda p: p["native_rle_cases"].pop(),
            lambda p: p["real_cases"][0]["threshold_results"][4].update(exact_box_conf_class_coefficients_and_order_equal=False),
            lambda p: p["native_rle_cases"][-1]["rows"][0].update(native_pixels_equal=False),
            lambda p: p["synthetic_nms_cases"][1].update(raw_candidates=299),
            lambda p: p["native_rle_cases"][0].update(instances=8),
        ]
        for mutate in mutations:
            altered = copy.deepcopy(report)
            mutate(altered)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.verify(altered, config, inventory)

    def test_cpu_or_different_provenance_is_rejected(self):
        report, config, inventory = passing_proof_fixture()
        for key, value in (("status", "CPU_ONLY"), ("checkpoint_sha256", "different"),
                           ("source_inventory_sha256", "changed"), ("source_inventory_signature", "changed"),
                           ("source_sha256", {}), ("inference", {}), ("nms", {})):
            altered = copy.deepcopy(report)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(altered, config, inventory)
        report["runtime"]["device_type"] = "cpu"
        with self.assertRaises(ValueError):
            self.verify(report, config, inventory)

    def test_wrong_or_cherry_picked_dev_samples_are_rejected(self):
        report, config, inventory = passing_proof_fixture()
        for change in (lambda p: p["real_cases"][0].update(source_image_sha256="different"),
                       lambda p: p["real_cases"][0].update(name="val/images/forbidden.jpg"),
                       lambda p: p["real_cases"].reverse(),
                       lambda p: p["real_sampling"].update(seed=123),
                       lambda p: p["real_cases"].pop()):
            altered = copy.deepcopy(report)
            change(altered)
            with self.subTest(mutation=change), self.assertRaises(ValueError):
                self.verify(altered, config, inventory)

    def test_all_empty_real_outputs_cannot_certify_native_reuse(self):
        report, config, inventory = passing_proof_fixture()
        for case in report["real_cases"]:
            case["retained_at_point1"] = 0
            for row in case["threshold_results"]:
                row.update(direct_count=0, reused_count=0)
        report["real_nonempty_candidates"] = 0
        with self.assertRaises(ValueError):
            self.verify(report, config, inventory)

    def test_runtime_flags_must_match_after_predictor_setup(self):
        report, _, _ = passing_proof_fixture()
        validated = report["runtime"]
        require_equal_runtime(validated, validated)
        for key, value in (("cudnn_benchmark", False), ("matmul_allow_tf32", True), ("inference_dtype", "float16"),
                           ("device_name", "other GPU"), ("batch", 2), ("device_index", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_equal_runtime(dict(validated, **{key: value}), validated)

    def test_fixed_sample_is_distinct_and_larger_scope_extends_it(self):
        first = real_sample_ordinals(2421, 5)
        self.assertEqual(len(set(first)), 5)
        self.assertEqual(real_sample_ordinals(2421, 8)[:5], first)
        self.assertEqual(real_sample_ordinals(2421, 5), first)

    def test_frozen_baseline_cannot_be_replaced_by_self_recorded_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "upstream/data/manifests"
            folder.mkdir(parents=True)
            shutil.copyfile(P2_ROOT / "upstream_manifest.json", root / "upstream_manifest.json")
            for name in FROZEN_MANIFEST_SHA256:
                shutil.copyfile(P2_ROOT / "upstream/data/manifests" / name, folder / name)
            verify_frozen_baseline(root)
            modified = folder / "dev_videos.txt"
            modified.write_text("changed dev set\n")
            baseline = json.loads((root / "upstream_manifest.json").read_text())
            baseline["files"]["data/manifests/dev_videos.txt"] = sha256(modified)
            (root / "upstream_manifest.json").write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError, "acquisition baseline"):
                verify_frozen_baseline(root)


if __name__ == "__main__":
    unittest.main()
