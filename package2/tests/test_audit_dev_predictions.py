"""Focused CPU audit tests; actual COCO codec cases need pycocotools, never torch."""
from pathlib import Path
from contextlib import closing
import shutil
import sqlite3
import sys
import unittest
import uuid

import numpy as np

P2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P2 / "tools"))
from audit_dev_predictions import audit_database
from collect_dev_predictions import KIND, SelectionCandidateReader, SelectionCandidateWriter

try:
    from pycocotools import mask as mask_utils
except ImportError:
    mask_utils = None


def frame(ordinal):
    return {"ordinal": ordinal, "video_id": "test-dev", "frame_idx": ordinal,
            "height": 3, "width": 4, "image_sha256": "synthetic-only", "image_id": ordinal}


TRANSFORM = {"original_shape_hw": [3, 4], "artifact_kind": KIND,
             "mask_encoding": "original native binary logit>0 COCO RLE; no fp16-work-mask dependency"}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = P2 / "tmp" / ("audit_dev_test_" + uuid.uuid4().hex)
        self.temp.mkdir(parents=True)
        self.path = self.temp / "synthetic.sqlite"
        self.sources = [frame(0), frame(1)]

    def tearDown(self):
        target = self.temp.resolve()
        target.relative_to((P2 / "tmp").resolve())
        if not target.name.startswith("audit_dev_test_"):
            raise RuntimeError("Refusing to remove an unexpected fixture path")
        shutil.rmtree(target)

    def write(self, candidates=None, complete=True):
        writer = SelectionCandidateWriter(self.path, {"expected_frames": 2})
        writer.append(self.sources[0], TRANSFORM, candidates or [])
        writer.append(self.sources[1], TRANSFORM, [])
        if complete:
            writer.finish(2)
        writer.close()

    def audit(self):
        reader = SelectionCandidateReader(self.path)
        try:
            return audit_database(reader, self.sources, mask_utils)
        finally:
            reader.close()

    def test_complete_empty_rows_are_counted_at_all_thresholds(self):
        self.write()
        result = self.audit()
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["candidate_instances"], 0)
        self.assertEqual(len(result["threshold_counts"]), 17)
        self.assertTrue(all(row["empty_prediction_frames"] == 2 for row in result["threshold_counts"]))

    def test_incomplete_flag_fails(self):
        self.write(complete=False)
        with self.assertRaisesRegex(ValueError, "not marked complete"):
            self.audit()

    def test_false_complete_scope_fails(self):
        self.write()
        with closing(sqlite3.connect(self.path)) as database:
            database.execute("DELETE FROM frames WHERE ordinal=1")
            database.commit()
        with self.assertRaisesRegex(ValueError, "every expected dev row"):
            self.audit()

    def test_existing_reader_detects_payload_corruption(self):
        self.write()
        with closing(sqlite3.connect(self.path)) as database:
            database.execute("UPDATE frames SET payload_sha256='modified' WHERE ordinal=0")
            database.commit()
        with self.assertRaisesRegex(ValueError, "corrupted"):
            self.audit()

    def test_source_identity_mismatch_fails(self):
        self.write()
        self.sources[0]["image_id"] = 99
        with self.assertRaisesRegex(ValueError, "source identity"):
            self.audit()

    def test_orphan_candidates_fail(self):
        self.write()
        with closing(sqlite3.connect(self.path)) as database:
            database.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?)",
                             (99, 0, b"", b"", 0, 3, 4, b"", 0))
            database.commit()
        with self.assertRaisesRegex(ValueError, "Orphan"):
            self.audit()

    @staticmethod
    def candidate(index, confidence, empty=False):
        mask = np.zeros((3, 4), dtype=np.uint8)
        if not empty:
            mask[1:3, 1:4] = 1
        encoded = mask_utils.encode(np.asfortranarray(mask))
        return {"candidate_index": index, "conf": np.float32(confidence), "class_id": 0,
                "box": np.asarray([0, 0, 4, 3], np.float32),
                "rle": {"size": list(encoded["size"]), "counts": encoded["counts"].decode("ascii")},
                "mask_area": int(mask_utils.area(encoded))}

    @unittest.skipIf(mask_utils is None, "pycocotools is unavailable in this Python runtime")
    def test_real_rle_and_inclusive_float32_thresholds(self):
        self.write([self.candidate(0, .1, empty=True), self.candidate(1, .55),
                    self.candidate(2, np.nextafter(np.float32(.55), np.float32(-np.inf))),
                    self.candidate(3, .9)])
        result = self.audit()
        counts = {row["threshold"]: row for row in result["threshold_counts"]}
        self.assertEqual(result["zero_area_candidate_masks"], 1)
        self.assertEqual(result["nonempty_candidate_masks"], 3)
        self.assertEqual(counts[.1]["candidate_instances"], 4)
        self.assertEqual(counts[.55]["candidate_instances"], 2)
        self.assertEqual(counts[.9]["candidate_instances"], 1)

    @unittest.skipIf(mask_utils is None, "pycocotools is unavailable in this Python runtime")
    def test_structurally_valid_but_wrong_recorded_mask_area_fails(self):
        bad = self.candidate(0, .5)
        bad["mask_area"] = 1
        self.write([bad])
        with self.assertRaisesRegex(ValueError, "RLE area differs"):
            self.audit()

    @unittest.skipIf(mask_utils is None, "pycocotools is unavailable in this Python runtime")
    def test_lossless_but_noncanonical_rle_fails(self):
        bad = self.candidate(0, .5, empty=True)
        bad["rle"]["counts"] += "0"  # An extra zero-length foreground run changes no pixels.
        self.write([bad])
        with self.assertRaisesRegex(ValueError, "canonical lossless"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
