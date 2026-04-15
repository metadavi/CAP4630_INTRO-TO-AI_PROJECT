"""
Unit Tests — ALPR Pipeline
===========================
Tests the pure-logic components without requiring trained model weights
or a real camera feed.

Run with:
    python -m pytest tests/ -v
    # or
    python tests/test_pipeline.py
"""

import sys
import os
import unittest
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── CharSegmenter ─────────────────────────────────────────────────────────

class TestCharSegmenter(unittest.TestCase):

    def setUp(self):
        from src.segmenter import CharSegmenter
        self.seg = CharSegmenter()

    def test_empty_input_returns_empty(self):
        result = self.seg.segment(np.array([]))
        self.assertEqual(result, [])

    def test_blank_image_returns_empty(self):
        blank = np.zeros((40, 200), dtype=np.uint8)
        result = self.seg.segment(blank)
        self.assertEqual(result, [])

    def test_synthetic_characters_detected(self):
        """Three white rectangles on a black background = 3 character crops."""
        img = np.zeros((60, 300), dtype=np.uint8)
        # Draw 3 fake "characters"
        for i in range(3):
            x = 20 + i * 90
            img[10:50, x:x + 30] = 255

        result = self.seg.segment(img)
        self.assertEqual(len(result), 3,
                         f"Expected 3 crops, got {len(result)}")

    def test_crops_are_square(self):
        img = np.zeros((60, 300), dtype=np.uint8)
        img[10:50, 20:50] = 255
        result = self.seg.segment(img)
        if result:
            h, w = result[0].shape[:2]
            self.assertEqual(h, w, "Crop should be square after padding")


# ── FrameVoter ────────────────────────────────────────────────────────────

class TestFrameVoter(unittest.TestCase):

    def setUp(self):
        from src.voter import FrameVoter
        self.voter = FrameVoter(window=5, min_hits=2)

    def test_returns_none_before_window_full(self):
        for _ in range(4):
            result = self.voter.update("ABC1234", 0.80)
        self.assertIsNone(result)

    def test_returns_winner_when_window_full(self):
        for _ in range(5):
            self.voter.update("ABC1234", 0.90)
        result = self.voter.update("ABC1234", 0.90)
        self.assertIsNotNone(result)
        plate, conf = result
        self.assertEqual(plate, "ABC1234")
        self.assertGreater(conf, 0.0)

    def test_empty_readings_ignored(self):
        for _ in range(5):
            self.voter.update("", 0.0)
        result = self.voter.update("ABC1234", 0.85)
        # Buffer never had a consistent candidate
        # (all empty readings are dropped)
        self.assertIsNone(result)

    def test_reset_clears_buffer(self):
        for _ in range(5):
            self.voter.update("ABC1234", 0.90)
        self.voter.reset()
        self.assertEqual(self.voter.buffer_size, 0)

    def test_min_hits_threshold(self):
        """With min_hits=3, two identical readings should not decide."""
        from src.voter import FrameVoter
        voter = FrameVoter(window=5, min_hits=3)
        voter.update("XYZ999", 0.90)
        voter.update("XYZ999", 0.90)
        voter.update("OTHER1", 0.90)
        voter.update("OTHER2", 0.90)
        result = voter.update("OTHER3", 0.90)
        self.assertIsNone(result)


# ── AccessController ──────────────────────────────────────────────────────

class TestAccessController(unittest.TestCase):

    def setUp(self):
        import tempfile, csv
        # Create a temporary whitelist
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        )
        writer = csv.DictWriter(
            self.tmp, fieldnames=["plate", "owner", "notes", "expires"]
        )
        writer.writeheader()
        writer.writerow({"plate": "ABC1234", "owner": "Test",
                         "notes": "", "expires": ""})
        writer.writerow({"plate": "EXP0001", "owner": "Expired",
                         "notes": "", "expires": "2020-01-01"})
        self.tmp.close()

        from src.access_control import AccessController
        self.ac = AccessController(whitelist_path=self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_allowed_plate_high_confidence(self):
        from src.access_control import Decision
        decision, _ = self.ac.check("ABC1234", confidence=0.90)
        self.assertEqual(decision, Decision.ALLOWED)

    def test_denied_plate_not_on_whitelist(self):
        from src.access_control import Decision
        decision, _ = self.ac.check("ZZZ9999", confidence=0.90)
        self.assertEqual(decision, Decision.DENIED)

    def test_uncertain_low_confidence(self):
        from src.access_control import Decision
        decision, _ = self.ac.check("ABC1234", confidence=0.10)
        self.assertEqual(decision, Decision.UNCERTAIN)

    def test_expired_plate_denied(self):
        from src.access_control import Decision
        decision, info = self.ac.check("EXP0001", confidence=0.95)
        self.assertEqual(decision, Decision.DENIED)
        self.assertEqual(info.get("reason"), "expired")

    def test_add_and_remove_plate(self):
        from src.access_control import Decision
        self.ac.add_plate("NEW001", owner="New Owner")
        decision, _ = self.ac.check("NEW001", confidence=0.90)
        self.assertEqual(decision, Decision.ALLOWED)

        removed = self.ac.remove_plate("NEW001")
        self.assertTrue(removed)
        decision2, _ = self.ac.check("NEW001", confidence=0.90)
        self.assertEqual(decision2, Decision.DENIED)


# ── EventLogger ───────────────────────────────────────────────────────────

class TestEventLogger(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.log_dir = tempfile.mkdtemp()
        from src.logger import EventLogger
        self.logger = EventLogger(log_dir=self.log_dir)

    def test_log_creates_file(self):
        self.logger.log("ABC1234", 0.90, "ALLOWED", "Test Owner")
        files = os.listdir(self.log_dir)
        self.assertTrue(any(f.endswith(".csv") for f in files))

    def test_recent_returns_logged_event(self):
        self.logger.log("XYZ999", 0.75, "DENIED")
        recent = self.logger.recent(n=5)
        self.assertTrue(len(recent) >= 1)
        self.assertEqual(recent[0]["plate"], "XYZ999")
        self.assertEqual(recent[0]["decision"], "DENIED")


# ── OCRReader (unit — no CNN weights needed for segmentation path) ────────

class TestOCRReaderFallback(unittest.TestCase):
    """
    Tests that OCRReader gracefully returns ('', 0.0) when there
    are no segmentable characters (no trained weights required).
    """

    def test_empty_crop_returns_empty(self):
        # We need to instantiate without loading weights — mock the classifier
        import unittest.mock as mock
        from src.ocr_reader import OCRReader
        from src.detector   import PlateDetector
        from src.segmenter  import CharSegmenter

        with mock.patch("src.char_classifier.CharClassifier.__init__",
                        return_value=None):
            with mock.patch("os.path.exists", return_value=False):
                reader = OCRReader.__new__(OCRReader)
                reader.detector   = PlateDetector.__new__(PlateDetector)
                reader.segmenter  = CharSegmenter()
                reader.classifier = mock.MagicMock()
                reader.classifier.predict_batch.return_value = []

                result = reader.read(np.zeros((40, 200, 3), dtype=np.uint8))
                # Black image → no characters segmented → ("", 0.0)
                self.assertEqual(result, ("", 0.0))


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
