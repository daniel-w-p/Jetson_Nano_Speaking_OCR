import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scene import describe


class DescribeTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(describe([]), "Nic pewnego nie widzę.")

    def test_filters_low_confidence_and_unknown(self):
        detections = [
            {"class": "dog", "conf": 0.54},
            {"class": "banana", "conf": 0.99},
            {"class": "person", "conf": 0.90},
        ]
        self.assertEqual(describe(detections), "Widzę osobę.")

    def test_deduplicates_and_limits(self):
        detections = [
            {"class": "person", "conf": 0.90}, {"class": "person", "conf": 0.80},
            {"class": "dog", "conf": 0.85}, {"class": "cat", "conf": 0.70},
            {"class": "car", "conf": 0.60},
        ]
        self.assertEqual(describe(detections), "Widzę osobę, psa i kota.")


if __name__ == "__main__":
    unittest.main()
