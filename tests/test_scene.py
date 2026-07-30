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
        self.assertEqual(
            describe(detections),
            "Podsumowanie obrazu, liczba wystąpień: Osoba raz.",
        )

    def test_deduplicates_and_limits(self):
        detections = [
            {"class": "person", "conf": 0.90}, {"class": "person", "conf": 0.80},
            {"class": "dog", "conf": 0.85}, {"class": "cat", "conf": 0.70},
            {"class": "car", "conf": 0.60},
        ]
        self.assertEqual(
            describe(detections),
            "Podsumowanie obrazu, liczba wystąpień: "
            "Osoby 2 razy. Pies raz. Kot raz.",
        )

    def test_reports_number_when_multiple_people_are_visible(self):
        detections = [
            {"class": "person", "conf": 0.91},
            {"class": "person", "conf": 0.88},
            {"class": "person", "conf": 0.72},
        ]

        self.assertEqual(
            describe(detections),
            "Podsumowanie obrazu, liczba wystąpień: Osoby 3 razy.",
        )

    def test_does_not_count_people_below_confidence_threshold(self):
        detections = [
            {"class": "person", "conf": 0.91},
            {"class": "person", "conf": 0.54},
        ]

        self.assertEqual(
            describe(detections),
            "Podsumowanie obrazu, liczba wystąpień: Osoba raz.",
        )

    def test_counts_every_supported_object_class(self):
        detections = [
            {"class": "bird", "conf": 0.92},
            {"class": "bird", "conf": 0.88},
            {"class": "bird", "conf": 0.84},
            {"class": "bird", "conf": 0.80},
            {"class": "cat", "conf": 0.90},
        ]

        self.assertEqual(
            describe(detections),
            "Podsumowanie obrazu, liczba wystąpień: "
            "Ptaki 4 razy. Kot raz.",
        )


if __name__ == "__main__":
    unittest.main()
