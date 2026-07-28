import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config


class ConfigTests(unittest.TestCase):
    def test_default_ocr_settings_have_valid_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.json"
            with mock.patch.object(config, "CONFIG_PATH", missing_path):
                settings = config.load_config()["ocr"]

        self.assertTrue(
            {
                "language",
                "page_segmentation_mode",
                "max_characters",
                "scale_factor",
                "threshold_block_size",
                "threshold_c",
                "page_detection",
            }.issubset(settings)
        )
        self.assertGreater(settings["scale_factor"], 0)
        self.assertGreater(settings["threshold_block_size"], 1)
        self.assertEqual(settings["threshold_block_size"] % 2, 1)

        detection = settings["page_detection"]
        self.assertTrue(
            {
                "enabled",
                "max_width",
                "blur_kernel",
                "canny_low",
                "canny_high",
                "contour_candidates",
                "approx_epsilon_ratio",
                "min_page_area_ratio",
                "debug_images",
            }.issubset(detection)
        )
        self.assertIsInstance(detection["enabled"], bool)
        self.assertIsInstance(detection["debug_images"], bool)
        self.assertGreater(detection["max_width"], 0)
        self.assertGreater(detection["blur_kernel"], 0)
        self.assertEqual(detection["blur_kernel"] % 2, 1)
        self.assertGreaterEqual(detection["canny_low"], 0)
        self.assertGreater(detection["canny_high"], detection["canny_low"])
        self.assertGreater(detection["contour_candidates"], 0)
        self.assertGreater(detection["approx_epsilon_ratio"], 0)
        self.assertLess(detection["approx_epsilon_ratio"], 1)
        self.assertGreater(detection["min_page_area_ratio"], 0)
        self.assertLess(detection["min_page_area_ratio"], 1)

    def test_nested_override_keeps_other_page_detection_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            with config_path.open("w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "ocr": {
                            "page_detection": {
                                "canny_low": 70,
                            }
                        }
                    },
                    stream,
                )

            with mock.patch.object(config, "CONFIG_PATH", config_path):
                settings = config.load_config()["ocr"]["page_detection"]

        self.assertEqual(settings["canny_low"], 70)
        self.assertEqual(settings["canny_high"], 150)
        self.assertEqual(settings["max_width"], 800)
        self.assertTrue(settings["enabled"])

    def test_example_ocr_settings_match_defaults(self):
        example_path = config.PROJECT_ROOT / "config" / "config.example.json"
        with example_path.open("r", encoding="utf-8") as stream:
            example = json.load(stream)

        self.assertEqual(example["ocr"], config.DEFAULTS["ocr"])


if __name__ == "__main__":
    unittest.main()
