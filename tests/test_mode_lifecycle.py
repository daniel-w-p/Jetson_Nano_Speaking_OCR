import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")

import main_demo


class _FakeYoloSession:
    def __init__(self):
        self.stop_count = 0

    def stop(self):
        self.stop_count += 1


class ModeLifecycleTests(unittest.TestCase):
    def test_read_stops_yolo_before_ocr(self):
        camera = object()
        yolo = _FakeYoloSession()
        with mock.patch.object(main_demo, "read_once", return_value="tekst") as read:
            result = main_demo.run_action("read", camera, yolo)

        self.assertEqual(result, "tekst")
        self.assertEqual(yolo.stop_count, 1)
        read.assert_called_once_with(camera_session=camera)

    def test_describe_reuses_passed_yolo_session(self):
        camera = object()
        yolo = _FakeYoloSession()
        with mock.patch.object(
            main_demo,
            "describe_once",
            return_value="Widzę osobę.",
        ) as describe:
            result = main_demo.run_action("describe", camera, yolo)

        self.assertEqual(result, "Widzę osobę.")
        self.assertEqual(yolo.stop_count, 0)
        describe.assert_called_once_with(
            camera_session=camera,
            yolo_session=yolo,
        )


if __name__ == "__main__":
    unittest.main()
