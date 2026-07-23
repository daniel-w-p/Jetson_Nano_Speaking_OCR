import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")

import camera


class _FakeFrame:
    def __init__(self, number):
        self.number = number

    def copy(self):
        return _FakeFrame(self.number)


class _FakeCapture:
    def __init__(self):
        self.frame_number = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, _property, _value):
        return True

    def get(self, property_id):
        if property_id == 3:
            return 1280
        if property_id == 4:
            return 720
        return 0

    def read(self):
        time.sleep(0.001)
        self.frame_number += 1
        return True, _FakeFrame(self.frame_number)

    def release(self):
        self.released = True


class CameraSessionTests(unittest.TestCase):
    def test_reuses_stream_and_returns_a_fresh_frame(self):
        capture = _FakeCapture()
        settings = {
            "device": 0,
            "capture_width": 1280,
            "capture_height": 720,
            "pixel_format": "MJPG",
            "fps": 15,
            "warmup_frames": 2,
            "frame_timeout_seconds": 1.0,
        }
        cv2_values = {
            "CAP_V4L2": 200,
            "CAP_PROP_FOURCC": 6,
            "CAP_PROP_FRAME_WIDTH": 3,
            "CAP_PROP_FRAME_HEIGHT": 4,
            "CAP_PROP_FPS": 5,
        }
        patches = [
            mock.patch.object(camera.cv2, name, value, create=True)
            for name, value in cv2_values.items()
        ]
        video_capture_mock = mock.Mock(return_value=capture)
        video_capture = mock.patch.object(
            camera.cv2,
            "VideoCapture",
            new=video_capture_mock,
            create=True,
        )
        patches.extend(
            [
                video_capture,
                mock.patch.object(
                    camera.cv2,
                    "VideoWriter_fourcc",
                    return_value=1234,
                    create=True,
                ),
            ]
        )

        for patch in patches:
            patch.start()
        try:
            session = camera.CameraSession(settings).start()
            warm_sequence = session._sequence
            frame = session.capture_frame()
            session.close()
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertGreater(frame.number, warm_sequence)
        self.assertTrue(capture.released)
        video_capture_mock.assert_called_once_with(200)


if __name__ == "__main__":
    unittest.main()
