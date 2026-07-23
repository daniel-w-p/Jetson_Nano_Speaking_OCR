"""Camera capture helpers with a reusable latest-frame session."""

import threading
import time

import cv2


class CameraSession:
    """Keep a V4L2 camera streaming and expose the newest complete frame."""

    def __init__(self, settings):
        self.settings = dict(settings)
        self._capture = None
        self._thread = None
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest_frame = None
        self._sequence = 0
        self._failure = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self

        device = int(self.settings["device"])
        backend = getattr(cv2, "CAP_V4L2", 200)
        # JetPack 4.6.1's Python binding only accepts one constructor argument.
        # OpenCV's legacy domain-offset form selects the same explicit backend.
        capture = cv2.VideoCapture(device + backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Cannot open camera / Nie można otworzyć kamery.")

        pixel_format = str(self.settings.get("pixel_format", "MJPG"))
        if len(pixel_format) != 4:
            capture.release()
            raise RuntimeError("Camera pixel_format must contain exactly four characters.")
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*pixel_format),
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.settings["capture_width"]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.settings["capture_height"]))
        capture.set(cv2.CAP_PROP_FPS, float(self.settings.get("fps", 15)))

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected_width = int(self.settings["capture_width"])
        expected_height = int(self.settings["capture_height"])
        if (actual_width, actual_height) != (expected_width, expected_height):
            capture.release()
            raise RuntimeError(
                "Camera rejected {0}x{1}; active mode is {2}x{3}.".format(
                    expected_width, expected_height, actual_width, actual_height
                )
            )

        self._capture = capture
        self._stop.clear()
        self._failure = None
        self._latest_frame = None
        self._sequence = 0
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="nano-speaker-camera",
        )
        self._thread.daemon = True
        self._thread.start()

        warmup_frames = max(1, int(self.settings.get("warmup_frames", 30)))
        timeout = float(self.settings.get("frame_timeout_seconds", 6.0))
        deadline = time.monotonic() + timeout
        startup_error = None
        with self._condition:
            while self._sequence < warmup_frames:
                if self._failure is not None:
                    startup_error = self._failure
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    startup_error = (
                        "Camera warm-up timed out after {0:.1f} seconds.".format(timeout)
                    )
                    break
                self._condition.wait(remaining)
        if startup_error is not None:
            self.close()
            raise RuntimeError(startup_error)
        return self

    def _reader_loop(self):
        consecutive_failures = 0
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    with self._condition:
                        self._failure = (
                            "Camera stopped delivering frames / "
                            "Kamera przestała dostarczać obraz."
                        )
                        self._condition.notify_all()
                    return
                time.sleep(0.02)
                continue

            consecutive_failures = 0
            with self._condition:
                self._latest_frame = frame
                self._sequence += 1
                self._condition.notify_all()

    def capture_frame(self):
        if self._thread is None or not self._thread.is_alive():
            if self._failure is not None:
                raise RuntimeError(self._failure)
            raise RuntimeError("Camera session is not running.")

        timeout = float(self.settings.get("frame_timeout_seconds", 6.0))
        deadline = time.monotonic() + timeout
        with self._condition:
            previous_sequence = self._sequence
            while self._sequence <= previous_sequence:
                if self._failure is not None:
                    raise RuntimeError(self._failure)
                if self._stop.is_set():
                    raise RuntimeError("Camera session was closed.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "No fresh camera frame after {0:.1f} seconds.".format(timeout)
                    )
                self._condition.wait(remaining)
            return self._latest_frame.copy()

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        capture = self._capture
        if capture is not None:
            capture.release()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        self._capture = None

    def __enter__(self):
        return self.start()

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()


def capture_frame(settings):
    """Compatibility helper for one-shot scripts."""
    with CameraSession(settings) as session:
        return session.capture_frame()


def resize_to_width(frame, width):
    height = int(frame.shape[0] * (float(width) / frame.shape[1]))
    return cv2.resize(frame, (int(width), height), interpolation=cv2.INTER_AREA)
