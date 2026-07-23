"""Lifecycle manager for the isolated TensorRT YOLO worker."""

import json
import os
import select
import subprocess
import sys
import tempfile
import threading

import cv2

from config import PROJECT_ROOT, ensure_runtime_dir


class YoloSession:
    """Reuse YOLO for consecutive descriptions and release it before OCR."""

    def __init__(self, startup_timeout=90.0, inference_timeout=30.0):
        self.startup_timeout = float(startup_timeout)
        self.inference_timeout = float(inference_timeout)
        self._process = None
        self._responses = None
        self._lock = threading.Lock()

    @property
    def active(self):
        return self._process is not None and self._process.poll() is None

    def _read_response(self, timeout):
        if self._responses is None:
            raise RuntimeError("YOLO response channel is not open.")
        readable, _, _ = select.select([self._responses], [], [], timeout)
        if not readable:
            raise RuntimeError(
                "YOLO worker timed out after {0:.1f} seconds.".format(timeout)
            )
        line = self._responses.readline()
        if not line:
            return_code = self._process.poll() if self._process is not None else None
            raise RuntimeError(
                "YOLO worker exited unexpectedly (exit code: {0}).".format(return_code)
            )
        response = json.loads(line)
        if response.get("status") == "error":
            raise RuntimeError("YOLO worker failed: {0}".format(response.get("error")))
        return response

    def _start_unlocked(self):
        if self.active:
            return
        self._stop_unlocked()

        read_fd, write_fd = os.pipe()
        worker = PROJECT_ROOT / "src" / "yolo_worker.py"
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(worker),
                    "--response-fd",
                    str(write_fd),
                ],
                stdin=subprocess.PIPE,
                universal_newlines=True,
                pass_fds=(write_fd,),
                cwd=str(PROJECT_ROOT),
            )
        except Exception:
            os.close(read_fd)
            os.close(write_fd)
            raise

        os.close(write_fd)
        self._process = process
        self._responses = os.fdopen(read_fd, "r", 1)
        try:
            response = self._read_response(self.startup_timeout)
        except Exception:
            self._stop_unlocked()
            raise
        if response.get("status") != "ready":
            self._stop_unlocked()
            raise RuntimeError("Unexpected YOLO startup response.")

    def start(self):
        with self._lock:
            self._start_unlocked()

    def infer(self, frame):
        with self._lock:
            self._start_unlocked()
            runtime = (
                "/dev/shm"
                if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)
                else str(ensure_runtime_dir())
            )
            descriptor, image_path = tempfile.mkstemp(
                prefix="nano-speaker-yolo-",
                suffix=".bmp",
                dir=runtime,
            )
            os.close(descriptor)
            try:
                if not cv2.imwrite(image_path, frame):
                    raise RuntimeError("Cannot write temporary YOLO input image.")
                request = json.dumps({"command": "infer", "image": image_path})
                self._process.stdin.write(request + "\n")
                self._process.stdin.flush()
                response = self._read_response(self.inference_timeout)
                if response.get("status") != "ok":
                    raise RuntimeError("Unexpected YOLO inference response.")
                return response["detections"], response["inference_time"]
            finally:
                try:
                    os.unlink(image_path)
                except OSError:
                    pass

    def _stop_unlocked(self):
        process = self._process
        responses = self._responses
        self._process = None
        self._responses = None
        if process is None:
            if responses is not None:
                responses.close()
            return

        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"command": "stop"}) + "\n")
                process.stdin.flush()
                try:
                    if responses is not None:
                        readable, _, _ = select.select([responses], [], [], 3.0)
                        if readable:
                            responses.readline()
                except (OSError, ValueError):
                    pass
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=5)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.kill()
                    process.wait()
        finally:
            if responses is not None:
                responses.close()

    def stop(self):
        with self._lock:
            self._stop_unlocked()

    close = stop

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.stop()
