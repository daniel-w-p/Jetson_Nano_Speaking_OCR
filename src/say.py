"""Persistent offline Polish speech synthesis through Piper and ALSA."""

import atexit
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from config import PROJECT_ROOT, load_config, project_path


class SpeechEngine:
    """Keep one Piper voice process loaded for the lifetime of the session."""

    def __init__(self, settings=None):
        self.settings = dict(settings or load_config()["speech"])
        self._process = None
        self._lock = threading.Lock()

        self.piper = project_path(self.settings["binary"])
        self.model = PROJECT_ROOT / "models" / "piper" / self.settings["voice"]
        self.model_config = Path(str(self.model) + ".json")
        for required in (self.piper, self.model, self.model_config):
            if not required.exists():
                raise RuntimeError("Missing speech file: {0}".format(required))

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return
        self._stop()
        self._process = subprocess.Popen(
            [
                str(self.piper),
                "--model",
                str(self.model),
                "--json-input",
                "--quiet",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )

    def _stop(self):
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.kill()
                    process.wait()
        finally:
            if process.stdout is not None:
                process.stdout.close()

    def _synthesize(self, text, wav_path):
        request = json.dumps(
            {"text": text, "output_file": wav_path},
            ensure_ascii=False,
        )
        for attempt in range(2):
            self._start()
            try:
                self._process.stdin.write((request + "\n").encode("utf-8"))
                self._process.stdin.flush()
                response = self._process.stdout.readline().decode("utf-8").strip()
            except (BrokenPipeError, OSError):
                response = ""

            if (
                response == wav_path
                and os.path.exists(wav_path)
                and os.path.getsize(wav_path) > 44
            ):
                return

            return_code = self._process.poll()
            self._stop()
            if attempt == 1:
                raise RuntimeError(
                    "Persistent Piper failed (exit code: {0}).".format(return_code)
                )

    def say(self, text):
        text = " ".join(str(text).split()).strip()
        if not text:
            return

        descriptor, wav_path = tempfile.mkstemp(
            prefix="nano-speaker-",
            suffix=".wav",
        )
        os.close(descriptor)
        try:
            with self._lock:
                self._synthesize(text, wav_path)
                command = ["aplay", "-q"]
                if self.settings.get("aplay_device"):
                    command.extend(["-D", self.settings["aplay_device"]])
                command.append(wav_path)
                subprocess.run(command, check=True)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def close(self):
        with self._lock:
            self._stop()


_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def get_speech_engine():
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = SpeechEngine()
        return _ENGINE


def say(text):
    get_speech_engine().say(text)


def shutdown_speech():
    global _ENGINE
    with _ENGINE_LOCK:
        engine = _ENGINE
        _ENGINE = None
    if engine is not None:
        engine.close()


atexit.register(shutdown_speech)


if __name__ == "__main__":
    say("Test syntezy mowy. Widzę osobę.")
