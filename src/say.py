"""Offline Polish speech synthesis through Piper and ALSA."""

import os
import subprocess
import tempfile
import threading
from pathlib import Path

from config import PROJECT_ROOT, load_config, project_path


_LOCK = threading.Lock()


def say(text):
    text = " ".join(str(text).split()).strip()
    if not text:
        return

    settings = load_config()["speech"]
    piper = project_path(settings["binary"])
    model = PROJECT_ROOT / "models" / "piper" / settings["voice"]
    model_config = Path(str(model) + ".json")
    for required in (piper, model, model_config):
        if not required.exists():
            raise RuntimeError("Missing speech file: {0}".format(required))

    descriptor, wav_path = tempfile.mkstemp(prefix="nano-speaker-", suffix=".wav")
    os.close(descriptor)
    try:
        with _LOCK:
            subprocess.run(
                [str(piper), "--model", str(model), "--output_file", wav_path],
                input=text.encode("utf-8"),
                check=True,
            )
            command = ["aplay", "-q"]
            if settings.get("aplay_device"):
                command.extend(["-D", settings["aplay_device"]])
            command.append(wav_path)
            subprocess.run(command, check=True)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


if __name__ == "__main__":
    say("Test syntezy mowy. Widzę osobę.")
