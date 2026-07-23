import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import say


class _FakePiperInput:
    def __init__(self, process):
        self.process = process
        self.pending = ""
        self.closed = False

    def write(self, value):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        self.pending += value
        return len(value)

    def flush(self):
        line, self.pending = self.pending.split("\n", 1)
        request = json.loads(line)
        self.process.requests.append(request)
        with open(request["output_file"], "wb") as stream:
            stream.write(b"R" * 100)
        self.process.responses.append(request["output_file"] + "\n")

    def close(self):
        self.closed = True


class _FakePiperOutput:
    def __init__(self, process):
        self.process = process
        self.closed = False

    def readline(self):
        return self.process.responses.pop(0).encode("utf-8")

    def close(self):
        self.closed = True


class _FakePiperProcess:
    def __init__(self):
        self.requests = []
        self.responses = []
        self.return_code = None
        self.stdin = _FakePiperInput(self)
        self.stdout = _FakePiperOutput(self)

    def poll(self):
        return self.return_code

    def wait(self, timeout=None):
        self.return_code = 0
        return 0

    def terminate(self):
        self.return_code = -15

    def kill(self):
        self.return_code = -9


class SpeechEngineTests(unittest.TestCase):
    def test_reuses_one_piper_process_for_multiple_messages(self):
        process = _FakePiperProcess()
        settings = {
            "binary": "bin/piper-jetson/piper",
            "voice": "pl_PL-gosia-medium.onnx",
            "aplay_device": "",
        }
        with mock.patch.object(Path, "exists", return_value=True):
            engine = say.SpeechEngine(settings)

        with mock.patch.object(say.subprocess, "Popen", return_value=process) as popen:
            with mock.patch.object(say.subprocess, "run") as run:
                engine.say("Pierwszy komunikat.")
                engine.say("Drugi komunikat.")
                engine.close()

        self.assertEqual(popen.call_count, 1)
        self.assertEqual(
            [request["text"] for request in process.requests],
            ["Pierwszy komunikat.", "Drugi komunikat."],
        )
        self.assertEqual(run.call_count, 2)
        self.assertTrue(process.stdin.closed)


if __name__ == "__main__":
    unittest.main()
