"""Configuration shared by all assistant modes (Python 3.6 compatible)."""

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = Path(os.environ.get("NANO_SPEAKER_RUNTIME", str(PROJECT_ROOT / "tmp")))
CONFIG_PATH = Path(
    os.environ.get("NANO_SPEAKER_CONFIG", str(PROJECT_ROOT / "config" / "config.json"))
)


DEFAULTS = {
    "camera": {
        "device": 0,
        "capture_width": 1280,
        "capture_height": 720,
        "warmup_frames": 10,
    },
    "ocr": {"language": "pol", "page_segmentation_mode": 6, "max_characters": 700},
    "speech": {
        "binary": "bin/piper-jetson/piper",
        "voice": "pl_PL-gosia-medium.onnx",
        "aplay_device": "",
    },
    "yolo": {
        "confidence": 0.55,
        "input_width": 640,
        "max_objects": 3,
        "engine": "vendor/JetsonYolov5/yolov5/build/yolov5n.engine",
        "plugin": "vendor/JetsonYolov5/yolov5/build/libmyplugins.so",
    },
    "gpio": {
        "enabled": False,
        "describe_pin": 11,
        "read_pin": 13,
        "board_numbering": True,
        "bounce_time_ms": 500,
    },
}


def _merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config():
    if not CONFIG_PATH.exists():
        return DEFAULTS
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return _merge(DEFAULTS, json.load(stream))


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_runtime_dir():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return RUNTIME_DIR
