"""Run one TensorRT scene detection and speak the result."""

import sys

from camera import capture_frame, resize_to_width
from config import PROJECT_ROOT, load_config, project_path
from say import say
from scene import describe


def describe_once(speak=True):
    config = load_config()
    settings = config["yolo"]
    vendor = PROJECT_ROOT / "vendor" / "JetsonYolov5"
    sys.path.insert(0, str(vendor))
    from yoloDet import YoloTRT

    plugin = project_path(settings["plugin"])
    engine = project_path(settings["engine"])
    for required in (plugin, engine):
        if not required.exists():
            raise RuntimeError("Missing YOLO artifact: {0}".format(required))
    model = YoloTRT(
        library=str(plugin), engine=str(engine),
        conf=float(settings["confidence"]), yolo_ver="v5"
    )
    frame = capture_frame(config["camera"])
    frame = resize_to_width(frame, int(settings["input_width"]))
    detections, _inference_time = model.Inference(frame)
    text = describe(detections, float(settings["confidence"]), int(settings["max_objects"]))
    if speak:
        say(text)
    return text


if __name__ == "__main__":
    describe_once()
