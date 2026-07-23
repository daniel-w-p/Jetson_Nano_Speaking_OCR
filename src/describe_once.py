"""Run one isolated TensorRT scene detection and speak the result."""

from camera import capture_frame, resize_to_width
from config import load_config
from say import say
from scene import describe
from yolo_session import YoloSession


def describe_once(speak=True, camera_session=None, yolo_session=None):
    config = load_config()
    settings = config["yolo"]
    if camera_session is None:
        frame = capture_frame(config["camera"])
    else:
        frame = camera_session.capture_frame()
    frame = resize_to_width(frame, int(settings["input_width"]))
    if yolo_session is None:
        with YoloSession() as temporary_yolo:
            detections, _inference_time = temporary_yolo.infer(frame)
    else:
        detections, _inference_time = yolo_session.infer(frame)
    text = describe(detections, float(settings["confidence"]), int(settings["max_objects"]))
    if speak:
        say(text)
    return text


if __name__ == "__main__":
    describe_once()
