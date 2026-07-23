"""Isolated TensorRT worker; process exit guarantees CUDA resource cleanup."""

import argparse
import json
import os
import sys
import traceback

import cv2

from config import PROJECT_ROOT, load_config, project_path


def _normalize_detections(detections):
    normalized = []
    for detection in detections:
        box = detection.get("box", [])
        if hasattr(box, "tolist"):
            box = box.tolist()
        normalized.append(
            {
                "class": str(detection.get("class", "")),
                "conf": float(detection.get("conf", 0.0)),
                "box": [float(value) for value in box],
            }
        )
    return normalized


def _emit(stream, payload):
    stream.write(json.dumps(payload) + "\n")
    stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-fd", type=int, required=True)
    args = parser.parse_args()
    response_stream = os.fdopen(args.response_fd, "w", 1)

    try:
        settings = load_config()["yolo"]
        vendor = PROJECT_ROOT / "vendor" / "JetsonYolov5"
        sys.path.insert(0, str(vendor))
        from yoloDet import YoloTRT

        plugin = project_path(settings["plugin"])
        engine = project_path(settings["engine"])
        for required in (plugin, engine):
            if not required.exists():
                raise RuntimeError("Missing YOLO artifact: {0}".format(required))

        model = YoloTRT(
            library=str(plugin),
            engine=str(engine),
            conf=float(settings["confidence"]),
            yolo_ver="v5",
        )
        _emit(response_stream, {"status": "ready"})
    except Exception as error:
        traceback.print_exc()
        _emit(response_stream, {"status": "error", "error": str(error)})
        response_stream.close()
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("command")
            if command == "stop":
                _emit(response_stream, {"status": "stopped"})
                break
            if command != "infer":
                raise ValueError("Unknown YOLO worker command: {0}".format(command))

            image = cv2.imread(request["image"], cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("Cannot read YOLO input image.")
            detections, inference_time = model.Inference(image)
            _emit(
                response_stream,
                {
                    "status": "ok",
                    "detections": _normalize_detections(detections),
                    "inference_time": float(inference_time),
                },
            )
        except Exception as error:
            traceback.print_exc()
            _emit(response_stream, {"status": "error", "error": str(error)})

    response_stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
