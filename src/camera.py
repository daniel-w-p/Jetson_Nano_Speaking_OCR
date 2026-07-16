"""Camera capture helpers."""

import cv2


def capture_frame(settings):
    cap = cv2.VideoCapture(int(settings["device"]))
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera / Nie można otworzyć kamery.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(settings["capture_width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(settings["capture_height"]))
    frame = None
    try:
        for _ in range(max(1, int(settings["warmup_frames"]))):
            ok, candidate = cap.read()
            if ok:
                frame = candidate
    finally:
        cap.release()

    if frame is None:
        raise RuntimeError("No frame received / Nie udało się pobrać obrazu.")
    return frame


def resize_to_width(frame, width):
    height = int(frame.shape[0] * (float(width) / frame.shape[1]))
    return cv2.resize(frame, (int(width), height), interpolation=cv2.INTER_AREA)
