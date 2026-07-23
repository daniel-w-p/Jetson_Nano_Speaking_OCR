"""Capture, preprocess and read a Polish printed page."""

import subprocess

import cv2

from camera import capture_frame
from config import ensure_runtime_dir, load_config
from say import say


def preprocess_for_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )


def ocr_image(path, settings):
    result = subprocess.run(
        [
            "tesseract",
            str(path),
            "stdout",
            "-l",
            settings["language"],
            "--psm",
            str(settings["page_segmentation_mode"]),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Tesseract failed: {0}".format(result.stderr.strip()))
    return result.stdout.strip()


def read_once(speak=True, camera_session=None):
    config = load_config()
    runtime = ensure_runtime_dir()
    if speak:
        say("Robię zdjęcie kartki.")
    if camera_session is None:
        frame = capture_frame(config["camera"])
    else:
        frame = camera_session.capture_frame()
    cv2.imwrite(str(runtime / "page_raw.jpg"), frame)
    prepared = preprocess_for_ocr(frame)
    image_path = runtime / "page_prepared.png"
    cv2.imwrite(str(image_path), prepared)
    if speak:
        say("Rozpoznaję tekst.")
    text = ocr_image(image_path, config["ocr"])
    if len(text) < 3:
        if speak:
            say("Nie udało mi się rozpoznać tekstu. Przybliż kartkę i popraw światło.")
        return ""
    limit = int(config["ocr"]["max_characters"])
    spoken_text = text[:limit] + (". Dalszy tekst pomijam." if len(text) > limit else "")
    if speak:
        say("Czytam. " + spoken_text)
    return text


if __name__ == "__main__":
    read_once()
