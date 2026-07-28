"""Capture, preprocess and read a Polish printed page."""

import math
import subprocess

import cv2

from camera import capture_frame
from config import ensure_runtime_dir, load_config
from say import say


MIN_WARP_SIDE_PIXELS = 32


def resize_for_page_detection(gray, settings):
    """Return a detection-sized image and scales back to the source image."""
    max_width = int(settings["max_width"])
    if max_width <= 0:
        raise ValueError("page_detection.max_width must be greater than zero")

    height, width = gray.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Page detection image must have positive dimensions")
    if width <= max_width:
        return gray, 1.0, 1.0

    target_height = max(1, int(round(height * (float(max_width) / width))))
    resized = cv2.resize(
        gray,
        (max_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    scale_x = width / float(max_width)
    scale_y = height / float(target_height)
    return resized, scale_x, scale_y


def detect_page_edges(gray, settings):
    """Build a Canny edge map on a detection-sized grayscale image."""
    blur_kernel = int(settings["blur_kernel"])
    if blur_kernel <= 0 or blur_kernel % 2 == 0:
        raise ValueError("page_detection.blur_kernel must be positive and odd")

    canny_low = int(settings["canny_low"])
    canny_high = int(settings["canny_high"])
    if canny_low < 0:
        raise ValueError("page_detection.canny_low must not be negative")
    if canny_high <= canny_low:
        raise ValueError(
            "page_detection.canny_high must be greater than canny_low"
        )

    detection_gray, scale_x, scale_y = resize_for_page_detection(gray, settings)
    blurred = cv2.GaussianBlur(
        detection_gray,
        (blur_kernel, blur_kernel),
        0,
    )
    edges = cv2.Canny(blurred, canny_low, canny_high)
    return edges, scale_x, scale_y


def find_page_contours(edges, settings):
    """Return the largest contours from OpenCV 3 or 4 findContours output."""
    candidate_limit = int(settings["contour_candidates"])
    if candidate_limit <= 0:
        raise ValueError(
            "page_detection.contour_candidates must be greater than zero"
        )

    result = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if len(result) == 2:
        contours = result[0]
    elif len(result) == 3:
        contours = result[1]
    else:
        raise RuntimeError("Unexpected cv2.findContours result")

    return sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )[:candidate_limit]


def _quadrilateral_points(approximation):
    points = []
    for point in approximation:
        coordinates = point[0] if len(point) == 1 else point
        if len(coordinates) != 2:
            return None
        points.append((float(coordinates[0]), float(coordinates[1])))
    return points


def select_page_quadrilateral(contours, image_shape, settings):
    """Return the first geometrically valid page candidate, or None."""
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Page detection image must have positive dimensions")

    epsilon_ratio = float(settings["approx_epsilon_ratio"])
    if epsilon_ratio <= 0 or epsilon_ratio >= 1:
        raise ValueError(
            "page_detection.approx_epsilon_ratio must be between zero and one"
        )

    min_area_ratio = float(settings["min_page_area_ratio"])
    if min_area_ratio <= 0 or min_area_ratio >= 1:
        raise ValueError(
            "page_detection.min_page_area_ratio must be between zero and one"
        )

    image_area = float(height * width)
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area / image_area < min_area_ratio:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue

        approximation = cv2.approxPolyDP(
            contour,
            epsilon_ratio * perimeter,
            True,
        )
        if len(approximation) != 4:
            continue
        if not cv2.isContourConvex(approximation):
            continue

        points = _quadrilateral_points(approximation)
        if points is None or len(set(points)) != 4:
            continue

        has_short_side = False
        for index, point in enumerate(points):
            next_point = points[(index + 1) % 4]
            delta_x = next_point[0] - point[0]
            delta_y = next_point[1] - point[1]
            if delta_x * delta_x + delta_y * delta_y < 4.0:
                has_short_side = True
                break
        if has_short_side:
            continue

        return approximation

    return None


def order_page_corners(points):
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    corners = _quadrilateral_points(points)
    if corners is None or len(corners) != 4 or len(set(corners)) != 4:
        raise ValueError("Page corners must contain four distinct points")

    center_x = sum(point[0] for point in corners) / 4.0
    center_y = sum(point[1] for point in corners) / 4.0
    ordered = sorted(
        corners,
        key=lambda point: math.atan2(
            point[1] - center_y,
            point[0] - center_x,
        ),
    )

    top_left_index = min(
        range(4),
        key=lambda index: (
            ordered[index][0] + ordered[index][1],
            ordered[index][1],
            ordered[index][0],
        ),
    )
    ordered = ordered[top_left_index:] + ordered[:top_left_index]

    signed_area = 0.0
    for index, point in enumerate(ordered):
        next_point = ordered[(index + 1) % 4]
        signed_area += (
            point[0] * next_point[1] - point[1] * next_point[0]
        )
    if signed_area <= 0:
        ordered = [ordered[0]] + list(reversed(ordered[1:]))

    return ordered


def scale_page_corners(corners, scale_x, scale_y, image_shape):
    """Map detection coordinates to the source image and clamp the result."""
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Source image must have positive dimensions")
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Page corner scales must be greater than zero")
    if len(corners) != 4:
        raise ValueError("Exactly four ordered page corners are required")

    max_x = float(width - 1)
    max_y = float(height - 1)
    scaled = []
    for point in corners:
        x = min(max(float(point[0]) * scale_x, 0.0), max_x)
        y = min(max(float(point[1]) * scale_y, 0.0), max_y)
        scaled.append((x, y))

    if len(set(scaled)) != 4:
        raise ValueError("Scaled page corners must remain distinct")
    return scaled


def map_page_candidate(page_candidate, scale_x, scale_y, image_shape):
    """Order and scale a candidate, returning None for invalid geometry."""
    if page_candidate is None:
        return None
    try:
        ordered_corners = order_page_corners(page_candidate)
        return scale_page_corners(
            ordered_corners,
            scale_x,
            scale_y,
            image_shape,
        )
    except ValueError:
        return None


def warp_page(gray, corners):
    """Rectify an ordered page quadrilateral, or return None if unusable."""
    if len(corners) != 4:
        return None
    try:
        source_points = [
            (float(point[0]), float(point[1])) for point in corners
        ]
    except (TypeError, ValueError, IndexError):
        return None
    if len(set(source_points)) != 4:
        return None
    if not all(
        math.isfinite(coordinate)
        for point in source_points
        for coordinate in point
    ):
        return None

    top_left, top_right, bottom_right, bottom_left = source_points
    top_width = math.hypot(
        top_right[0] - top_left[0],
        top_right[1] - top_left[1],
    )
    bottom_width = math.hypot(
        bottom_right[0] - bottom_left[0],
        bottom_right[1] - bottom_left[1],
    )
    left_height = math.hypot(
        bottom_left[0] - top_left[0],
        bottom_left[1] - top_left[1],
    )
    right_height = math.hypot(
        bottom_right[0] - top_right[0],
        bottom_right[1] - top_right[1],
    )
    target_width = int(round(max(top_width, bottom_width)))
    target_height = int(round(max(left_height, right_height)))
    if (
        target_width < MIN_WARP_SIDE_PIXELS
        or target_height < MIN_WARP_SIDE_PIXELS
    ):
        return None

    try:
        import numpy
    except ImportError:
        raise RuntimeError("NumPy is required for page perspective correction")

    destination_points = [
        (0.0, 0.0),
        (float(target_width - 1), 0.0),
        (float(target_width - 1), float(target_height - 1)),
        (0.0, float(target_height - 1)),
    ]
    source = numpy.asarray(source_points, dtype=numpy.float32)
    destination = numpy.asarray(destination_points, dtype=numpy.float32)

    opencv_error = getattr(cv2, "error", None)
    try:
        transform = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(
            gray,
            transform,
            (target_width, target_height),
        )
    except Exception as error:
        if opencv_error is not None and isinstance(error, opencv_error):
            return None
        raise
    return warped if warped is not None else None


def select_ocr_source(gray, corners):
    """Choose a rectified page or safely fall back to the full gray frame."""
    if corners is None:
        return gray, None, True
    warped = warp_page(gray, corners)
    if warped is None:
        return gray, None, True
    return warped, warped, False


def threshold_for_ocr(gray_page, settings):
    block_size = int(settings["threshold_block_size"])
    if block_size <= 1 or block_size % 2 == 0:
        raise ValueError(
            "ocr.threshold_block_size must be an odd number greater than one"
        )
    threshold_c = float(settings["threshold_c"])
    if not math.isfinite(threshold_c):
        raise ValueError("ocr.threshold_c must be finite")

    return cv2.adaptiveThreshold(
        gray_page,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        threshold_c,
    )


def prepare_ocr_image(gray_page, settings):
    """Scale and binarize a grayscale page without detector-side blur."""
    scale_factor = float(settings["scale_factor"])
    if scale_factor <= 0 or not math.isfinite(scale_factor):
        raise ValueError("ocr.scale_factor must be finite and greater than zero")

    resized = cv2.resize(
        gray_page,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )
    return threshold_for_ocr(resized, settings)


def preprocess_for_ocr(frame, settings):
    """Prepare the full frame for OCR and report how it was processed."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detection_settings = settings["page_detection"]
    edges = None
    contours = []
    detection_scale = (1.0, 1.0)
    page_candidate = None
    corners = None
    warped = None
    if detection_settings["enabled"]:
        edges, scale_x, scale_y = detect_page_edges(
            gray,
            detection_settings,
        )
        contours = find_page_contours(edges, detection_settings)
        detection_scale = (scale_x, scale_y)
        page_candidate = select_page_quadrilateral(
            contours,
            edges.shape,
            detection_settings,
        )
        corners = map_page_candidate(
            page_candidate,
            scale_x,
            scale_y,
            gray.shape,
        )

    ocr_source, warped, used_fallback = select_ocr_source(gray, corners)

    prepared = prepare_ocr_image(ocr_source, settings)
    metadata = {
        "page_detected": corners is not None,
        "used_fallback": used_fallback,
        "corners": corners,
        "edges": edges,
        "contours": contours,
        "detection_scale": detection_scale,
        "warped": warped,
    }
    return prepared, metadata


def _remove_runtime_artifact(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_ocr_debug_images(runtime, frame, metadata, enabled):
    """Write fixed-name OCR diagnostics without accumulating SD-card files."""
    if not enabled:
        return

    edges_path = runtime / "page_edges.png"
    edges = metadata["edges"]
    if edges is None:
        _remove_runtime_artifact(edges_path)
    else:
        cv2.imwrite(str(edges_path), edges)

    overlay = frame.copy()
    corners = metadata["corners"]
    if corners is not None and len(corners) == 4:
        points = [
            (int(round(point[0])), int(round(point[1])))
            for point in corners
        ]
        for index, point in enumerate(points):
            cv2.line(
                overlay,
                point,
                points[(index + 1) % 4],
                (0, 255, 0),
                3,
            )
    cv2.imwrite(str(runtime / "page_detected.jpg"), overlay)

    warped_path = runtime / "page_warped.png"
    warped = metadata["warped"]
    if warped is None:
        _remove_runtime_artifact(warped_path)
    else:
        cv2.imwrite(str(warped_path), warped)


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
    prepared, metadata = preprocess_for_ocr(frame, config["ocr"])
    image_path = runtime / "page_prepared.png"
    cv2.imwrite(str(image_path), prepared)
    write_ocr_debug_images(
        runtime,
        frame,
        metadata,
        bool(config["ocr"]["page_detection"]["debug_images"]),
    )
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
