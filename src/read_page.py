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


def close_page_edge_gaps(edges, settings):
    """Close short gaps in Canny edges before contour detection."""
    kernel_size = int(settings["edge_close_kernel"])
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "page_detection.edge_close_kernel must be positive and odd"
        )

    iterations = int(settings["edge_close_iterations"])
    if iterations <= 0:
        raise ValueError(
            "page_detection.edge_close_iterations must be greater than zero"
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )
    return cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations,
    )


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
    raw_edges = cv2.Canny(blurred, canny_low, canny_high)
    edges = close_page_edge_gaps(raw_edges, settings)
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


def _valid_page_quadrilaterals(contours, image_shape, settings):
    """Yield valid quadrilaterals with their contour areas."""
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

        yield approximation, contour_area


def select_page_quadrilateral(contours, image_shape, settings):
    """Return the first geometrically valid page candidate, or None."""
    for approximation, _contour_area in _valid_page_quadrilaterals(
        contours,
        image_shape,
        settings,
    ):
        return approximation

    return None


def _is_frame_outline(points, image_shape, tolerance):
    """Return whether a quadrilateral is just the artificial image frame."""
    height, width = image_shape[:2]
    try:
        ordered = order_page_corners(points)
    except ValueError:
        return False

    frame_corners = [
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (float(width - 1), float(height - 1)),
        (0.0, float(height - 1)),
    ]
    return all(
        abs(point[0] - frame[0]) <= tolerance
        and abs(point[1] - frame[1]) <= tolerance
        for point, frame in zip(ordered, frame_corners)
    )


def _polygon_contains_point(points, point):
    """Return whether a point lies inside or on a quadrilateral."""
    polygon = _quadrilateral_points(points)
    if polygon is None or len(polygon) != 4:
        return False

    point_x, point_y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        current_x, current_y = current
        previous_x, previous_y = previous

        cross_product = (
            (point_x - previous_x) * (current_y - previous_y)
            - (point_y - previous_y) * (current_x - previous_x)
        )
        if abs(cross_product) <= 1e-6:
            min_x = min(previous_x, current_x)
            max_x = max(previous_x, current_x)
            min_y = min(previous_y, current_y)
            max_y = max(previous_y, current_y)
            if min_x <= point_x <= max_x and min_y <= point_y <= max_y:
                return True

        crosses_ray = (current_y > point_y) != (previous_y > point_y)
        if crosses_ray:
            intersection_x = (
                (previous_x - current_x)
                * (point_y - current_y)
                / (previous_y - current_y)
                + current_x
            )
            if point_x < intersection_x:
                inside = not inside
        previous = current
    return inside


def select_clipped_page_quadrilateral(edges, settings):
    """Infer clipped page corners by closing visible edges on the frame."""
    if not settings["clipped_page_fallback"]:
        return None

    border_thickness = int(settings["frame_border_thickness"])
    if border_thickness <= 0:
        raise ValueError(
            "page_detection.frame_border_thickness must be greater than zero"
        )

    height, width = edges.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Page detection image must have positive dimensions")

    bordered_edges = edges.copy()
    cv2.rectangle(
        bordered_edges,
        (0, 0),
        (width - 1, height - 1),
        255,
        border_thickness,
    )
    contours = find_page_contours(bordered_edges, settings)
    frame_tolerance = float(border_thickness + 1)
    image_center = ((width - 1) / 2.0, (height - 1) / 2.0)
    candidates = []
    for approximation, contour_area in _valid_page_quadrilaterals(
        contours,
        bordered_edges.shape,
        settings,
    ):
        if _is_frame_outline(
            approximation,
            bordered_edges.shape,
            frame_tolerance,
        ):
            continue
        candidates.append(
            (
                _polygon_contains_point(approximation, image_center),
                contour_area,
                approximation,
            )
        )

    if not candidates:
        return None
    candidates.sort(
        key=lambda candidate: (candidate[0], candidate[1]),
        reverse=True,
    )
    return candidates[0][2]


def find_page_candidate(edges, contours, settings):
    """Return a normal or frame-inferred candidate and an inference flag."""
    candidate = select_page_quadrilateral(
        contours,
        edges.shape,
        settings,
    )
    if candidate is not None:
        return candidate, False

    candidate = select_clipped_page_quadrilateral(edges, settings)
    return candidate, candidate is not None


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


def estimate_text_skew(binary_page, settings):
    """Estimate the dominant text-line angle in degrees, or return None."""
    deskew_settings = settings["deskew"]
    kernel_width = int(deskew_settings["line_kernel_width"])
    kernel_height = int(deskew_settings["line_kernel_height"])
    if kernel_width <= 0 or kernel_height <= 0:
        raise ValueError("ocr.deskew line kernel dimensions must be positive")

    hough_threshold = int(deskew_settings["hough_threshold"])
    if hough_threshold <= 0:
        raise ValueError("ocr.deskew.hough_threshold must be greater than zero")

    min_length_ratio = float(deskew_settings["min_line_length_ratio"])
    max_gap_ratio = float(deskew_settings["max_line_gap_ratio"])
    if min_length_ratio <= 0 or min_length_ratio > 1:
        raise ValueError(
            "ocr.deskew.min_line_length_ratio must be between zero and one"
        )
    if max_gap_ratio < 0 or max_gap_ratio > 1:
        raise ValueError(
            "ocr.deskew.max_line_gap_ratio must be between zero and one"
        )

    min_lines = int(deskew_settings["min_lines"])
    if min_lines <= 0:
        raise ValueError("ocr.deskew.min_lines must be greater than zero")

    max_angle = float(deskew_settings["max_angle_degrees"])
    if max_angle <= 0 or max_angle >= 45 or not math.isfinite(max_angle):
        raise ValueError(
            "ocr.deskew.max_angle_degrees must be between zero and 45"
        )

    height, width = binary_page.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("OCR deskew image must have positive dimensions")

    text_mask = cv2.bitwise_not(binary_page)
    line_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, kernel_height),
    )
    line_mask = cv2.morphologyEx(
        text_mask,
        cv2.MORPH_CLOSE,
        line_kernel,
    )
    min_line_length = max(1, int(round(width * min_length_ratio)))
    max_line_gap = max(0, int(round(width * max_gap_ratio)))
    lines = cv2.HoughLinesP(
        line_mask,
        1,
        math.pi / 180.0,
        hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return None

    candidates = []
    for line in lines:
        coordinates = line[0] if len(line) == 1 else line
        if len(coordinates) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in coordinates]
        delta_x = x2 - x1
        delta_y = y2 - y1
        length = math.hypot(delta_x, delta_y)
        if length < min_line_length:
            continue

        angle = math.degrees(math.atan2(delta_y, delta_x))
        while angle > 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        if abs(angle) <= max_angle:
            candidates.append((angle, length))

    if len(candidates) < min_lines:
        return None

    candidates.sort(key=lambda candidate: candidate[0])
    half_weight = sum(candidate[1] for candidate in candidates) / 2.0
    accumulated_weight = 0.0
    for angle, length in candidates:
        accumulated_weight += length
        if accumulated_weight >= half_weight:
            return angle
    return candidates[-1][0]


def rotate_gray_image(gray_page, angle):
    """Rotate a grayscale page without clipping its expanded bounds."""
    if not math.isfinite(angle):
        raise ValueError("OCR deskew angle must be finite")
    height, width = gray_page.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("OCR deskew image must have positive dimensions")

    center = (width / 2.0, height / 2.0)
    transform = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(float(transform[0][0]))
    sine = abs(float(transform[0][1]))
    target_width = max(1, int(math.ceil(width * cosine + height * sine)))
    target_height = max(1, int(math.ceil(height * cosine + width * sine)))
    transform[0][2] += target_width / 2.0 - center[0]
    transform[1][2] += target_height / 2.0 - center[1]

    return cv2.warpAffine(
        gray_page,
        transform,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def deskew_ocr_source(gray_page, settings):
    """Deskew text lines while retaining the original image on uncertainty."""
    deskew_settings = settings["deskew"]
    if not deskew_settings["enabled"]:
        return gray_page, None, False

    min_angle = float(deskew_settings["min_angle_degrees"])
    max_angle = float(deskew_settings["max_angle_degrees"])
    if (
        min_angle < 0
        or min_angle >= max_angle
        or not math.isfinite(min_angle)
    ):
        raise ValueError(
            "ocr.deskew.min_angle_degrees must be non-negative and "
            "less than max_angle_degrees"
        )

    preliminary_binary = threshold_for_ocr(gray_page, settings)
    angle = estimate_text_skew(preliminary_binary, settings)
    if angle is None or abs(angle) < min_angle:
        return gray_page, angle, False

    opencv_error = getattr(cv2, "error", None)
    try:
        rotated = rotate_gray_image(gray_page, angle)
    except Exception as error:
        if opencv_error is not None and isinstance(error, opencv_error):
            return gray_page, angle, False
        raise
    if rotated is None:
        return gray_page, angle, False
    return rotated, angle, True


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
    corners_inferred = False
    corners = None
    warped = None
    if detection_settings["enabled"]:
        edges, scale_x, scale_y = detect_page_edges(
            gray,
            detection_settings,
        )
        contours = find_page_contours(edges, detection_settings)
        detection_scale = (scale_x, scale_y)
        page_candidate, corners_inferred = find_page_candidate(
            edges,
            contours,
            detection_settings,
        )
        corners = map_page_candidate(
            page_candidate,
            scale_x,
            scale_y,
            gray.shape,
        )

    ocr_source, warped, used_fallback = select_ocr_source(gray, corners)
    deskewed_source, deskew_angle, deskew_applied = deskew_ocr_source(
        ocr_source,
        settings,
    )

    prepared = prepare_ocr_image(deskewed_source, settings)
    metadata = {
        "page_detected": corners is not None,
        "corners_inferred": corners_inferred and corners is not None,
        "used_fallback": used_fallback,
        "corners": corners,
        "edges": edges,
        "contours": contours,
        "detection_scale": detection_scale,
        "warped": warped,
        "deskew_angle": deskew_angle,
        "deskew_applied": deskew_applied,
        "deskewed": deskewed_source if deskew_applied else None,
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
        outline_color = (
            (0, 165, 255)
            if metadata.get("corners_inferred", False)
            else (0, 255, 0)
        )
        points = [
            (int(round(point[0])), int(round(point[1])))
            for point in corners
        ]
        for index, point in enumerate(points):
            cv2.line(
                overlay,
                point,
                points[(index + 1) % 4],
                outline_color,
                3,
            )
    cv2.imwrite(str(runtime / "page_detected.jpg"), overlay)

    warped_path = runtime / "page_warped.png"
    warped = metadata["warped"]
    if warped is None:
        _remove_runtime_artifact(warped_path)
    else:
        cv2.imwrite(str(warped_path), warped)

    deskewed_path = runtime / "page_deskewed.png"
    deskewed = metadata.get("deskewed")
    if deskewed is None:
        _remove_runtime_artifact(deskewed_path)
    else:
        cv2.imwrite(str(deskewed_path), deskewed)


def any_ocr_debug_flag_enabled(settings):
    """Return whether any boolean debug_* option is enabled recursively."""
    for key, value in settings.items():
        if isinstance(value, dict):
            if any_ocr_debug_flag_enabled(value):
                return True
        elif key.startswith("debug_") and value is True:
            return True
    return False


def write_ocr_debug_status(metadata, settings):
    """Print which page-detection fallback produced the OCR input."""
    if not any_ocr_debug_flag_enabled(settings):
        return

    detection_settings = settings["page_detection"]
    if metadata["used_fallback"]:
        if not detection_settings["enabled"]:
            status = "OCR całej klatki; detekcja kartki jest wyłączona."
        elif metadata.get("corners_inferred", False):
            status = (
                "OCR całej klatki; transformacja narożników domkniętych "
                "granicą kadru nie powiodła się."
            )
        elif metadata["page_detected"]:
            status = (
                "OCR całej klatki; transformacja pełnego czworokąta "
                "nie powiodła się."
            )
        else:
            status = (
                "OCR całej klatki; nie znaleziono wiarygodnego "
                "czworokąta kartki."
            )
    elif metadata.get("corners_inferred", False):
        status = "narożniki domknięte granicą kadru."
    else:
        status = "brak; wykryto pełny czworokąt kartki."

    print("[OCR debug] Fallback: {0}".format(status), flush=True)

    deskew_settings = settings.get("deskew")
    if deskew_settings is None:
        return
    angle = metadata.get("deskew_angle")
    if not deskew_settings["enabled"]:
        deskew_status = "wyłączony."
    elif angle is None:
        deskew_status = "pominięty; za mało wiarygodnych linii tekstu."
    elif metadata.get("deskew_applied", False):
        deskew_status = "zastosowano obrót {0:.2f}°.".format(angle)
    elif abs(angle) < float(deskew_settings["min_angle_degrees"]):
        deskew_status = (
            "pominięty; zmierzony kąt {0:.2f}° jest poniżej progu."
        ).format(angle)
    else:
        deskew_status = (
            "pominięty; obrót o {0:.2f}° nie powiódł się."
        ).format(angle)
    print("[OCR debug] Deskew: {0}".format(deskew_status), flush=True)


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
    write_ocr_debug_status(metadata, config["ocr"])
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
