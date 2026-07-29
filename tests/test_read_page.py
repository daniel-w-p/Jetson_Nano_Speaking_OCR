import sys
import tempfile
import types
import unittest
from itertools import permutations
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")

import config
import read_page


class PreprocessTests(unittest.TestCase):
    def test_detection_resize_limits_width_and_returns_source_scales(self):
        gray = mock.Mock()
        gray.shape = (333, 1000)
        resized = object()
        resize = mock.Mock(return_value=resized)
        patches = [
            mock.patch.object(
                read_page.cv2,
                "INTER_AREA",
                5,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "resize",
                new=resize,
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual, scale_x, scale_y = read_page.resize_for_page_detection(
                gray,
                {"max_width": 800},
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertIs(actual, resized)
        resize.assert_called_once_with(gray, (800, 266), interpolation=5)
        self.assertEqual(scale_x, 1.25)
        self.assertAlmostEqual(scale_y, 333.0 / 266.0)

    def test_detection_resize_does_not_upscale_small_image(self):
        gray = mock.Mock()
        gray.shape = (480, 640)
        resize = mock.Mock()

        with mock.patch.object(
            read_page.cv2,
            "resize",
            new=resize,
            create=True,
        ):
            actual, scale_x, scale_y = read_page.resize_for_page_detection(
                gray,
                {"max_width": 800},
            )

        self.assertIs(actual, gray)
        self.assertEqual((scale_x, scale_y), (1.0, 1.0))
        resize.assert_not_called()

    def test_detection_resize_rejects_non_positive_limit(self):
        gray = mock.Mock()
        gray.shape = (480, 640)

        with self.assertRaisesRegex(ValueError, "max_width"):
            read_page.resize_for_page_detection(gray, {"max_width": 0})

    def test_detection_edges_blurs_resized_image_and_runs_canny(self):
        gray = object()
        detection_gray = object()
        blurred = object()
        edges = object()
        settings = config.DEFAULTS["ocr"]["page_detection"]
        resize_detection = mock.Mock(
            return_value=(detection_gray, 1.6, 1.6)
        )
        gaussian_blur = mock.Mock(return_value=blurred)
        canny = mock.Mock(return_value=edges)
        patches = [
            mock.patch.object(
                read_page,
                "resize_for_page_detection",
                new=resize_detection,
            ),
            mock.patch.object(
                read_page.cv2,
                "GaussianBlur",
                new=gaussian_blur,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "Canny",
                new=canny,
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual, scale_x, scale_y = read_page.detect_page_edges(
                gray,
                settings,
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertIs(actual, edges)
        self.assertEqual((scale_x, scale_y), (1.6, 1.6))
        resize_detection.assert_called_once_with(gray, settings)
        gaussian_blur.assert_called_once_with(detection_gray, (5, 5), 0)
        canny.assert_called_once_with(blurred, 50, 150)

    def test_detection_edges_reject_invalid_blur_and_canny_settings(self):
        defaults = config.DEFAULTS["ocr"]["page_detection"]
        invalid_settings = [
            ({"blur_kernel": 4}, "blur_kernel"),
            ({"canny_low": -1}, "canny_low"),
            ({"canny_low": 150, "canny_high": 150}, "canny_high"),
        ]

        for override, expected_error in invalid_settings:
            settings = dict(defaults)
            settings.update(override)
            with self.subTest(settings=override):
                with self.assertRaisesRegex(ValueError, expected_error):
                    read_page.detect_page_edges(object(), settings)

    def test_find_page_contours_supports_opencv_4_and_limits_by_area(self):
        edges = object()
        small = object()
        medium = object()
        large = object()
        areas = {small: 10, medium: 20, large: 30}
        find_contours = mock.Mock(
            return_value=([small, large, medium], object())
        )
        contour_area = mock.Mock(side_effect=lambda contour: areas[contour])
        patches = [
            mock.patch.object(
                read_page.cv2,
                "RETR_LIST",
                6,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "CHAIN_APPROX_SIMPLE",
                7,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "findContours",
                new=find_contours,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "contourArea",
                new=contour_area,
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual = read_page.find_page_contours(
                edges,
                {"contour_candidates": 2},
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertEqual(actual, [large, medium])
        find_contours.assert_called_once_with(edges, 6, 7)

    def test_find_page_contours_supports_opencv_3_result(self):
        edges = object()
        contour = object()
        find_contours = mock.Mock(
            return_value=(object(), [contour], object())
        )
        patches = [
            mock.patch.object(
                read_page.cv2,
                "RETR_LIST",
                6,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "CHAIN_APPROX_SIMPLE",
                7,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "findContours",
                new=find_contours,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "contourArea",
                return_value=10,
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual = read_page.find_page_contours(
                edges,
                {"contour_candidates": 10},
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertEqual(actual, [contour])

    def test_find_page_contours_rejects_invalid_limit_and_result(self):
        with self.assertRaisesRegex(ValueError, "contour_candidates"):
            read_page.find_page_contours(
                object(),
                {"contour_candidates": 0},
            )

        find_contours = mock.Mock(return_value=([],))
        patches = [
            mock.patch.object(
                read_page.cv2,
                "RETR_LIST",
                6,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "CHAIN_APPROX_SIMPLE",
                7,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "findContours",
                new=find_contours,
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "findContours"):
                read_page.find_page_contours(
                    object(),
                    {"contour_candidates": 10},
                )
        finally:
            for patch in reversed(patches):
                patch.stop()

    def test_select_page_quadrilateral_returns_first_valid_candidate(self):
        invalid_contour = object()
        valid_contour = object()
        later_contour = object()
        invalid_approximation = [
            [[0, 0]],
            [[80, 0]],
            [[90, 50]],
            [[40, 90]],
            [[0, 50]],
        ]
        valid_approximation = [
            [[10, 10]],
            [[90, 10]],
            [[90, 90]],
            [[10, 90]],
        ]
        later_approximation = [
            [[20, 20]],
            [[80, 20]],
            [[80, 80]],
            [[20, 80]],
        ]
        approximations = {
            invalid_contour: invalid_approximation,
            valid_contour: valid_approximation,
            later_contour: later_approximation,
        }
        contour_area = mock.Mock(return_value=5000)
        arc_length = mock.Mock(return_value=200)
        approx_poly = mock.Mock(
            side_effect=lambda contour, _epsilon, _closed: approximations[contour]
        )
        is_convex = mock.Mock(return_value=True)
        patches = [
            mock.patch.object(
                read_page.cv2,
                "contourArea",
                new=contour_area,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "arcLength",
                new=arc_length,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "approxPolyDP",
                new=approx_poly,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "isContourConvex",
                new=is_convex,
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual = read_page.select_page_quadrilateral(
                [invalid_contour, valid_contour, later_contour],
                (100, 100),
                {
                    "approx_epsilon_ratio": 0.02,
                    "min_page_area_ratio": 0.20,
                },
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertIs(actual, valid_approximation)
        approx_poly.assert_has_calls(
            [
                mock.call(invalid_contour, 4.0, True),
                mock.call(valid_contour, 4.0, True),
            ]
        )
        self.assertEqual(approx_poly.call_count, 2)

    def test_select_page_quadrilateral_rejects_invalid_geometry(self):
        too_small = object()
        non_convex = object()
        duplicated_corner = object()
        short_side = object()
        valid_shapes = {
            non_convex: [
                [[0, 0]],
                [[80, 0]],
                [[20, 20]],
                [[0, 80]],
            ],
            duplicated_corner: [
                [[0, 0]],
                [[80, 0]],
                [[80, 0]],
                [[0, 80]],
            ],
            short_side: [
                [[0, 0]],
                [[1, 0]],
                [[80, 80]],
                [[0, 80]],
            ],
        }
        areas = {
            too_small: 1000,
            non_convex: 5000,
            duplicated_corner: 5000,
            short_side: 5000,
        }
        contour_area = mock.Mock(side_effect=lambda contour: areas[contour])
        arc_length = mock.Mock(return_value=200)
        approx_poly = mock.Mock(
            side_effect=lambda contour, _epsilon, _closed: valid_shapes[contour]
        )
        is_convex = mock.Mock(
            side_effect=lambda approximation: approximation
            is not valid_shapes[non_convex]
        )
        patches = [
            mock.patch.object(
                read_page.cv2,
                "contourArea",
                new=contour_area,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "arcLength",
                new=arc_length,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "approxPolyDP",
                new=approx_poly,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "isContourConvex",
                new=is_convex,
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual = read_page.select_page_quadrilateral(
                [too_small, non_convex, duplicated_corner, short_side],
                (100, 100),
                {
                    "approx_epsilon_ratio": 0.02,
                    "min_page_area_ratio": 0.20,
                },
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertIsNone(actual)
        self.assertNotIn(
            mock.call(too_small, 4.0, True),
            approx_poly.mock_calls,
        )

    def test_select_page_quadrilateral_validates_settings_and_image(self):
        valid_settings = {
            "approx_epsilon_ratio": 0.02,
            "min_page_area_ratio": 0.20,
        }
        with self.assertRaisesRegex(ValueError, "positive dimensions"):
            read_page.select_page_quadrilateral(
                [],
                (0, 100),
                valid_settings,
            )

        for key in ("approx_epsilon_ratio", "min_page_area_ratio"):
            settings = dict(valid_settings)
            settings[key] = 0
            with self.subTest(setting=key):
                with self.assertRaisesRegex(ValueError, key):
                    read_page.select_page_quadrilateral(
                        [],
                        (100, 100),
                        settings,
                    )

    def test_clipped_page_fallback_adds_frame_and_prefers_centered_page(self):
        edges = mock.Mock()
        edges.shape = (100, 100)
        bordered_edges = mock.Mock()
        bordered_edges.shape = (100, 100)
        edges.copy.return_value = bordered_edges
        frame_contour = object()
        off_center_contour = object()
        centered_contour = object()
        frame = [
            [[0, 0]],
            [[99, 0]],
            [[99, 99]],
            [[0, 99]],
        ]
        off_center = [
            [[0, 10]],
            [[40, 10]],
            [[40, 90]],
            [[0, 90]],
        ]
        centered = [
            [[30, 20]],
            [[70, 20]],
            [[70, 80]],
            [[30, 80]],
        ]
        rectangle = mock.Mock()
        settings = dict(config.DEFAULTS["ocr"]["page_detection"])

        with mock.patch.object(
            read_page.cv2,
            "rectangle",
            new=rectangle,
            create=True,
        ):
            with mock.patch.object(
                read_page,
                "find_page_contours",
                return_value=[
                    frame_contour,
                    off_center_contour,
                    centered_contour,
                ],
            ) as find_contours:
                with mock.patch.object(
                    read_page,
                    "_valid_page_quadrilaterals",
                    return_value=[
                        (frame, 9801.0),
                        (off_center, 3200.0),
                        (centered, 2400.0),
                    ],
                ):
                    actual = read_page.select_clipped_page_quadrilateral(
                        edges,
                        settings,
                    )

        self.assertIs(actual, centered)
        edges.copy.assert_called_once_with()
        rectangle.assert_called_once_with(
            bordered_edges,
            (0, 0),
            (99, 99),
            255,
            3,
        )
        find_contours.assert_called_once_with(bordered_edges, settings)

    def test_clipped_page_fallback_can_be_disabled(self):
        edges = mock.Mock()
        settings = dict(config.DEFAULTS["ocr"]["page_detection"])
        settings["clipped_page_fallback"] = False

        actual = read_page.select_clipped_page_quadrilateral(
            edges,
            settings,
        )

        self.assertIsNone(actual)
        edges.copy.assert_not_called()

    def test_clipped_page_fallback_validates_border_thickness(self):
        settings = dict(config.DEFAULTS["ocr"]["page_detection"])
        settings["frame_border_thickness"] = 0

        with self.assertRaisesRegex(ValueError, "frame_border_thickness"):
            read_page.select_clipped_page_quadrilateral(
                mock.Mock(),
                settings,
            )

    def test_find_page_candidate_uses_clipped_fallback_only_when_needed(self):
        edges = mock.Mock()
        edges.shape = (100, 100)
        contours = [object()]
        settings = config.DEFAULTS["ocr"]["page_detection"]
        normal = object()
        inferred = object()

        with mock.patch.object(
            read_page,
            "select_page_quadrilateral",
            side_effect=[normal, None],
        ) as select_normal:
            with mock.patch.object(
                read_page,
                "select_clipped_page_quadrilateral",
                return_value=inferred,
            ) as select_clipped:
                normal_result = read_page.find_page_candidate(
                    edges,
                    contours,
                    settings,
                )
                inferred_result = read_page.find_page_candidate(
                    edges,
                    contours,
                    settings,
                )

        self.assertEqual(normal_result, (normal, False))
        self.assertEqual(inferred_result, (inferred, True))
        self.assertEqual(select_normal.call_count, 2)
        select_clipped.assert_called_once_with(edges, settings)

    def test_order_page_corners_is_stable_for_all_input_permutations(self):
        expected = [
            (20.0, 10.0),
            (90.0, 30.0),
            (80.0, 100.0),
            (10.0, 80.0),
        ]

        for permutation in permutations(expected):
            opencv_points = [[list(point)] for point in permutation]
            with self.subTest(points=permutation):
                self.assertEqual(
                    read_page.order_page_corners(opencv_points),
                    expected,
                )

    def test_order_page_corners_rejects_missing_or_duplicated_points(self):
        with self.assertRaisesRegex(ValueError, "four distinct"):
            read_page.order_page_corners(
                [
                    [[0, 0]],
                    [[10, 0]],
                    [[10, 0]],
                    [[0, 10]],
                ]
            )

        with self.assertRaisesRegex(ValueError, "four distinct"):
            read_page.order_page_corners(
                [
                    [[0, 0]],
                    [[10, 0]],
                    [[0, 10]],
                ]
            )

    def test_scale_page_corners_maps_and_clamps_to_source_image(self):
        actual = read_page.scale_page_corners(
            [
                (-1, -1),
                (100, 0),
                (100, 50),
                (0, 50),
            ],
            1.6,
            1.5,
            (75, 160),
        )

        self.assertEqual(
            actual,
            [
                (0.0, 0.0),
                (159.0, 0.0),
                (159.0, 74.0),
                (0.0, 74.0),
            ],
        )

    def test_scale_page_corners_rejects_invalid_scale_and_dimensions(self):
        corners = [(0, 0), (10, 0), (10, 10), (0, 10)]
        with self.assertRaisesRegex(ValueError, "positive dimensions"):
            read_page.scale_page_corners(corners, 1.0, 1.0, (0, 100))
        with self.assertRaisesRegex(ValueError, "scales"):
            read_page.scale_page_corners(corners, 0, 1.0, (100, 100))

    def test_map_page_candidate_falls_back_for_invalid_geometry(self):
        candidate = object()
        order_corners = mock.Mock(
            side_effect=ValueError("invalid corners")
        )
        scale_corners = mock.Mock()
        patches = [
            mock.patch.object(
                read_page,
                "order_page_corners",
                new=order_corners,
            ),
            mock.patch.object(
                read_page,
                "scale_page_corners",
                new=scale_corners,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual = read_page.map_page_candidate(
                candidate,
                1.6,
                1.6,
                (720, 1280),
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertIsNone(actual)
        order_corners.assert_called_once_with(candidate)
        scale_corners.assert_not_called()

    def test_warp_page_builds_transform_with_longest_opposite_sides(self):
        gray = object()
        source_array = object()
        destination_array = object()
        transform = object()
        warped = object()
        numpy_module = types.ModuleType("numpy")
        numpy_module.float32 = object()
        numpy_module.asarray = mock.Mock(
            side_effect=[source_array, destination_array]
        )
        get_transform = mock.Mock(return_value=transform)
        warp_perspective = mock.Mock(return_value=warped)
        patches = [
            mock.patch.dict(sys.modules, {"numpy": numpy_module}),
            mock.patch.object(
                read_page.cv2,
                "getPerspectiveTransform",
                new=get_transform,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "warpPerspective",
                new=warp_perspective,
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual = read_page.warp_page(
                gray,
                [
                    (10, 10),
                    (110, 20),
                    (100, 220),
                    (20, 210),
                ],
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertIs(actual, warped)
        numpy_module.asarray.assert_has_calls(
            [
                mock.call(
                    [
                        (10.0, 10.0),
                        (110.0, 20.0),
                        (100.0, 220.0),
                        (20.0, 210.0),
                    ],
                    dtype=numpy_module.float32,
                ),
                mock.call(
                    [
                        (0.0, 0.0),
                        (99.0, 0.0),
                        (99.0, 199.0),
                        (0.0, 199.0),
                    ],
                    dtype=numpy_module.float32,
                ),
            ]
        )
        get_transform.assert_called_once_with(source_array, destination_array)
        warp_perspective.assert_called_once_with(
            gray,
            transform,
            (100, 200),
        )

    def test_warp_page_rejects_tiny_or_invalid_corners(self):
        self.assertIsNone(
            read_page.warp_page(
                object(),
                [(0, 0), (10, 0), (10, 10), (0, 10)],
            )
        )
        self.assertIsNone(
            read_page.warp_page(
                object(),
                [(0, 0), (100, 0), (100, 0), (0, 100)],
            )
        )
        self.assertIsNone(
            read_page.warp_page(
                object(),
                [(0, 0), (100, 0), (100, 100)],
            )
        )

    def test_warp_page_uses_fallback_for_opencv_error(self):
        class FakeOpenCvError(Exception):
            pass

        numpy_module = types.ModuleType("numpy")
        numpy_module.float32 = object()
        numpy_module.asarray = mock.Mock(return_value=object())
        patches = [
            mock.patch.dict(sys.modules, {"numpy": numpy_module}),
            mock.patch.object(
                read_page.cv2,
                "error",
                FakeOpenCvError,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "getPerspectiveTransform",
                side_effect=FakeOpenCvError("invalid transform"),
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual = read_page.warp_page(
                object(),
                [(0, 0), (100, 0), (100, 100), (0, 100)],
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertIsNone(actual)

    def test_select_ocr_source_falls_back_without_valid_warp(self):
        gray = object()
        corners = object()
        warped = object()
        warp = mock.Mock(side_effect=[None, warped])

        with mock.patch.object(read_page, "warp_page", new=warp):
            no_corners = read_page.select_ocr_source(gray, None)
            failed_warp = read_page.select_ocr_source(gray, corners)
            successful_warp = read_page.select_ocr_source(gray, corners)

        self.assertEqual(no_corners, (gray, None, True))
        self.assertEqual(failed_warp, (gray, None, True))
        self.assertEqual(successful_warp, (warped, warped, False))
        self.assertEqual(warp.call_count, 2)

    def test_preprocess_skips_detection_and_uses_frame_when_disabled(self):
        frame = object()
        gray = object()
        prepared = object()
        settings = dict(config.DEFAULTS["ocr"])
        settings["page_detection"] = dict(settings["page_detection"])
        settings["page_detection"]["enabled"] = False
        cvt_color = mock.Mock(return_value=gray)
        detect_edges = mock.Mock()
        prepare_ocr = mock.Mock(return_value=prepared)
        warp = mock.Mock()
        patches = [
            mock.patch.object(
                read_page.cv2,
                "COLOR_BGR2GRAY",
                1,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "cvtColor",
                new=cvt_color,
                create=True,
            ),
            mock.patch.object(
                read_page,
                "detect_page_edges",
                new=detect_edges,
            ),
            mock.patch.object(
                read_page,
                "prepare_ocr_image",
                new=prepare_ocr,
            ),
            mock.patch.object(
                read_page,
                "warp_page",
                new=warp,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual, metadata = read_page.preprocess_for_ocr(frame, settings)
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertIs(actual, prepared)
        detect_edges.assert_not_called()
        warp.assert_not_called()
        prepare_ocr.assert_called_once_with(gray, settings)
        self.assertEqual(
            metadata,
            {
                "page_detected": False,
                "corners_inferred": False,
                "used_fallback": True,
                "corners": None,
                "edges": None,
                "contours": [],
                "detection_scale": (1.0, 1.0),
                "warped": None,
            },
        )

    def test_preprocess_connects_detection_warp_and_ocr_stages(self):
        frame = object()
        gray = mock.Mock()
        gray.shape = (720, 1280)
        edges = mock.Mock()
        edges.shape = (450, 800)
        contours = [object()]
        page_candidate = object()
        ordered_corners = object()
        source_corners = object()
        warped_page = object()
        prepared = object()
        settings = config.DEFAULTS["ocr"]
        cvt_color = mock.Mock(return_value=gray)
        detect_edges = mock.Mock(return_value=(edges, 1.6, 1.6))
        find_contours = mock.Mock(return_value=contours)
        find_candidate = mock.Mock(return_value=(page_candidate, True))
        order_corners = mock.Mock(return_value=ordered_corners)
        scale_corners = mock.Mock(return_value=source_corners)
        warp = mock.Mock(return_value=warped_page)
        prepare_ocr = mock.Mock(return_value=prepared)
        patches = [
            mock.patch.object(
                read_page.cv2,
                "COLOR_BGR2GRAY",
                1,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "cvtColor",
                new=cvt_color,
                create=True,
            ),
            mock.patch.object(
                read_page,
                "detect_page_edges",
                new=detect_edges,
            ),
            mock.patch.object(
                read_page,
                "find_page_contours",
                new=find_contours,
            ),
            mock.patch.object(
                read_page,
                "find_page_candidate",
                new=find_candidate,
            ),
            mock.patch.object(
                read_page,
                "order_page_corners",
                new=order_corners,
            ),
            mock.patch.object(
                read_page,
                "scale_page_corners",
                new=scale_corners,
            ),
            mock.patch.object(
                read_page,
                "warp_page",
                new=warp,
            ),
            mock.patch.object(
                read_page,
                "prepare_ocr_image",
                new=prepare_ocr,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual, metadata = read_page.preprocess_for_ocr(frame, settings)
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertIs(actual, prepared)
        cvt_color.assert_called_once_with(frame, 1)
        detect_edges.assert_called_once_with(
            gray,
            settings["page_detection"],
        )
        find_contours.assert_called_once_with(
            edges,
            settings["page_detection"],
        )
        find_candidate.assert_called_once_with(
            edges,
            contours,
            settings["page_detection"],
        )
        order_corners.assert_called_once_with(page_candidate)
        scale_corners.assert_called_once_with(
            ordered_corners,
            1.6,
            1.6,
            gray.shape,
        )
        warp.assert_called_once_with(gray, source_corners)
        prepare_ocr.assert_called_once_with(warped_page, settings)
        self.assertEqual(
            metadata,
            {
                "page_detected": True,
                "corners_inferred": True,
                "used_fallback": False,
                "corners": source_corners,
                "edges": edges,
                "contours": contours,
                "detection_scale": (1.6, 1.6),
                "warped": warped_page,
            },
        )

    def test_threshold_uses_ocr_settings(self):
        gray = object()
        prepared = object()
        settings = dict(config.DEFAULTS["ocr"])
        settings["threshold_block_size"] = 15
        settings["threshold_c"] = 3
        adaptive_threshold = mock.Mock(return_value=prepared)
        patches = [
            mock.patch.object(
                read_page.cv2,
                "ADAPTIVE_THRESH_GAUSSIAN_C",
                3,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "THRESH_BINARY",
                4,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "adaptiveThreshold",
                new=adaptive_threshold,
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual = read_page.threshold_for_ocr(gray, settings)
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertIs(actual, prepared)
        adaptive_threshold.assert_called_once_with(gray, 255, 3, 4, 15, 3.0)

    def test_threshold_rejects_invalid_block_size_and_constant(self):
        for block_size in (0, 1, 14):
            settings = dict(config.DEFAULTS["ocr"])
            settings["threshold_block_size"] = block_size
            with self.subTest(block_size=block_size):
                with self.assertRaisesRegex(ValueError, "threshold_block_size"):
                    read_page.threshold_for_ocr(object(), settings)

        settings = dict(config.DEFAULTS["ocr"])
        settings["threshold_c"] = float("inf")
        with self.assertRaisesRegex(ValueError, "threshold_c"):
            read_page.threshold_for_ocr(object(), settings)

    def test_prepare_ocr_image_resizes_then_thresholds_without_blur(self):
        gray = object()
        resized = object()
        prepared = object()
        settings = config.DEFAULTS["ocr"]
        resize = mock.Mock(return_value=resized)
        threshold = mock.Mock(return_value=prepared)
        gaussian_blur = mock.Mock()
        patches = [
            mock.patch.object(
                read_page.cv2,
                "INTER_CUBIC",
                2,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "resize",
                new=resize,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "GaussianBlur",
                new=gaussian_blur,
                create=True,
            ),
            mock.patch.object(
                read_page,
                "threshold_for_ocr",
                new=threshold,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            actual = read_page.prepare_ocr_image(gray, settings)
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertIs(actual, prepared)
        resize.assert_called_once_with(
            gray,
            None,
            fx=1.6,
            fy=1.6,
            interpolation=2,
        )
        threshold.assert_called_once_with(resized, settings)
        gaussian_blur.assert_not_called()

    def test_prepare_ocr_image_rejects_invalid_scale(self):
        for scale_factor in (0, -1, float("inf")):
            settings = dict(config.DEFAULTS["ocr"])
            settings["scale_factor"] = scale_factor
            with self.subTest(scale_factor=scale_factor):
                with self.assertRaisesRegex(ValueError, "scale_factor"):
                    read_page.prepare_ocr_image(object(), settings)


class DebugImageTests(unittest.TestCase):
    def test_debug_images_write_edges_overlay_and_warp(self):
        runtime = Path("runtime")
        frame = mock.Mock()
        overlay = object()
        frame.copy.return_value = overlay
        edges = object()
        warped = object()
        metadata = {
            "corners": [
                (10.2, 20.4),
                (100.6, 20.4),
                (100.6, 200.8),
                (10.2, 200.8),
            ],
            "edges": edges,
            "warped": warped,
        }
        line = mock.Mock()
        imwrite = mock.Mock()
        patches = [
            mock.patch.object(
                read_page.cv2,
                "line",
                new=line,
                create=True,
            ),
            mock.patch.object(
                read_page.cv2,
                "imwrite",
                new=imwrite,
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            read_page.write_ocr_debug_images(
                runtime,
                frame,
                metadata,
                True,
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

        line.assert_has_calls(
            [
                mock.call(overlay, (10, 20), (101, 20), (0, 255, 0), 3),
                mock.call(overlay, (101, 20), (101, 201), (0, 255, 0), 3),
                mock.call(overlay, (101, 201), (10, 201), (0, 255, 0), 3),
                mock.call(overlay, (10, 201), (10, 20), (0, 255, 0), 3),
            ]
        )
        imwrite.assert_has_calls(
            [
                mock.call(str(runtime / "page_edges.png"), edges),
                mock.call(str(runtime / "page_detected.jpg"), overlay),
                mock.call(str(runtime / "page_warped.png"), warped),
            ]
        )

    def test_debug_images_draw_inferred_corners_in_orange(self):
        runtime = Path("runtime")
        frame = mock.Mock()
        overlay = object()
        frame.copy.return_value = overlay
        metadata = {
            "corners": [
                (10, 20),
                (100, 20),
                (100, 200),
                (10, 200),
            ],
            "corners_inferred": True,
            "edges": None,
            "warped": None,
        }
        line = mock.Mock()

        with mock.patch.object(
            read_page.cv2,
            "line",
            new=line,
            create=True,
        ):
            with mock.patch.object(
                read_page.cv2,
                "imwrite",
                create=True,
            ):
                read_page.write_ocr_debug_images(
                    runtime,
                    frame,
                    metadata,
                    True,
                )

        self.assertEqual(line.call_count, 4)
        for call in line.call_args_list:
            self.assertEqual(call[0][-2:], ((0, 165, 255), 3))

    def test_debug_images_do_nothing_when_disabled(self):
        frame = mock.Mock()
        imwrite = mock.Mock()

        with mock.patch.object(
            read_page.cv2,
            "imwrite",
            new=imwrite,
            create=True,
        ):
            read_page.write_ocr_debug_images(
                Path("runtime"),
                frame,
                {},
                False,
            )

        frame.copy.assert_not_called()
        imwrite.assert_not_called()

    def test_debug_fallback_removes_stale_optional_images(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            edges_path = runtime / "page_edges.png"
            warped_path = runtime / "page_warped.png"
            edges_path.write_bytes(b"old edges")
            warped_path.write_bytes(b"old warp")
            frame = mock.Mock()
            overlay = object()
            frame.copy.return_value = overlay
            line = mock.Mock()
            imwrite = mock.Mock()
            metadata = {
                "corners": None,
                "edges": None,
                "warped": None,
            }
            patches = [
                mock.patch.object(
                    read_page.cv2,
                    "line",
                    new=line,
                    create=True,
                ),
                mock.patch.object(
                    read_page.cv2,
                    "imwrite",
                    new=imwrite,
                    create=True,
                ),
            ]

            for patch in patches:
                patch.start()
            try:
                read_page.write_ocr_debug_images(
                    runtime,
                    frame,
                    metadata,
                    True,
                )
            finally:
                for patch in reversed(patches):
                    patch.stop()

            self.assertFalse(edges_path.exists())
            self.assertFalse(warped_path.exists())
            line.assert_not_called()
            imwrite.assert_called_once_with(
                str(runtime / "page_detected.jpg"),
                overlay,
            )


class OcrImageTests(unittest.TestCase):
    def test_ocr_image_invokes_tesseract_and_strips_output(self):
        process = mock.Mock(
            returncode=0,
            stdout="  Rozpoznany tekst.\n",
            stderr="",
        )
        run = mock.Mock(return_value=process)
        settings = {
            "language": "pol",
            "page_segmentation_mode": 6,
        }
        image_path = Path("runtime") / "page_prepared.png"

        with mock.patch.object(read_page.subprocess, "run", new=run):
            actual = read_page.ocr_image(image_path, settings)

        self.assertEqual(actual, "Rozpoznany tekst.")
        run.assert_called_once_with(
            [
                "tesseract",
                str(image_path),
                "stdout",
                "-l",
                "pol",
                "--psm",
                "6",
            ],
            stdout=read_page.subprocess.PIPE,
            stderr=read_page.subprocess.PIPE,
            universal_newlines=True,
        )

    def test_ocr_image_raises_with_tesseract_error(self):
        process = mock.Mock(
            returncode=1,
            stdout="",
            stderr="missing language data\n",
        )

        with mock.patch.object(
            read_page.subprocess,
            "run",
            return_value=process,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "missing language data",
            ):
                read_page.ocr_image(
                    Path("runtime") / "page_prepared.png",
                    {
                        "language": "pol",
                        "page_segmentation_mode": 6,
                    },
                )


class ReadOnceTests(unittest.TestCase):
    def _run_with_ocr_text(self, text, max_characters=700):
        frame = object()
        prepared = object()
        runtime = Path("runtime")
        ocr_settings = dict(config.DEFAULTS["ocr"])
        ocr_settings["page_detection"] = dict(
            ocr_settings["page_detection"]
        )
        ocr_settings["max_characters"] = max_characters
        loaded_config = {"camera": {}, "ocr": ocr_settings}
        camera_session = mock.Mock()
        camera_session.capture_frame.return_value = frame
        metadata = {
            "page_detected": False,
            "used_fallback": True,
            "corners": None,
            "edges": None,
            "warped": None,
        }
        say = mock.Mock()
        patches = [
            mock.patch.object(
                read_page,
                "load_config",
                return_value=loaded_config,
            ),
            mock.patch.object(
                read_page,
                "ensure_runtime_dir",
                return_value=runtime,
            ),
            mock.patch.object(
                read_page,
                "preprocess_for_ocr",
                return_value=(prepared, metadata),
            ),
            mock.patch.object(
                read_page,
                "write_ocr_debug_images",
            ),
            mock.patch.object(
                read_page,
                "ocr_image",
                return_value=text,
            ),
            mock.patch.object(
                read_page,
                "say",
                new=say,
            ),
            mock.patch.object(
                read_page.cv2,
                "imwrite",
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            result = read_page.read_once(
                speak=True,
                camera_session=camera_session,
            )
        finally:
            for patch in reversed(patches):
                patch.stop()
        return result, say

    def test_read_once_reports_empty_ocr_result(self):
        result, say = self._run_with_ocr_text("x")

        self.assertEqual(result, "")
        self.assertEqual(
            say.call_args_list,
            [
                mock.call("Robię zdjęcie kartki."),
                mock.call("Rozpoznaję tekst."),
                mock.call(
                    "Nie udało mi się rozpoznać tekstu. "
                    "Przybliż kartkę i popraw światło."
                ),
            ],
        )

    def test_read_once_limits_speech_but_returns_full_text(self):
        result, say = self._run_with_ocr_text(
            "abcdefgh",
            max_characters=5,
        )

        self.assertEqual(result, "abcdefgh")
        self.assertEqual(
            say.call_args_list,
            [
                mock.call("Robię zdjęcie kartki."),
                mock.call("Rozpoznaję tekst."),
                mock.call("Czytam. abcde. Dalszy tekst pomijam."),
            ],
        )

    def test_read_once_uses_one_shot_capture_without_camera_session(self):
        frame = object()
        prepared = object()
        runtime = Path("runtime")
        camera_settings = {"device": 3}
        ocr_settings = config.DEFAULTS["ocr"]
        metadata = {
            "page_detected": False,
            "used_fallback": True,
            "corners": None,
            "edges": None,
            "warped": None,
        }
        capture = mock.Mock(return_value=frame)
        preprocess = mock.Mock(return_value=(prepared, metadata))
        patches = [
            mock.patch.object(
                read_page,
                "load_config",
                return_value={
                    "camera": camera_settings,
                    "ocr": ocr_settings,
                },
            ),
            mock.patch.object(
                read_page,
                "ensure_runtime_dir",
                return_value=runtime,
            ),
            mock.patch.object(
                read_page,
                "capture_frame",
                new=capture,
            ),
            mock.patch.object(
                read_page,
                "preprocess_for_ocr",
                new=preprocess,
            ),
            mock.patch.object(
                read_page,
                "write_ocr_debug_images",
            ),
            mock.patch.object(
                read_page,
                "ocr_image",
                return_value="Rozpoznany tekst",
            ),
            mock.patch.object(
                read_page.cv2,
                "imwrite",
                create=True,
            ),
        ]

        for patch in patches:
            patch.start()
        try:
            result = read_page.read_once(
                speak=False,
                camera_session=None,
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

        self.assertEqual(result, "Rozpoznany tekst")
        capture.assert_called_once_with(camera_settings)
        preprocess.assert_called_once_with(frame, ocr_settings)

    def test_read_once_passes_ocr_settings_to_preprocessing(self):
        frame = object()
        prepared = object()
        runtime = Path("runtime")
        ocr_settings = {
            "language": "pol",
            "page_segmentation_mode": 6,
            "max_characters": 700,
            "scale_factor": 1.6,
            "threshold_block_size": 31,
            "threshold_c": 11,
            "page_detection": {
                "debug_images": False,
            },
        }
        loaded_config = {"camera": {}, "ocr": ocr_settings}
        camera_session = mock.Mock()
        camera_session.capture_frame.return_value = frame
        metadata = {
            "page_detected": False,
            "used_fallback": True,
            "corners": None,
            "edges": None,
            "warped": None,
        }
        write_debug = mock.Mock()

        with mock.patch.object(
            read_page,
            "load_config",
            return_value=loaded_config,
        ):
            with mock.patch.object(
                read_page,
                "ensure_runtime_dir",
                return_value=runtime,
            ):
                with mock.patch.object(
                    read_page,
                    "preprocess_for_ocr",
                    return_value=(prepared, metadata),
                ) as preprocess:
                    with mock.patch.object(
                        read_page,
                        "write_ocr_debug_images",
                        new=write_debug,
                    ):
                        with mock.patch.object(
                            read_page,
                            "ocr_image",
                            return_value="Rozpoznany tekst",
                        ) as ocr:
                            with mock.patch.object(
                                read_page.cv2,
                                "imwrite",
                                create=True,
                            ) as imwrite:
                                result = read_page.read_once(
                                    speak=False,
                                    camera_session=camera_session,
                                )

        self.assertEqual(result, "Rozpoznany tekst")
        preprocess.assert_called_once_with(frame, ocr_settings)
        imwrite.assert_has_calls(
            [
                mock.call(str(runtime / "page_raw.jpg"), frame),
                mock.call(str(runtime / "page_prepared.png"), prepared),
            ]
        )
        ocr.assert_called_once_with(runtime / "page_prepared.png", ocr_settings)
        write_debug.assert_called_once_with(
            runtime,
            frame,
            metadata,
            False,
        )


if __name__ == "__main__":
    unittest.main()
