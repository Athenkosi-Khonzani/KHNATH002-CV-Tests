from __future__ import annotations

import cv2
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CV_ROOT = SCRIPT_DIR.parent
CALIBRATION_DIR = CV_ROOT / "calibration"
RESULTS_DIR = SCRIPT_DIR / "results"

CAMERA_CALIBRATION_FILE = CALIBRATION_DIR / "camera_calibration.json"
ARENA_CALIBRATION_FILE = CALIBRATION_DIR / "arena_calibration.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# USER SETTINGS
# ============================================================

CAMERA_URL = "http://196.24.154.126:8080/video"

ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
ROVER_MARKER_ID = 29

# Step 6 showed only +0.06 deg recommended correction,
# so we keep the heading offset at zero.
MARKER_HEADING_OFFSET_DEG = 0.0

# Step 7 mechanical compensation:
# at heading 0, rover centre is 64 mm LEFT and 90 mm FORWARD
# of the ArUco centre.
ROVER_CENTRE_LEFT_MM = 64.0
ROVER_CENTRE_FORWARD_MM = 90.0

# Number of valid rover-pose samples averaged per test point.
SAMPLES_PER_POINT = 30

# Aim for 5-9 positions across the arena.
MIN_RECOMMENDED_POINTS = 5

RECONNECT_DELAY_S = 1.0


# ============================================================
# LOAD CALIBRATION
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


camera_data = load_json(CAMERA_CALIBRATION_FILE)
arena_data = load_json(ARENA_CALIBRATION_FILE)

camera_matrix = np.array(
    camera_data["camera_matrix"],
    dtype=np.float64
)

dist_coeffs = np.array(
    camera_data["distortion_coefficients"],
    dtype=np.float64
)

camera_width = int(camera_data["image_width"])
camera_height = int(camera_data["image_height"])

H_world = np.array(
    arena_data["homography_image_to_world"],
    dtype=np.float64
)

H_birdseye = np.array(
    arena_data["homography_image_to_birdseye"],
    dtype=np.float64
)

birdseye_width = int(arena_data["birdseye_width"])
birdseye_height = int(arena_data["birdseye_height"])

arena_width_mm = float(arena_data["arena_width_mm"])
arena_height_mm = float(arena_data["arena_height_mm"])


# ============================================================
# PRECOMPUTE UNDISTORTION
# ============================================================

map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix,
    dist_coeffs,
    None,
    camera_matrix,
    (camera_width, camera_height),
    cv2.CV_32FC1
)


# ============================================================
# ARUCO DETECTOR
# ============================================================

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    ARUCO_DICTIONARY
)

detector_parameters = cv2.aruco.DetectorParameters()
detector_parameters.cornerRefinementMethod = (
    cv2.aruco.CORNER_REFINE_SUBPIX
)

if hasattr(cv2.aruco, "ArucoDetector"):
    aruco_detector = cv2.aruco.ArucoDetector(
        aruco_dictionary,
        detector_parameters
    )
else:
    aruco_detector = None


# ============================================================
# ANGLE HELPERS
# ============================================================

def circular_mean_deg(values_deg: list[float]) -> float:
    values_rad = np.deg2rad(
        np.asarray(values_deg, dtype=np.float64)
    )

    mean_sin = np.mean(np.sin(values_rad))
    mean_cos = np.mean(np.cos(values_rad))

    angle = math.degrees(
        math.atan2(mean_sin, mean_cos)
    )

    return float(angle % 360.0)


def circular_std_deg(values_deg: list[float]) -> float:
    values_rad = np.deg2rad(
        np.asarray(values_deg, dtype=np.float64)
    )

    mean_sin = np.mean(np.sin(values_rad))
    mean_cos = np.mean(np.cos(values_rad))

    r = max(
        1e-12,
        math.hypot(mean_sin, mean_cos)
    )

    return float(
        math.degrees(
            math.sqrt(
                max(0.0, -2.0 * math.log(r))
            )
        )
    )


# ============================================================
# CAMERA CONNECTION
# ============================================================

def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture()

    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
    except Exception:
        pass

    cap.open(CAMERA_URL)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera stream: {CAMERA_URL}"
        )

    return cap


def reconnect_camera(
    cap: cv2.VideoCapture | None
) -> cv2.VideoCapture:

    if cap is not None:
        cap.release()

    print("Camera stream lost. Reconnecting...")
    time.sleep(RECONNECT_DELAY_S)

    while True:
        try:
            new_cap = open_camera()
            print("Camera reconnected.")
            return new_cap
        except RuntimeError as exc:
            print(exc)
            time.sleep(RECONNECT_DELAY_S)


# ============================================================
# IMAGE / PROJECTIVE HELPERS
# ============================================================

def undistort_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]

    if (w, h) != (camera_width, camera_height):
        raise RuntimeError(
            f"Camera returned {w}x{h}, but calibration expects "
            f"{camera_width}x{camera_height}."
        )

    return cv2.remap(
        frame,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR
    )


def transform_points(
    points_xy: np.ndarray,
    homography: np.ndarray
) -> np.ndarray:

    points = np.asarray(
        points_xy,
        dtype=np.float32
    ).reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(
        points,
        homography
    )

    return transformed.reshape(-1, 2)


def line_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray
) -> np.ndarray:

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denominator = (
        (x1 - x2) * (y3 - y4)
        - (y1 - y2) * (x3 - x4)
    )

    if abs(denominator) < 1e-9:
        return np.mean(
            np.array([p1, p2, p3, p4]),
            axis=0
        )

    det12 = x1 * y2 - y1 * x2
    det34 = x3 * y4 - y3 * x4

    px = (
        det12 * (x3 - x4)
        - (x1 - x2) * det34
    ) / denominator

    py = (
        det12 * (y3 - y4)
        - (y1 - y2) * det34
    ) / denominator

    return np.array(
        [px, py],
        dtype=np.float64
    )


# ============================================================
# ARUCO DETECTION
# ============================================================

def detect_rover_marker(
    undistorted: np.ndarray
) -> np.ndarray | None:

    gray = cv2.cvtColor(
        undistorted,
        cv2.COLOR_BGR2GRAY
    )

    if aruco_detector is not None:
        corners, ids, _ = (
            aruco_detector.detectMarkers(gray)
        )
    else:
        corners, ids, _ = (
            cv2.aruco.detectMarkers(
                gray,
                aruco_dictionary,
                parameters=detector_parameters
            )
        )

    if ids is None:
        return None

    for marker_corners, marker_id in zip(
        corners,
        ids.flatten()
    ):
        if int(marker_id) == ROVER_MARKER_ID:
            return (
                marker_corners
                .reshape(4, 2)
                .astype(np.float64)
            )

    return None


# ============================================================
# HEADING + POSITION COMPENSATION
# ============================================================

def calculate_heading_deg(
    corners_world: np.ndarray
) -> float:

    top_centre = 0.5 * (
        corners_world[0]
        + corners_world[1]
    )

    bottom_centre = 0.5 * (
        corners_world[2]
        + corners_world[3]
    )

    vector = top_centre - bottom_centre

    dx = float(vector[0])
    dy = float(vector[1])

    return float(
        (
            math.degrees(
                math.atan2(dx, dy)
            )
            + MARKER_HEADING_OFFSET_DEG
        )
        % 360.0
    )


def marker_to_rover_centre(
    marker_x_mm: float,
    marker_y_mm: float,
    heading_deg: float
) -> tuple[float, float]:

    theta = math.radians(
        heading_deg
    )

    forward_x = math.sin(theta)
    forward_y = math.cos(theta)

    left_x = -math.cos(theta)
    left_y = math.sin(theta)

    offset_x_mm = (
        ROVER_CENTRE_FORWARD_MM * forward_x
        + ROVER_CENTRE_LEFT_MM * left_x
    )

    offset_y_mm = (
        ROVER_CENTRE_FORWARD_MM * forward_y
        + ROVER_CENTRE_LEFT_MM * left_y
    )

    return (
        float(marker_x_mm + offset_x_mm),
        float(marker_y_mm + offset_y_mm),
    )


def calculate_rover_pose(
    corners_px: np.ndarray
) -> dict:

    corners_px = np.asarray(
        corners_px,
        dtype=np.float64
    ).reshape(4, 2)

    marker_centre_px = line_intersection(
        corners_px[0],
        corners_px[2],
        corners_px[1],
        corners_px[3]
    )

    corners_world = transform_points(
        corners_px,
        H_world
    ).astype(np.float64)

    marker_centre_world = transform_points(
        marker_centre_px.reshape(1, 2),
        H_world
    )[0].astype(np.float64)

    heading_deg = calculate_heading_deg(
        corners_world
    )

    rover_x_mm, rover_y_mm = marker_to_rover_centre(
        float(marker_centre_world[0]),
        float(marker_centre_world[1]),
        heading_deg
    )

    return {
        "x_mm": float(rover_x_mm),
        "y_mm": float(rover_y_mm),
        "heading_deg": float(heading_deg),
    }


# ============================================================
# LIVE DISPLAY
# ============================================================

def draw_live_view(
    birdseye: np.ndarray,
    corners_px: np.ndarray | None,
    pose: dict | None,
    point_count: int
) -> np.ndarray:

    display = birdseye.copy()

    cv2.rectangle(
        display,
        (0, 0),
        (birdseye_width, 95),
        (20, 20, 20),
        -1
    )

    if pose is None:
        pose_text = "ROVER MARKER NOT FOUND"
        colour = (0, 220, 255)
    else:
        pose_text = (
            f"ROVER: x={pose['x_mm']:.1f} mm   "
            f"y={pose['y_mm']:.1f} mm   "
            f"heading={pose['heading_deg']:.1f} deg"
        )
        colour = (0, 255, 0)

    cv2.putText(
        display,
        pose_text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        colour,
        2
    )

    cv2.putText(
        display,
        (
            f"Validated points: {point_count}   "
            f"SPACE = add measured test point"
        ),
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1
    )

    cv2.putText(
        display,
        (
            "S=save results   R=reset   Q=quit"
        ),
        (15, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1
    )

    if corners_px is not None:
        corners_bird = transform_points(
            corners_px,
            H_birdseye
        )

        polygon = np.round(
            corners_bird
        ).astype(np.int32)

        cv2.polylines(
            display,
            [polygon],
            True,
            (0, 255, 0),
            3
        )

    return display


# ============================================================
# COLLECT MULTIPLE SAMPLES AT ONE PHYSICAL POSITION
# ============================================================

def capture_pose_samples(
    cap: cv2.VideoCapture
) -> tuple[
    cv2.VideoCapture,
    list[dict]
]:

    samples: list[dict] = []

    print()
    print(
        f"Collecting {SAMPLES_PER_POINT} valid rover-pose samples..."
    )

    while len(samples) < SAMPLES_PER_POINT:

        ret, frame = cap.read()

        if not ret or frame is None:
            cap = reconnect_camera(cap)
            continue

        undistorted = undistort_frame(
            frame
        )

        corners = detect_rover_marker(
            undistorted
        )

        if corners is None:
            continue

        pose = calculate_rover_pose(
            corners
        )

        samples.append(
            pose
        )

        print(
            f"\r  {len(samples):2d}/{SAMPLES_PER_POINT} "
            f"x={pose['x_mm']:7.1f} "
            f"y={pose['y_mm']:7.1f} "
            f"h={pose['heading_deg']:6.1f}",
            end="",
            flush=True
        )

    print()

    return cap, samples


# ============================================================
# ANALYSE ONE VALIDATION POINT
# ============================================================

def analyse_test_point(
    point_id: int,
    true_x_mm: float,
    true_y_mm: float,
    samples: list[dict]
) -> dict:

    x_values = np.array(
        [s["x_mm"] for s in samples],
        dtype=np.float64
    )

    y_values = np.array(
        [s["y_mm"] for s in samples],
        dtype=np.float64
    )

    headings = [
        s["heading_deg"]
        for s in samples
    ]

    measured_x = float(
        np.mean(x_values)
    )

    measured_y = float(
        np.mean(y_values)
    )

    dx = measured_x - true_x_mm
    dy = measured_y - true_y_mm

    error_2d = math.hypot(
        dx,
        dy
    )

    arena_centre_x = (
        arena_width_mm / 2.0
    )

    arena_centre_y = (
        arena_height_mm / 2.0
    )

    radial_distance = math.hypot(
        true_x_mm - arena_centre_x,
        true_y_mm - arena_centre_y
    )

    return {
        "point_id":
            int(point_id),

        "true_x_mm":
            float(true_x_mm),

        "true_y_mm":
            float(true_y_mm),

        "measured_x_mm":
            measured_x,

        "measured_y_mm":
            measured_y,

        "dx_mm":
            float(dx),

        "dy_mm":
            float(dy),

        "error_2d_mm":
            float(error_2d),

        "x_std_mm":
            float(np.std(
                x_values,
                ddof=1
            )),

        "y_std_mm":
            float(np.std(
                y_values,
                ddof=1
            )),

        "heading_mean_deg":
            circular_mean_deg(
                headings
            ),

        "heading_std_deg":
            circular_std_deg(
                headings
            ),

        "distance_from_arena_centre_mm":
            float(radial_distance),

        "sample_count":
            len(samples),
    }


# ============================================================
# OVERALL SUMMARY
# ============================================================

def build_summary(
    results: list[dict]
) -> dict:

    if not results:
        return {}

    errors = np.array(
        [
            r["error_2d_mm"]
            for r in results
        ],
        dtype=np.float64
    )

    dx_values = np.array(
        [r["dx_mm"] for r in results],
        dtype=np.float64
    )

    dy_values = np.array(
        [r["dy_mm"] for r in results],
        dtype=np.float64
    )

    radial_distances = np.array(
        [
            r["distance_from_arena_centre_mm"]
            for r in results
        ],
        dtype=np.float64
    )

    # A simple diagnostic for whether error tends to grow away
    # from the arena centre. With few points, treat this only as
    # a clue, not a definitive statistical conclusion.
    radial_error_correlation = None

    if (
        len(results) >= 3
        and np.std(radial_distances) > 1e-9
        and np.std(errors) > 1e-9
    ):
        radial_error_correlation = float(
            np.corrcoef(
                radial_distances,
                errors
            )[0, 1]
        )

    return {
        "point_count":
            len(results),

        "mean_error_2d_mm":
            float(np.mean(errors)),

        "median_error_2d_mm":
            float(np.median(errors)),

        "max_error_2d_mm":
            float(np.max(errors)),

        "rmse_2d_mm":
            float(
                math.sqrt(
                    np.mean(
                        errors ** 2
                    )
                )
            ),

        "mean_dx_bias_mm":
            float(np.mean(dx_values)),

        "mean_dy_bias_mm":
            float(np.mean(dy_values)),

        "radial_error_correlation":
            radial_error_correlation,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: list[dict]
) -> tuple[Path, Path]:

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    summary = build_summary(
        results
    )

    document = {
        "configuration": {
            "dictionary":
                "DICT_4X4_50",

            "marker_id":
                ROVER_MARKER_ID,

            "rover_offset_body_mm": {
                "left":
                    ROVER_CENTRE_LEFT_MM,

                "forward":
                    ROVER_CENTRE_FORWARD_MM,
            },

            "samples_per_point":
                SAMPLES_PER_POINT,
        },

        "arena": {
            "width_mm":
                arena_width_mm,

            "height_mm":
                arena_height_mm,
        },

        "summary":
            summary,

        "points":
            results,
    }

    json_path = (
        RESULTS_DIR
        / f"{timestamp}_position_validation.json"
    )

    csv_path = (
        RESULTS_DIR
        / f"{timestamp}_position_validation.csv"
    )

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            document,
            file,
            indent=4
        )

    fieldnames = [
        "point_id",
        "true_x_mm",
        "true_y_mm",
        "measured_x_mm",
        "measured_y_mm",
        "dx_mm",
        "dy_mm",
        "error_2d_mm",
        "x_std_mm",
        "y_std_mm",
        "heading_mean_deg",
        "heading_std_deg",
        "distance_from_arena_centre_mm",
        "sample_count",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    key: result[key]
                    for key in fieldnames
                }
            )

    return json_path, csv_path


# ============================================================
# REPORT
# ============================================================

def print_point_result(
    result: dict
) -> None:

    print()
    print(
        f"Point {result['point_id']}: "
        f"true=({result['true_x_mm']:.1f}, "
        f"{result['true_y_mm']:.1f}) mm"
    )

    print(
        f"          measured="
        f"({result['measured_x_mm']:.1f}, "
        f"{result['measured_y_mm']:.1f}) mm"
    )

    print(
        f"          dx={result['dx_mm']:+.1f} mm   "
        f"dy={result['dy_mm']:+.1f} mm   "
        f"2D error={result['error_2d_mm']:.1f} mm"
    )

    print(
        f"          jitter: "
        f"sigma_x={result['x_std_mm']:.2f} mm   "
        f"sigma_y={result['y_std_mm']:.2f} mm"
    )


def print_summary(
    results: list[dict]
) -> None:

    if not results:
        return

    summary = build_summary(
        results
    )

    print()
    print("POSITION VALIDATION SUMMARY")
    print("===========================")

    print(
        f"Points tested : "
        f"{summary['point_count']}"
    )

    print(
        f"Mean error    : "
        f"{summary['mean_error_2d_mm']:.2f} mm"
    )

    print(
        f"Median error  : "
        f"{summary['median_error_2d_mm']:.2f} mm"
    )

    print(
        f"Max error     : "
        f"{summary['max_error_2d_mm']:.2f} mm"
    )

    print(
        f"RMSE          : "
        f"{summary['rmse_2d_mm']:.2f} mm"
    )

    print(
        f"Mean X bias   : "
        f"{summary['mean_dx_bias_mm']:+.2f} mm"
    )

    print(
        f"Mean Y bias   : "
        f"{summary['mean_dy_bias_mm']:+.2f} mm"
    )

    correlation = summary[
        "radial_error_correlation"
    ]

    if correlation is not None:
        print(
            f"Error vs distance-from-centre correlation: "
            f"{correlation:+.3f}"
        )


# ============================================================
# SAFE CONSOLE INPUT
# ============================================================

def ask_true_coordinates(
    point_number: int
) -> tuple[float, float] | None:

    print()
    print(
        f"Validation point {point_number}"
    )

    print(
        "Enter the PHYSICALLY MEASURED rover-centre coordinates."
    )

    print(
        "Press Enter without a value to cancel this point."
    )

    x_text = input(
        "True X [mm]: "
    ).strip()

    if x_text == "":
        return None

    y_text = input(
        "True Y [mm]: "
    ).strip()

    if y_text == "":
        return None

    try:
        x_mm = float(x_text)
        y_mm = float(y_text)
    except ValueError:
        print(
            "Invalid number. Point cancelled."
        )
        return None

    return x_mm, y_mm


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("AI LEARNER - STEP 7B POSITION VALIDATION")
    print("========================================")
    print()
    print(
        "This validates the COMPENSATED rover centre, "
        "not the raw ArUco centre."
    )
    print()
    print(
        f"Mechanical compensation: "
        f"{ROVER_CENTRE_LEFT_MM:.0f} mm LEFT, "
        f"{ROVER_CENTRE_FORWARD_MM:.0f} mm FORWARD."
    )
    print()
    print(
        f"Recommended: test at least "
        f"{MIN_RECOMMENDED_POINTS} positions across the arena."
    )
    print()
    print("Controls:")
    print("  SPACE : add one physically measured test point")
    print("  S     : save JSON + CSV results")
    print("  R     : reset all test points")
    print("  Q     : quit")
    print()

    cap = open_camera()

    results: list[dict] = []

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            cap = reconnect_camera(
                cap
            )
            continue

        undistorted = undistort_frame(
            frame
        )

        corners = detect_rover_marker(
            undistorted
        )

        pose = None

        if corners is not None:
            pose = calculate_rover_pose(
                corners
            )

        birdseye = cv2.warpPerspective(
            undistorted,
            H_birdseye,
            (
                birdseye_width,
                birdseye_height
            )
        )

        display = draw_live_view(
            birdseye,
            corners,
            pose,
            len(results)
        )

        cv2.imshow(
            "Position Validation",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            ord("Q")
        ):
            break

        elif key == ord(" "):

            if corners is None:
                print(
                    "\nMarker 29 is not currently detected."
                )
                continue

            # Temporarily allow console interaction cleanly.
            coordinates = ask_true_coordinates(
                len(results) + 1
            )

            if coordinates is None:
                continue

            true_x_mm, true_y_mm = (
                coordinates
            )

            cap, samples = (
                capture_pose_samples(
                    cap
                )
            )

            result = analyse_test_point(
                point_id=len(results) + 1,
                true_x_mm=true_x_mm,
                true_y_mm=true_y_mm,
                samples=samples
            )

            results.append(
                result
            )

            print_point_result(
                result
            )

        elif key in (
            ord("r"),
            ord("R")
        ):
            results.clear()
            print(
                "\nAll position-validation points reset."
            )

        elif key in (
            ord("s"),
            ord("S")
        ):

            if not results:
                print(
                    "\nNo test points to save yet."
                )
                continue

            json_path, csv_path = (
                save_results(
                    results
                )
            )

            print_summary(
                results
            )

            print()
            print("Saved:")
            print(f"  {json_path}")
            print(f"  {csv_path}")

    if results:
        print_summary(
            results
        )

    cap.release()
    cv2.destroyAllWindows()
    print()


if __name__ == "__main__":
    main()
