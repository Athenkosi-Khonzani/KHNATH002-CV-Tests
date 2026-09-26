from __future__ import annotations

import cv2
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CV_ROOT = SCRIPT_DIR.parent
CALIBRATION_DIR = CV_ROOT / "calibration"
RESULTS_DIR = SCRIPT_DIR / "results"

CAMERA_CALIBRATION_FILE = CALIBRATION_DIR / "camera_calibration.json"
ARENA_CALIBRATION_FILE = CALIBRATION_DIR / "arena_calibration.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_URL = "http://196.24.154.126:8080/video"

ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
ROVER_MARKER_ID = 29

MARKER_HEADING_OFFSET_DEG = 0.0

TARGET_HEADINGS_DEG = [
    0.0,
    45.0,
    90.0,
    135.0,
    180.0,
    225.0,
    270.0,
    315.0,
]

SAMPLES_PER_HEADING = 30
RECONNECT_DELAY_S = 1.0


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


camera_data = load_json(CAMERA_CALIBRATION_FILE)
arena_data = load_json(ARENA_CALIBRATION_FILE)

camera_matrix = np.array(camera_data["camera_matrix"], dtype=np.float64)
dist_coeffs = np.array(camera_data["distortion_coefficients"], dtype=np.float64)
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

map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix,
    dist_coeffs,
    None,
    camera_matrix,
    (camera_width, camera_height),
    cv2.CV_32FC1
)

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


def normalise_angle_deg(angle: float) -> float:
    return float(angle % 360.0)


def signed_angle_error_deg(measured: float, reference: float) -> float:
    return float(
        ((measured - reference + 180.0) % 360.0) - 180.0
    )


def circular_mean_deg(angles_deg: list[float]) -> float:
    angles_rad = np.deg2rad(
        np.asarray(angles_deg, dtype=np.float64)
    )

    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))

    mean_angle = math.degrees(
        math.atan2(mean_sin, mean_cos)
    )

    return normalise_angle_deg(mean_angle)


def circular_std_deg(angles_deg: list[float]) -> float:
    angles_rad = np.deg2rad(
        np.asarray(angles_deg, dtype=np.float64)
    )

    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))

    r = max(
        1e-12,
        math.hypot(mean_sin, mean_cos)
    )

    std_rad = math.sqrt(
        max(0.0, -2.0 * math.log(r))
    )

    return math.degrees(std_rad)


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


def reconnect_camera(cap: cv2.VideoCapture | None) -> cv2.VideoCapture:
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


def undistort_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]

    if (w, h) != (camera_width, camera_height):
        raise RuntimeError(
            f"Camera returned {w}x{h}, "
            f"but calibration expects {camera_width}x{camera_height}."
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


def detect_marker_29(
    undistorted: np.ndarray
) -> np.ndarray | None:

    gray = cv2.cvtColor(
        undistorted,
        cv2.COLOR_BGR2GRAY
    )

    if aruco_detector is not None:
        corners, ids, _ = aruco_detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            aruco_dictionary,
            parameters=detector_parameters
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


def calculate_heading_deg(
    corners_px: np.ndarray
) -> float:

    corners_world = transform_points(
        corners_px,
        H_world
    ).astype(np.float64)

    top_centre_world = 0.5 * (
        corners_world[0]
        + corners_world[1]
    )

    bottom_centre_world = 0.5 * (
        corners_world[2]
        + corners_world[3]
    )

    heading_vector = (
        top_centre_world
        - bottom_centre_world
    )

    dx = float(heading_vector[0])
    dy = float(heading_vector[1])

    return float(
        (
            math.degrees(
                math.atan2(dx, dy)
            )
            + MARKER_HEADING_OFFSET_DEG
        )
        % 360.0
    )


def draw_live_view(
    birdseye: np.ndarray,
    corners_px: np.ndarray | None,
    measured_heading: float | None,
    target_heading: float,
    target_index: int,
    captured_results: list[dict],
) -> np.ndarray:

    display = birdseye.copy()

    cv2.rectangle(
        display,
        (0, 0),
        (birdseye_width, 105),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        display,
        (
            f"Target {target_index + 1}/{len(TARGET_HEADINGS_DEG)}: "
            f"{target_heading:.0f} deg"
        ),
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    if measured_heading is None:
        measured_text = "Measured: MARKER NOT FOUND"
        measured_colour = (0, 220, 255)
    else:
        error = signed_angle_error_deg(
            measured_heading,
            target_heading
        )

        measured_text = (
            f"Measured: {measured_heading:.1f} deg   "
            f"instant error: {error:+.1f} deg"
        )

        measured_colour = (
            (0, 255, 0)
            if abs(error) <= 5.0
            else (0, 220, 255)
        )

    cv2.putText(
        display,
        measured_text,
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        measured_colour,
        2
    )

    cv2.putText(
        display,
        (
            f"Completed: {len(captured_results)}   "
            f"SPACE=capture {SAMPLES_PER_HEADING} samples"
        ),
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (210, 210, 210),
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

        centre = np.mean(
            corners_bird,
            axis=0
        )

        top_centre = 0.5 * (
            corners_bird[0]
            + corners_bird[1]
        )

        vector = top_centre - centre
        norm = float(np.linalg.norm(vector))

        if norm > 1e-6:
            arrow_end = (
                centre
                + vector / norm * 100.0
            )

            cv2.arrowedLine(
                display,
                tuple(np.round(centre).astype(int)),
                tuple(np.round(arrow_end).astype(int)),
                (0, 255, 255),
                4,
                tipLength=0.25
            )

    return display


def capture_heading_samples(
    cap: cv2.VideoCapture,
    target_heading: float
) -> tuple[cv2.VideoCapture, list[float]]:

    samples: list[float] = []

    print()
    print(
        f"Capturing {SAMPLES_PER_HEADING} valid samples "
        f"for target {target_heading:.0f} deg..."
    )

    while len(samples) < SAMPLES_PER_HEADING:

        ret, frame = cap.read()

        if not ret or frame is None:
            cap = reconnect_camera(cap)
            continue

        undistorted = undistort_frame(frame)
        corners = detect_marker_29(undistorted)

        if corners is None:
            continue

        heading = calculate_heading_deg(corners)
        samples.append(heading)

        print(
            f"\r  {len(samples):2d}/{SAMPLES_PER_HEADING}: "
            f"{heading:7.2f} deg",
            end="",
            flush=True
        )

    print()
    return cap, samples


def analyse_heading_samples(
    target_heading: float,
    samples: list[float]
) -> dict:

    mean_heading = circular_mean_deg(samples)
    circular_std = circular_std_deg(samples)

    sample_errors = [
        signed_angle_error_deg(
            sample,
            target_heading
        )
        for sample in samples
    ]

    return {
        "target_heading_deg": float(target_heading),
        "circular_mean_deg": float(mean_heading),
        "circular_std_deg": float(circular_std),
        "mean_signed_error_deg": float(np.mean(sample_errors)),
        "mean_absolute_error_deg": float(
            np.mean(np.abs(sample_errors))
        ),
        "max_absolute_error_deg": float(
            np.max(np.abs(sample_errors))
        ),
        "sample_count": len(samples),
        "samples_deg": [float(v) for v in samples],
    }


def calculate_recommended_offset(
    results: list[dict]
) -> float:

    if not results:
        return 0.0

    corrections = []

    for result in results:
        target = result["target_heading_deg"]
        measured = result["circular_mean_deg"]

        correction = signed_angle_error_deg(
            target,
            measured
        )

        corrections.append(
            correction % 360.0
        )

    mean_correction = circular_mean_deg(
        corrections
    )

    if mean_correction >= 180.0:
        mean_correction -= 360.0

    return float(mean_correction)


def save_results(
    results: list[dict]
) -> tuple[Path, Path]:

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    recommended_offset = (
        calculate_recommended_offset(
            results
        )
    )

    summary = {
        "dictionary": "DICT_4X4_50",
        "marker_id": ROVER_MARKER_ID,
        "samples_per_heading": SAMPLES_PER_HEADING,
        "heading_convention":
            "0=+Y/up, 90=+X/right, 180=down, 270=left",
        "current_offset_deg":
            MARKER_HEADING_OFFSET_DEG,
        "recommended_mounting_offset_deg":
            recommended_offset,
        "results": results,
    }

    json_path = (
        RESULTS_DIR
        / f"{timestamp}_heading_validation.json"
    )

    csv_path = (
        RESULTS_DIR
        / f"{timestamp}_heading_validation.csv"
    )

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "target_heading_deg",
                "circular_mean_deg",
                "circular_std_deg",
                "mean_signed_error_deg",
                "mean_absolute_error_deg",
                "max_absolute_error_deg",
                "sample_count",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result["target_heading_deg"],
                    result["circular_mean_deg"],
                    result["circular_std_deg"],
                    result["mean_signed_error_deg"],
                    result["mean_absolute_error_deg"],
                    result["max_absolute_error_deg"],
                    result["sample_count"],
                ]
            )

    return json_path, csv_path


def print_final_report(
    results: list[dict]
) -> None:

    print()
    print("HEADING VALIDATION RESULTS")
    print("==========================")

    for result in results:
        print(
            f"Target {result['target_heading_deg']:6.1f} deg | "
            f"mean {result['circular_mean_deg']:7.2f} deg | "
            f"std {result['circular_std_deg']:5.2f} deg | "
            f"MAE {result['mean_absolute_error_deg']:5.2f} deg | "
            f"max {result['max_absolute_error_deg']:5.2f} deg"
        )

    recommended_offset = (
        calculate_recommended_offset(
            results
        )
    )

    print()
    print(
        "Recommended fixed heading offset: "
        f"{recommended_offset:+.2f} deg"
    )

    print(
        "If this is consistently non-zero, either straighten "
        "the marker physically or later use:"
    )

    print(
        f"MARKER_HEADING_OFFSET_DEG = "
        f"{recommended_offset:.2f}"
    )


def main() -> None:

    print()
    print("AI LEARNER - STEP 6 HEADING VALIDATION")
    print("======================================")
    print()
    print("ArUco dictionary : DICT_4X4_50")
    print(f"Marker ID        : {ROVER_MARKER_ID}")
    print()
    print("Heading convention:")
    print("  0 deg   = UP / +Y")
    print("  90 deg  = RIGHT / +X")
    print("  180 deg = DOWN")
    print("  270 deg = LEFT")
    print()
    print(
        "Physically align the rover to the displayed target, "
        "then press SPACE."
    )
    print()
    print("Controls:")
    print("  SPACE : capture current target")
    print("  N     : skip current target")
    print("  R     : reset all results")
    print("  S     : save results")
    print("  Q     : quit")
    print()

    cap = open_camera()

    target_index = 0
    results: list[dict] = []

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            cap = reconnect_camera(cap)
            continue

        undistorted = undistort_frame(frame)
        corners = detect_marker_29(undistorted)

        measured_heading = None

        if corners is not None:
            measured_heading = calculate_heading_deg(
                corners
            )

        birdseye = cv2.warpPerspective(
            undistorted,
            H_birdseye,
            (birdseye_width, birdseye_height)
        )

        if target_index < len(TARGET_HEADINGS_DEG):
            target_heading = TARGET_HEADINGS_DEG[target_index]
        else:
            target_heading = TARGET_HEADINGS_DEG[-1]

        display = draw_live_view(
            birdseye,
            corners,
            measured_heading,
            target_heading,
            target_index,
            results
        )

        if target_index >= len(TARGET_HEADINGS_DEG):
            cv2.putText(
                display,
                "ALL TARGETS COMPLETE - press S to save",
                (15, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Heading Validation",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        elif key == ord(" "):

            if target_index >= len(TARGET_HEADINGS_DEG):
                print("\nAll target headings are complete.")
                continue

            if corners is None:
                print(
                    "\nMarker 29 is not currently detected."
                )
                continue

            target = TARGET_HEADINGS_DEG[target_index]

            cap, samples = capture_heading_samples(
                cap,
                target
            )

            result = analyse_heading_samples(
                target,
                samples
            )

            results.append(result)

            print(
                f"Target {target:.0f} deg complete: "
                f"mean={result['circular_mean_deg']:.2f}, "
                f"std={result['circular_std_deg']:.2f}, "
                f"MAE={result['mean_absolute_error_deg']:.2f}"
            )

            target_index += 1

            if target_index >= len(TARGET_HEADINGS_DEG):
                print_final_report(results)

        elif key in (ord("n"), ord("N")):
            if target_index < len(TARGET_HEADINGS_DEG):
                print(
                    f"\nSkipping target "
                    f"{TARGET_HEADINGS_DEG[target_index]:.0f} deg."
                )
                target_index += 1

        elif key in (ord("r"), ord("R")):
            results.clear()
            target_index = 0
            print("\nAll validation results reset.")

        elif key in (ord("s"), ord("S")):
            if not results:
                print("\nNo validation results to save yet.")
                continue

            json_path, csv_path = save_results(results)

            print_final_report(results)
            print()
            print("Saved:")
            print(f"  {json_path}")
            print(f"  {csv_path}")

    if results:
        print_final_report(results)

    cap.release()
    cv2.destroyAllWindows()
    print()


if __name__ == "__main__":
    main()
