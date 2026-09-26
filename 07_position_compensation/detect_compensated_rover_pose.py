from __future__ import annotations

import cv2
import json
import math
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CV_ROOT = SCRIPT_DIR.parent
CALIBRATION_DIR = CV_ROOT / "calibration"
CAPTURE_DIR = SCRIPT_DIR / "captures"

CAMERA_CALIBRATION_FILE = CALIBRATION_DIR / "camera_calibration.json"
ARENA_CALIBRATION_FILE = CALIBRATION_DIR / "arena_calibration.json"

CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_URL = "http://196.24.154.126:8080/video"

ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
ROVER_MARKER_ID = 29

MARKER_HEADING_OFFSET_DEG = 0.0

ROVER_CENTRE_LEFT_MM = 64.0
ROVER_CENTRE_FORWARD_MM = 90.0

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
arena_width_mm = float(arena_data["arena_width_mm"])
arena_height_mm = float(arena_data["arena_height_mm"])

map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix,
    dist_coeffs,
    None,
    camera_matrix,
    (camera_width, camera_height),
    cv2.CV_32FC1
)

aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
detector_parameters = cv2.aruco.DetectorParameters()
detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

if hasattr(cv2.aruco, "ArucoDetector"):
    aruco_detector = cv2.aruco.ArucoDetector(
        aruco_dictionary,
        detector_parameters
    )
else:
    aruco_detector = None


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture()
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
    except Exception:
        pass

    cap.open(CAMERA_URL)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera stream: {CAMERA_URL}")

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
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, homography)
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
        return np.mean(np.array([p1, p2, p3, p4]), axis=0)

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

    return np.array([px, py], dtype=np.float64)


def detect_rover_marker(
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
            math.degrees(math.atan2(dx, dy))
            + MARKER_HEADING_OFFSET_DEG
        )
        % 360.0
    )


def marker_to_rover_centre(
    marker_x_mm: float,
    marker_y_mm: float,
    heading_deg: float
) -> tuple[float, float, float, float]:
    """
    At heading 0:
      rover centre is 64 mm LEFT and 90 mm FORWARD
      of the ArUco centre.

    With heading clockwise from +Y:
      forward(theta) = (sin(theta),  cos(theta))
      left(theta)    = (-cos(theta), sin(theta))
    """

    theta = math.radians(heading_deg)

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

    rover_x_mm = marker_x_mm + offset_x_mm
    rover_y_mm = marker_y_mm + offset_y_mm

    return (
        float(rover_x_mm),
        float(rover_y_mm),
        float(offset_x_mm),
        float(offset_y_mm),
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

    (
        rover_x_mm,
        rover_y_mm,
        offset_x_mm,
        offset_y_mm,
    ) = marker_to_rover_centre(
        float(marker_centre_world[0]),
        float(marker_centre_world[1]),
        heading_deg
    )

    corners_bird = transform_points(
        corners_px,
        H_birdseye
    ).astype(np.float64)

    marker_centre_bird = transform_points(
        marker_centre_px.reshape(1, 2),
        H_birdseye
    )[0].astype(np.float64)

    rover_centre_bird = np.array(
        [
            rover_x_mm / arena_width_mm * (birdseye_width - 1),
            (arena_height_mm - rover_y_mm)
            / arena_height_mm
            * (birdseye_height - 1),
        ],
        dtype=np.float64
    )

    return {
        "marker_centre_world_mm": marker_centre_world,
        "rover_centre_world_mm": np.array(
            [rover_x_mm, rover_y_mm],
            dtype=np.float64
        ),
        "offset_world_mm": np.array(
            [offset_x_mm, offset_y_mm],
            dtype=np.float64
        ),
        "heading_deg": float(heading_deg),
        "corners_bird_px": corners_bird,
        "marker_centre_bird_px": marker_centre_bird,
        "rover_centre_bird_px": rover_centre_bird,
    }


def draw_pose(
    birdseye: np.ndarray,
    pose: dict
) -> np.ndarray:

    display = birdseye.copy()

    corners = np.round(
        pose["corners_bird_px"]
    ).astype(np.int32)

    marker_centre = np.round(
        pose["marker_centre_bird_px"]
    ).astype(np.int32)

    rover_centre = np.round(
        pose["rover_centre_bird_px"]
    ).astype(np.int32)

    cv2.polylines(
        display,
        [corners],
        True,
        (0, 255, 0),
        3
    )

    cv2.circle(
        display,
        tuple(marker_centre),
        7,
        (255, 0, 255),
        -1
    )

    cv2.circle(
        display,
        tuple(rover_centre),
        9,
        (255, 255, 0),
        -1
    )

    cv2.arrowedLine(
        display,
        tuple(marker_centre),
        tuple(rover_centre),
        (255, 255, 0),
        3,
        tipLength=0.22
    )

    theta = math.radians(
        pose["heading_deg"]
    )

    heading_length_px = 110.0

    forward_px = np.array(
        [
            math.sin(theta),
            -math.cos(theta)
        ],
        dtype=np.float64
    )

    heading_end = (
        pose["rover_centre_bird_px"]
        + heading_length_px * forward_px
    )

    cv2.arrowedLine(
        display,
        tuple(rover_centre),
        tuple(np.round(heading_end).astype(int)),
        (0, 255, 255),
        4,
        tipLength=0.25
    )

    marker_x, marker_y = (
        pose["marker_centre_world_mm"]
    )

    rover_x, rover_y = (
        pose["rover_centre_world_mm"]
    )

    dx, dy = pose["offset_world_mm"]

    heading = pose["heading_deg"]

    cv2.rectangle(
        display,
        (0, 0),
        (birdseye_width, 115),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        display,
        (
            f"ROVER CENTRE: x={rover_x:.1f} mm   "
            f"y={rover_y:.1f} mm   "
            f"heading={heading:.1f} deg"
        ),
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        (
            f"ArUco centre: "
            f"({marker_x:.1f}, {marker_y:.1f}) mm"
        ),
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 0, 255),
        2
    )

    cv2.putText(
        display,
        (
            f"Rotated compensation: "
            f"dx={dx:+.1f} mm   dy={dy:+.1f} mm"
        ),
        (15, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        (
            "MAGENTA=ArUco centre   "
            "CYAN=compensated rover centre"
        ),
        (15, 107),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1
    )

    return display


def build_pose_document(
    pose: dict
) -> dict:

    marker_x, marker_y = (
        pose["marker_centre_world_mm"]
    )

    rover_x, rover_y = (
        pose["rover_centre_world_mm"]
    )

    dx, dy = pose["offset_world_mm"]

    return {
        "marker": {
            "dictionary": "DICT_4X4_50",
            "id": ROVER_MARKER_ID,
            "centre_mm": {
                "x": round(float(marker_x), 2),
                "y": round(float(marker_y), 2),
            },
        },

        "rover_reference_offset_body_mm": {
            "left": ROVER_CENTRE_LEFT_MM,
            "forward": ROVER_CENTRE_FORWARD_MM,
        },

        "rotated_offset_world_mm": {
            "x": round(float(dx), 2),
            "y": round(float(dy), 2),
        },

        "rover_pose": {
            "x_mm": round(float(rover_x), 2),
            "y_mm": round(float(rover_y), 2),
            "heading_deg": round(float(pose["heading_deg"]), 2),
        },
    }


def save_snapshot(
    annotated: np.ndarray,
    document: dict
) -> None:

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    image_path = (
        CAPTURE_DIR
        / f"{timestamp}_compensated_rover_pose.png"
    )

    json_path = (
        CAPTURE_DIR
        / f"{timestamp}_compensated_rover_pose.json"
    )

    cv2.imwrite(
        str(image_path),
        annotated
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

    print()
    print("Saved:")
    print(f"  {image_path}")
    print(f"  {json_path}")
    print()


def main() -> None:

    print()
    print("AI LEARNER - STEP 7 COMPENSATED ROVER POSITION")
    print("==============================================")
    print()
    print("ArUco dictionary : DICT_4X4_50")
    print(f"Marker ID        : {ROVER_MARKER_ID}")
    print()
    print(
        "Rover centre relative to ArUco centre at heading 0:"
    )
    print(
        f"  LEFT    = {ROVER_CENTRE_LEFT_MM:.1f} mm"
    )
    print(
        f"  FORWARD = {ROVER_CENTRE_FORWARD_MM:.1f} mm"
    )
    print()
    print("Controls:")
    print("  S : save current compensated pose")
    print("  Q : quit")
    print()

    cap = open_camera()

    last_annotated = None
    last_document = None

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            cap = reconnect_camera(cap)
            continue

        undistorted = undistort_frame(frame)

        corners = detect_rover_marker(
            undistorted
        )

        birdseye = cv2.warpPerspective(
            undistorted,
            H_birdseye,
            (
                birdseye_width,
                birdseye_height
            )
        )

        if corners is not None:

            pose = calculate_rover_pose(
                corners
            )

            annotated = draw_pose(
                birdseye,
                pose
            )

            document = build_pose_document(
                pose
            )

            rover_x, rover_y = (
                pose["rover_centre_world_mm"]
            )

            heading = pose["heading_deg"]

            print(
                "\r"
                f"rover x={rover_x:7.1f} mm | "
                f"y={rover_y:7.1f} mm | "
                f"heading={heading:6.1f} deg",
                end="",
                flush=True
            )

            last_annotated = annotated.copy()
            last_document = document

        else:

            annotated = birdseye.copy()

            cv2.rectangle(
                annotated,
                (0, 0),
                (birdseye_width, 50),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                annotated,
                f"ROVER MARKER {ROVER_MARKER_ID} NOT FOUND",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 220, 255),
                2
            )

            last_document = None

        cv2.imshow(
            "Compensated Rover Position",
            annotated
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        elif key in (ord("s"), ord("S")):

            if (
                last_annotated is not None
                and last_document is not None
            ):
                save_snapshot(
                    last_annotated,
                    last_document
                )
            else:
                print(
                    "\nCannot save: rover marker "
                    "is not currently detected."
                )

    cap.release()
    cv2.destroyAllWindows()
    print()


if __name__ == "__main__":
    main()
