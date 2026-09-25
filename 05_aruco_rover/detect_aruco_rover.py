from __future__ import annotations

import cv2
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
CAPTURE_DIR = SCRIPT_DIR / "captures"

CAMERA_CALIBRATION_FILE = CALIBRATION_DIR / "camera_calibration.json"
ARENA_CALIBRATION_FILE = CALIBRATION_DIR / "arena_calibration.json"

CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# USER SETTINGS
# ============================================================

CAMERA_URL = "http://196.47.212.136:8080/video"

ARUCO_DICTIONARY = cv2.aruco.DICT_6X6_250
ROVER_MARKER_ID = 29

# The marker's canonical TOP edge must physically point toward
# the rover's front.
#
# Leave this at 0.0 if it is mounted exactly that way.
# If later you discover the physical mounting is rotated by a
# fixed amount, compensate here.
MARKER_HEADING_OFFSET_DEG = 0.0

RECONNECT_DELAY_S = 1.0


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


camera_data = load_json(CAMERA_CALIBRATION_FILE)
arena_data = load_json(ARENA_CALIBRATION_FILE)


# ============================================================
# CAMERA CALIBRATION
# ============================================================

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


# ============================================================
# ARENA CALIBRATION
# ============================================================

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

# Sub-pixel refinement improves centre/orientation stability.
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
# PREPROCESSING
# ============================================================

def undistort_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]

    if (w, h) != (camera_width, camera_height):
        raise RuntimeError(
            f"Camera returned {w}x{h}, "
            f"but calibration expects "
            f"{camera_width}x{camera_height}."
        )

    return cv2.remap(
        frame,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# PROJECTIVE HELPERS
# ============================================================

def transform_points(
    points_xy: np.ndarray,
    homography: np.ndarray
) -> np.ndarray:
    """
    points_xy shape: (N, 2)
    returns shape:   (N, 2)
    """

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
    """
    Intersection of line p1->p2 and p3->p4.

    For an ArUco quadrilateral, the diagonals should intersect
    at the projective centre of the marker.
    """

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

    determinant_12 = x1 * y2 - y1 * x2
    determinant_34 = x3 * y4 - y3 * x4

    px = (
        determinant_12 * (x3 - x4)
        - (x1 - x2) * determinant_34
    ) / denominator

    py = (
        determinant_12 * (y3 - y4)
        - (y1 - y2) * determinant_34
    ) / denominator

    return np.array(
        [px, py],
        dtype=np.float64
    )


# ============================================================
# ARUCO DETECTION
# ============================================================

def detect_aruco(
    undistorted: np.ndarray
):
    gray = cv2.cvtColor(
        undistorted,
        cv2.COLOR_BGR2GRAY
    )

    if aruco_detector is not None:
        corners, ids, rejected = (
            aruco_detector.detectMarkers(gray)
        )
    else:
        corners, ids, rejected = (
            cv2.aruco.detectMarkers(
                gray,
                aruco_dictionary,
                parameters=detector_parameters
            )
        )

    return corners, ids, rejected


# ============================================================
# ROVER POSE FROM MARKER
# ============================================================

def calculate_rover_pose(
    corners_px: np.ndarray
) -> dict:
    """
    OpenCV ArUco corner order:

        0 = canonical top-left
        1 = canonical top-right
        2 = canonical bottom-right
        3 = canonical bottom-left

    The marker's canonical TOP must point toward the physical
    rover front.

    Heading convention:
        0   = +Y / up
        90  = +X / right
        180 = down
        270 = left
    """

    corners_px = np.asarray(
        corners_px,
        dtype=np.float64
    ).reshape(4, 2)

    # Projectively correct image-space marker centre:
    # intersection of the two diagonals.
    centre_px = line_intersection(
        corners_px[0],
        corners_px[2],
        corners_px[1],
        corners_px[3]
    )

    # Transform all corners and centre into arena world space.
    corners_world = transform_points(
        corners_px,
        H_world
    ).astype(np.float64)

    centre_world = transform_points(
        centre_px.reshape(1, 2),
        H_world
    )[0].astype(np.float64)

    # Canonical top edge = corners 0 and 1.
    # Canonical bottom edge = corners 2 and 3.
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

    heading_deg = (
        math.degrees(
            math.atan2(dx, dy)
        )
        + MARKER_HEADING_OFFSET_DEG
    ) % 360.0

    # Also transform to the bird's-eye image for clean display.
    corners_bird = transform_points(
        corners_px,
        H_birdseye
    ).astype(np.float64)

    centre_bird = transform_points(
        centre_px.reshape(1, 2),
        H_birdseye
    )[0].astype(np.float64)

    top_centre_bird = 0.5 * (
        corners_bird[0]
        + corners_bird[1]
    )

    return {
        "centre_px": centre_px,
        "centre_world_mm": centre_world,
        "corners_px": corners_px,
        "corners_world_mm": corners_world,
        "corners_bird_px": corners_bird,
        "centre_bird_px": centre_bird,
        "top_centre_bird_px": top_centre_bird,
        "heading_deg": float(heading_deg),
    }


# ============================================================
# VISUALISATION
# ============================================================

CORNER_COLOURS = [
    (0, 255, 255),   # corner 0
    (0, 255, 0),     # corner 1
    (255, 0, 0),     # corner 2
    (255, 0, 255),   # corner 3
]


def draw_on_undistorted(
    image: np.ndarray,
    corners_px: np.ndarray,
    pose: dict
) -> None:

    polygon = np.round(
        corners_px
    ).astype(np.int32)

    cv2.polylines(
        image,
        [polygon],
        True,
        (0, 255, 0),
        3
    )

    for index, point in enumerate(polygon):
        point_tuple = (
            int(point[0]),
            int(point[1])
        )

        cv2.circle(
            image,
            point_tuple,
            8,
            CORNER_COLOURS[index],
            -1
        )

        cv2.putText(
            image,
            str(index),
            (
                point_tuple[0] + 9,
                point_tuple[1] - 9
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            CORNER_COLOURS[index],
            2
        )

    centre = tuple(
        np.round(
            pose["centre_px"]
        ).astype(int)
    )

    cv2.circle(
        image,
        centre,
        7,
        (255, 255, 255),
        -1
    )

    cv2.putText(
        image,
        f"ArUco ID {ROVER_MARKER_ID}",
        (centre[0] + 14, centre[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


def draw_on_birdseye(
    birdseye: np.ndarray,
    pose: dict
) -> None:

    corners = np.round(
        pose["corners_bird_px"]
    ).astype(np.int32)

    centre = np.round(
        pose["centre_bird_px"]
    ).astype(np.int32)

    front = np.round(
        pose["top_centre_bird_px"]
    ).astype(np.int32)

    cv2.polylines(
        birdseye,
        [corners],
        True,
        (0, 255, 0),
        3
    )

    for index, point in enumerate(corners):
        cv2.circle(
            birdseye,
            tuple(point),
            7,
            CORNER_COLOURS[index],
            -1
        )

        cv2.putText(
            birdseye,
            str(index),
            (
                int(point[0]) + 8,
                int(point[1]) - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            CORNER_COLOURS[index],
            2
        )

    cv2.circle(
        birdseye,
        tuple(centre),
        7,
        (255, 255, 255),
        -1
    )

    # Arrow from marker centre toward canonical marker top,
    # i.e. physical rover front.
    vector = (
        front.astype(np.float64)
        - centre.astype(np.float64)
    )

    norm = float(np.linalg.norm(vector))

    if norm > 1e-6:
        arrow_length = 100.0
        arrow_end = (
            centre.astype(np.float64)
            + vector / norm * arrow_length
        )

        arrow_end = tuple(
            np.round(arrow_end).astype(int)
        )

        cv2.arrowedLine(
            birdseye,
            tuple(centre),
            arrow_end,
            (0, 255, 255),
            4,
            tipLength=0.25
        )

    x_mm, y_mm = pose[
        "centre_world_mm"
    ]

    heading = pose[
        "heading_deg"
    ]

    line_1 = (
        f"ROVER: ({x_mm:.1f}, {y_mm:.1f}) mm"
    )

    line_2 = (
        f"HEADING: {heading:.1f} deg"
    )

    text_origin = (
        int(centre[0]) + 18,
        int(centre[1]) - 28
    )

    cv2.putText(
        birdseye,
        line_1,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        birdseye,
        line_2,
        (
            text_origin[0],
            text_origin[1] + 22
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )


# ============================================================
# SERIALISABLE OUTPUT
# ============================================================

def build_pose_document(
    pose: dict
) -> dict:

    x_mm, y_mm = pose[
        "centre_world_mm"
    ]

    corners_world = pose[
        "corners_world_mm"
    ]

    return {
        "marker_id": ROVER_MARKER_ID,
        "dictionary": "DICT_6X6_250",

        "position_mm": {
            "x": round(float(x_mm), 2),
            "y": round(float(y_mm), 2),
        },

        "heading_deg": round(
            float(pose["heading_deg"]),
            2
        ),

        "corners_world_mm": [
            {
                "index": index,
                "x": round(float(point[0]), 2),
                "y": round(float(point[1]), 2),
            }
            for index, point in enumerate(
                corners_world
            )
        ],
    }


# ============================================================
# SAVE SNAPSHOT
# ============================================================

def save_snapshot(
    annotated_undistorted: np.ndarray,
    annotated_birdseye: np.ndarray,
    document: dict
) -> None:

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    undistorted_path = (
        CAPTURE_DIR
        / f"{timestamp}_aruco_camera.png"
    )

    birdseye_path = (
        CAPTURE_DIR
        / f"{timestamp}_aruco_birdseye.png"
    )

    json_path = (
        CAPTURE_DIR
        / f"{timestamp}_aruco_pose.json"
    )

    cv2.imwrite(
        str(undistorted_path),
        annotated_undistorted
    )

    cv2.imwrite(
        str(birdseye_path),
        annotated_birdseye
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
    print(f"  {undistorted_path}")
    print(f"  {birdseye_path}")
    print(f"  {json_path}")
    print()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("AI LEARNER - ARUCO ROVER TEST")
    print("=============================")
    print()
    print("Dictionary : DICT_6X6_250")
    print(f"Rover ID   : {ROVER_MARKER_ID}")
    print()
    print(
        "Mount the canonical TOP of marker 29 "
        "toward the rover FRONT."
    )
    print()
    print("Controls:")
    print("  Q : quit")
    print("  S : save current detection + JSON")
    print()

    cap = open_camera()

    last_camera = None
    last_birdseye = None
    last_document = None

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

        corners, ids, _ = detect_aruco(
            undistorted
        )

        annotated_camera = (
            undistorted.copy()
        )

        birdseye = cv2.warpPerspective(
            undistorted,
            H_birdseye,
            (
                birdseye_width,
                birdseye_height
            )
        )

        annotated_birdseye = (
            birdseye.copy()
        )

        rover_found = False
        pose = None

        if ids is not None:

            flat_ids = ids.flatten()

            for marker_corners, marker_id in zip(
                corners,
                flat_ids
            ):

                if int(marker_id) != ROVER_MARKER_ID:
                    continue

                marker_corners = (
                    marker_corners
                    .reshape(4, 2)
                    .astype(np.float64)
                )

                pose = calculate_rover_pose(
                    marker_corners
                )

                draw_on_undistorted(
                    annotated_camera,
                    marker_corners,
                    pose
                )

                draw_on_birdseye(
                    annotated_birdseye,
                    pose
                )

                rover_found = True
                break

        if rover_found and pose is not None:

            x_mm, y_mm = (
                pose["centre_world_mm"]
            )

            heading = pose["heading_deg"]

            status = (
                f"ROVER FOUND | "
                f"x={x_mm:.1f} mm  "
                f"y={y_mm:.1f} mm  "
                f"heading={heading:.1f} deg"
            )

            status_colour = (
                0, 255, 0
            )

            last_document = (
                build_pose_document(
                    pose
                )
            )

            print(
                "\r"
                f"x={x_mm:7.1f} mm | "
                f"y={y_mm:7.1f} mm | "
                f"heading={heading:6.1f} deg",
                end="",
                flush=True
            )

        else:

            status = (
                f"ROVER MARKER {ROVER_MARKER_ID} "
                f"NOT FOUND"
            )

            status_colour = (
                0, 220, 255
            )

            last_document = None

        cv2.rectangle(
            annotated_birdseye,
            (0, 0),
            (birdseye_width, 48),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            annotated_birdseye,
            status,
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            status_colour,
            2
        )

        last_camera = (
            annotated_camera.copy()
        )

        last_birdseye = (
            annotated_birdseye.copy()
        )

        cv2.imshow(
            "ArUco Camera Detection",
            annotated_camera
        )

        cv2.imshow(
            "ArUco Rover Pose",
            annotated_birdseye
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            ord("Q")
        ):
            break

        elif key in (
            ord("s"),
            ord("S")
        ):

            if (
                last_camera is not None
                and last_birdseye is not None
                and last_document is not None
            ):
                save_snapshot(
                    last_camera,
                    last_birdseye,
                    last_document
                )

            else:
                print(
                    "\nCannot save pose: "
                    "marker 29 is not currently detected."
                )

    cap.release()
    cv2.destroyAllWindows()
    print()


if __name__ == "__main__":
    main()
