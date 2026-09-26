from __future__ import annotations

import cv2
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CV_ROOT = SCRIPT_DIR.parent
CALIBRATION_DIR = CV_ROOT / "calibration"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

CAMERA_CALIBRATION_FILE = CALIBRATION_DIR / "camera_calibration.json"
ARENA_CALIBRATION_FILE = CALIBRATION_DIR / "arena_calibration.json"
COLOUR_CALIBRATION_FILE = CALIBRATION_DIR / "colour_calibration.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# USER SETTINGS
# ============================================================

CAMERA_URL = "http://196.24.154.126:8080/video"

# Rover marker
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
ROVER_MARKER_ID = 29
MARKER_HEADING_OFFSET_DEG = 0.0

# Mechanical compensation established in Step 7
ROVER_CENTRE_LEFT_MM = 64.0
ROVER_CENTRE_FORWARD_MM = 90.0

# Block detection
MIN_BLOCK_AREA_MM2 = 10_000.0
MIN_BLOCK_RECTANGULARITY = 0.70
MIN_BLOCK_SIDE_MM = 40.0
BLOCK_MORPH_KERNEL_SIZE = 5

# Pose filtering from Step 8
POSITION_FILTER_TAU_S = 0.12
HEADING_FILTER_TAU_S = 0.08
MAX_TRANSLATIONAL_SPEED_MM_S = 2500.0
MAX_ANGULAR_SPEED_DEG_S = 900.0
POSITION_GATE_MARGIN_MM = 30.0
HEADING_GATE_MARGIN_DEG = 8.0
POSE_HOLD_SECONDS = 0.50
CONFIDENCE_WINDOW_FRAMES = 20
MIN_USABLE_MARKER_SIDE_PX = 20.0
FULL_CONF_MARKER_SIDE_PX = 60.0

# Runtime
RECONNECT_DELAY_S = 1.0
AUTO_WRITE_LATEST = True

DISPLAY_COLOURS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
}

COLOUR_ORDER = ["red", "green", "yellow", "blue"]

# New text-map output
TEXT_MAP_CELL_MM = 100.0
TEXT_MAP_MAX_LABEL_MM = 9999  # formatting helper


# ============================================================
# DATA TYPES
# ============================================================

@dataclass
class PoseMeasurement:
    x_mm: float
    y_mm: float
    heading_deg: float
    marker_side_px: float
    timestamp: float


@dataclass
class FilteredPose:
    x_mm: float
    y_mm: float
    heading_deg: float
    confidence: float
    valid: bool
    stale: bool
    age_s: float
    marker_detected: bool
    measurement_accepted: bool


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
colour_data = load_json(COLOUR_CALIBRATION_FILE)

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

colour_calibrations = colour_data["colours"]

MM_PER_PX_X = arena_width_mm / max(1, birdseye_width - 1)
MM_PER_PX_Y = arena_height_mm / max(1, birdseye_height - 1)


# ============================================================
# PRECOMPUTE CAMERA UNDISTORTION
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
# GENERAL HELPERS
# ============================================================

def shortest_angle_delta_deg(
    target_deg: float,
    current_deg: float
) -> float:
    return float(
        ((target_deg - current_deg + 180.0) % 360.0) - 180.0
    )


def ema_alpha(dt_s: float, tau_s: float) -> float:
    if tau_s <= 0:
        return 1.0
    return float(
        1.0 - math.exp(-max(0.0, dt_s) / tau_s)
    )


def transform_points(
    points_xy: np.ndarray,
    homography: np.ndarray
) -> np.ndarray:
    points = np.asarray(
        points_xy,
        dtype=np.float32
    ).reshape(-1, 1, 2)

    return cv2.perspectiveTransform(
        points,
        homography
    ).reshape(-1, 2)


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


def birdseye_to_world(
    x_px: float,
    y_px: float
) -> tuple[float, float]:
    x_mm = float(x_px) * MM_PER_PX_X
    y_mm = arena_height_mm - float(y_px) * MM_PER_PX_Y
    return x_mm, y_mm


def world_to_birdseye(
    x_mm: float,
    y_mm: float
) -> np.ndarray:
    return np.array(
        [
            x_mm / arena_width_mm * (birdseye_width - 1),
            (arena_height_mm - y_mm)
            / arena_height_mm * (birdseye_height - 1),
        ],
        dtype=np.float64
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
# BLOCK DETECTION
# ============================================================

def create_colour_mask(
    hsv_image: np.ndarray,
    calibration: dict
) -> np.ndarray:

    mask = np.zeros(
        hsv_image.shape[:2],
        dtype=np.uint8
    )

    for hue_low, hue_high in calibration["hue_ranges"]:
        lower = np.array(
            [
                hue_low,
                calibration["sat_min"],
                calibration["val_min"]
            ],
            dtype=np.uint8
        )

        upper = np.array(
            [
                hue_high,
                calibration["sat_max"],
                calibration["val_max"]
            ],
            dtype=np.uint8
        )

        partial = cv2.inRange(
            hsv_image,
            lower,
            upper
        )

        mask = cv2.bitwise_or(
            mask,
            partial
        )

    return mask


def clean_block_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            BLOCK_MORPH_KERNEL_SIZE,
            BLOCK_MORPH_KERNEL_SIZE
        )
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def contour_area_mm2(
    contour: np.ndarray
) -> float:
    return float(
        cv2.contourArea(contour)
        * MM_PER_PX_X
        * MM_PER_PX_Y
    )


def analyse_block_contour(
    contour: np.ndarray
) -> dict | None:

    rect = cv2.minAreaRect(
        contour
    )

    centre_px = rect[0]

    box_px = cv2.boxPoints(
        rect
    ).astype(np.float64)

    box_world = np.array(
        [
            birdseye_to_world(x, y)
            for x, y in box_px
        ],
        dtype=np.float64
    )

    edge_01 = float(
        np.linalg.norm(
            box_world[1] - box_world[0]
        )
    )

    edge_12 = float(
        np.linalg.norm(
            box_world[2] - box_world[1]
        )
    )

    width_mm = max(
        edge_01,
        edge_12
    )

    height_mm = min(
        edge_01,
        edge_12
    )

    if width_mm <= 0 or height_mm <= 0:
        return None

    area_mm2 = contour_area_mm2(
        contour
    )

    rectangle_area_mm2 = (
        width_mm * height_mm
    )

    rectangularity = (
        area_mm2 / rectangle_area_mm2
        if rectangle_area_mm2 > 0
        else 0.0
    )

    centre_mm = birdseye_to_world(
        centre_px[0],
        centre_px[1]
    )

    if edge_01 >= edge_12:
        start = box_world[0]
        end = box_world[1]
    else:
        start = box_world[1]
        end = box_world[2]

    vector = end - start

    orientation_deg = (
        math.degrees(
            math.atan2(
                vector[1],
                vector[0]
            )
        )
        % 180.0
    )

    return {
        "centre_px": (
            float(centre_px[0]),
            float(centre_px[1])
        ),
        "centre_mm": (
            float(centre_mm[0]),
            float(centre_mm[1])
        ),
        "box_px": box_px,
        "box_world_mm": box_world,
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "area_mm2": float(area_mm2),
        "rectangularity": float(rectangularity),
        "orientation_deg": float(orientation_deg),
    }


def detect_blocks(
    birdseye: np.ndarray
) -> list[dict]:

    hsv = cv2.cvtColor(
        birdseye,
        cv2.COLOR_BGR2HSV
    )

    blocks: list[dict] = []

    for colour in COLOUR_ORDER:
        mask = create_colour_mask(
            hsv,
            colour_calibrations[colour]
        )

        mask = clean_block_mask(
            mask
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < 3:
                continue

            info = analyse_block_contour(
                contour
            )

            if info is None:
                continue

            if info["area_mm2"] < MIN_BLOCK_AREA_MM2:
                continue

            if info["rectangularity"] < MIN_BLOCK_RECTANGULARITY:
                continue

            if (
                info["width_mm"] < MIN_BLOCK_SIDE_MM
                or info["height_mm"] < MIN_BLOCK_SIDE_MM
            ):
                continue

            blocks.append(
                {
                    "colour": colour,
                    **info
                }
            )

    blocks.sort(
        key=lambda item: (
            COLOUR_ORDER.index(item["colour"]),
            item["centre_mm"][0],
            item["centre_mm"][1]
        )
    )

    colour_counts = {
        colour: 0
        for colour in COLOUR_ORDER
    }

    for block in blocks:
        colour = block["colour"]
        colour_counts[colour] += 1

        block["id"] = (
            f"{colour}_block_"
            f"{colour_counts[colour]:02d}"
        )

    return blocks


# ============================================================
# ROVER RAW POSE
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

    vector = (
        top_centre
        - bottom_centre
    )

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

    offset_x = (
        ROVER_CENTRE_FORWARD_MM * forward_x
        + ROVER_CENTRE_LEFT_MM * left_x
    )

    offset_y = (
        ROVER_CENTRE_FORWARD_MM * forward_y
        + ROVER_CENTRE_LEFT_MM * left_y
    )

    return (
        float(marker_x_mm + offset_x),
        float(marker_y_mm + offset_y)
    )


def mean_marker_side_px(
    corners_px: np.ndarray
) -> float:

    lengths = []

    for index in range(4):
        next_index = (index + 1) % 4
        lengths.append(
            float(
                np.linalg.norm(
                    corners_px[next_index]
                    - corners_px[index]
                )
            )
        )

    return float(
        np.mean(lengths)
    )


def make_raw_pose_measurement(
    corners_px: np.ndarray,
    timestamp: float
) -> PoseMeasurement:

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

    rover_x, rover_y = marker_to_rover_centre(
        float(marker_centre_world[0]),
        float(marker_centre_world[1]),
        heading_deg
    )

    return PoseMeasurement(
        x_mm=rover_x,
        y_mm=rover_y,
        heading_deg=heading_deg,
        marker_side_px=mean_marker_side_px(corners_px),
        timestamp=timestamp
    )


# ============================================================
# TEMPORAL POSE FILTER
# ============================================================

class RoverPoseFilter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.initialised = False
        self.x_mm = 0.0
        self.y_mm = 0.0
        self.heading_deg = 0.0

        self.last_filter_time: float | None = None
        self.last_good_time: float | None = None

        self.recent_acceptance = deque(
            maxlen=CONFIDENCE_WINDOW_FRAMES
        )

        self.last_size_score = 0.0

    def marker_size_score(
        self,
        side_px: float
    ) -> float:

        if side_px <= MIN_USABLE_MARKER_SIDE_PX:
            return 0.0

        if side_px >= FULL_CONF_MARKER_SIDE_PX:
            return 1.0

        return float(
            (
                side_px - MIN_USABLE_MARKER_SIDE_PX
            )
            / (
                FULL_CONF_MARKER_SIDE_PX
                - MIN_USABLE_MARKER_SIDE_PX
            )
        )

    def is_plausible(
        self,
        measurement: PoseMeasurement
    ) -> bool:

        if not self.initialised:
            return True

        if self.last_filter_time is None:
            return True

        dt = max(
            1e-3,
            measurement.timestamp - self.last_filter_time
        )

        position_change = math.hypot(
            measurement.x_mm - self.x_mm,
            measurement.y_mm - self.y_mm
        )

        heading_change = abs(
            shortest_angle_delta_deg(
                measurement.heading_deg,
                self.heading_deg
            )
        )

        position_limit = (
            MAX_TRANSLATIONAL_SPEED_MM_S * dt
            + POSITION_GATE_MARGIN_MM
        )

        heading_limit = (
            MAX_ANGULAR_SPEED_DEG_S * dt
            + HEADING_GATE_MARGIN_DEG
        )

        return (
            position_change <= position_limit
            and heading_change <= heading_limit
        )

    def update(
        self,
        measurement: PoseMeasurement | None,
        now: float
    ) -> FilteredPose | None:

        marker_detected = measurement is not None
        accepted = False

        if measurement is not None:
            size_score = self.marker_size_score(
                measurement.marker_side_px
            )

            if (
                size_score > 0.0
                and self.is_plausible(measurement)
            ):
                accepted = True
                self.last_size_score = size_score

                if not self.initialised:
                    self.x_mm = measurement.x_mm
                    self.y_mm = measurement.y_mm
                    self.heading_deg = measurement.heading_deg
                    self.initialised = True
                else:
                    if self.last_filter_time is None:
                        dt = 1.0 / 30.0
                    else:
                        dt = max(
                            1e-3,
                            now - self.last_filter_time
                        )

                    alpha_position = ema_alpha(
                        dt,
                        POSITION_FILTER_TAU_S
                    )

                    alpha_heading = ema_alpha(
                        dt,
                        HEADING_FILTER_TAU_S
                    )

                    self.x_mm += (
                        alpha_position
                        * (measurement.x_mm - self.x_mm)
                    )

                    self.y_mm += (
                        alpha_position
                        * (measurement.y_mm - self.y_mm)
                    )

                    heading_delta = shortest_angle_delta_deg(
                        measurement.heading_deg,
                        self.heading_deg
                    )

                    self.heading_deg = (
                        self.heading_deg
                        + alpha_heading * heading_delta
                    ) % 360.0

                self.last_good_time = now

        self.recent_acceptance.append(
            1 if accepted else 0
        )

        self.last_filter_time = now

        if not self.initialised or self.last_good_time is None:
            return None

        age_s = max(
            0.0,
            now - self.last_good_time
        )

        valid = age_s <= POSE_HOLD_SECONDS
        stale = age_s > 0.05

        hit_ratio = (
            sum(self.recent_acceptance)
            / max(1, len(self.recent_acceptance))
        )

        base_confidence = (
            0.65 * hit_ratio
            + 0.35 * self.last_size_score
        )

        decay = math.exp(
            -3.0 * age_s / max(POSE_HOLD_SECONDS, 1e-6)
        )

        confidence = float(
            np.clip(
                base_confidence * decay,
                0.0,
                1.0
            )
        )

        if not valid:
            confidence = 0.0

        return FilteredPose(
            x_mm=float(self.x_mm),
            y_mm=float(self.y_mm),
            heading_deg=float(self.heading_deg),
            confidence=confidence,
            valid=valid,
            stale=stale,
            age_s=float(age_s),
            marker_detected=marker_detected,
            measurement_accepted=accepted
        )


# ============================================================
# WORLD STATE SERIALISATION
# ============================================================

def serialise_block(
    block: dict
) -> dict:

    polygon = [
        {
            "x": round(float(point[0]), 1),
            "y": round(float(point[1]), 1),
        }
        for point in block["box_world_mm"]
    ]

    return {
        "id": block["id"],
        "type": "coloured_block",
        "colour": block["colour"],
        "centre_mm": {
            "x": round(block["centre_mm"][0], 1),
            "y": round(block["centre_mm"][1], 1),
        },
        "size_mm": {
            "long_side": round(block["width_mm"], 1),
            "short_side": round(block["height_mm"], 1),
        },
        "orientation_deg": round(block["orientation_deg"], 1),
        "polygon_mm": polygon,
    }


def build_world_state(
    filtered_pose: FilteredPose | None,
    blocks: list[dict]
) -> dict:

    rover = {
        "available": False,
        "pose_valid": False,
        "confidence": 0.0,
        "pose": None,
    }

    if filtered_pose is not None:
        rover = {
            "available": True,
            "pose_valid": filtered_pose.valid,
            "confidence": round(filtered_pose.confidence, 3),
            "stale": filtered_pose.stale,
            "age_s": round(filtered_pose.age_s, 3),
            "pose": {
                "x_mm": round(filtered_pose.x_mm, 1),
                "y_mm": round(filtered_pose.y_mm, 1),
                "heading_deg": round(filtered_pose.heading_deg, 1),
            }
        }

    return {
        "schema": "ai_learner_world_state",
        "schema_version": "1.0",
        "coordinate_frame": {
            "units": "mm",
            "origin": "arena_bottom_left",
            "x_axis": "right",
            "y_axis": "up",
            "heading": {
                "0_deg": "+Y/up",
                "90_deg": "+X/right",
                "180_deg": "-Y/down",
                "270_deg": "-X/left",
            }
        },
        "arena": {
            "width_mm": round(arena_width_mm, 1),
            "height_mm": round(arena_height_mm, 1),
        },
        "rover": rover,
        "obstacles": [
            serialise_block(block)
            for block in blocks
        ],
        "perception": {
            "block_count": len(blocks),
            "rover_marker_id": ROVER_MARKER_ID,
            "rover_marker_dictionary": "DICT_4X4_50",
        }
    }


def world_state_to_markdown(
    world_state: dict
) -> str:

    lines = []

    lines.append("# AI Learner World State")
    lines.append("")

    arena = world_state["arena"]

    lines.append(
        f"Arena: {arena['width_mm']:.0f} mm × "
        f"{arena['height_mm']:.0f} mm."
    )

    lines.append(
        "Coordinate frame: origin at bottom-left, "
        "+X right, +Y up, heading 0° toward +Y."
    )

    lines.append("")

    rover = world_state["rover"]

    lines.append("## Rover")

    if not rover["available"] or rover["pose"] is None:
        lines.append("Rover pose unavailable.")
    else:
        pose = rover["pose"]

        lines.append(
            f"- Position: ({pose['x_mm']:.1f}, "
            f"{pose['y_mm']:.1f}) mm"
        )

        lines.append(
            f"- Heading: {pose['heading_deg']:.1f}°"
        )

        lines.append(
            f"- Pose valid: {rover['pose_valid']}"
        )

        lines.append(
            f"- Confidence: {rover['confidence']:.3f}"
        )

    lines.append("")
    lines.append("## Coloured blocks")

    obstacles = world_state["obstacles"]

    if not obstacles:
        lines.append("No coloured blocks detected.")
    else:
        for block in obstacles:
            centre = block["centre_mm"]
            size = block["size_mm"]

            lines.append(
                f"- {block['id']}: {block['colour']} block, "
                f"centre=({centre['x']:.1f}, {centre['y']:.1f}) mm, "
                f"size≈{size['long_side']:.1f}×{size['short_side']:.1f} mm, "
                f"orientation={block['orientation_deg']:.1f}°."
            )

    lines.append("")
    lines.append(
        "Use only this coordinate frame when reasoning about rover motion."
    )

    return "\n".join(lines)


# ============================================================
# NEW TEXT MAP OUTPUT
# ============================================================

def rover_heading_symbol(heading_deg: float) -> str:
    """
    8-direction symbol for the rover in the text map.
    """
    symbols = ["^", "/", ">", "\\", "v", "/", "<", "\\"]
    index = int(((heading_deg % 360.0) + 22.5) // 45.0) % 8
    return symbols[index]


def block_symbol(colour: str) -> str:
    return {
        "red": "R",
        "green": "G",
        "yellow": "Y",
        "blue": "B",
    }.get(colour, "?")


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def world_state_to_text_map(
    world_state: dict,
    cell_size_mm: float = TEXT_MAP_CELL_MM
) -> str:
    """
    Build a simple ASCII occupancy-style map from the current world state.

    Legend:
      ^ > v < / \  = rover heading
      R G Y B      = coloured blocks
      *            = overlap / multiple objects in one cell
      .            = empty
    """

    arena = world_state["arena"]
    width_mm = float(arena["width_mm"])
    height_mm = float(arena["height_mm"])

    cols = max(1, int(math.ceil(width_mm / cell_size_mm)))
    rows = max(1, int(math.ceil(height_mm / cell_size_mm)))

    grid = [
        ["." for _ in range(cols)]
        for _ in range(rows)
    ]

    def world_to_cell(x_mm: float, y_mm: float) -> tuple[int, int]:
        col = clamp_int(
            int(x_mm // cell_size_mm),
            0,
            cols - 1
        )

        row_from_bottom = clamp_int(
            int(y_mm // cell_size_mm),
            0,
            rows - 1
        )

        # display top row first
        row = rows - 1 - row_from_bottom
        return row, col

    def place_symbol(row: int, col: int, symbol: str) -> None:
        current = grid[row][col]

        if current == ".":
            grid[row][col] = symbol
        elif current == symbol:
            pass
        else:
            grid[row][col] = "*"

    # Place blocks first
    for block in world_state["obstacles"]:
        x_mm = float(block["centre_mm"]["x"])
        y_mm = float(block["centre_mm"]["y"])
        row, col = world_to_cell(x_mm, y_mm)
        place_symbol(row, col, block_symbol(block["colour"]))

    # Place rover second so it is represented too.
    rover = world_state["rover"]
    if rover["available"] and rover["pose"] is not None:
        x_mm = float(rover["pose"]["x_mm"])
        y_mm = float(rover["pose"]["y_mm"])
        heading_deg = float(rover["pose"]["heading_deg"])
        row, col = world_to_cell(x_mm, y_mm)
        place_symbol(row, col, rover_heading_symbol(heading_deg))

    lines: list[str] = []
    lines.append("AI LEARNER WORLD STATE - TEXT MAP")
    lines.append(
        f"Arena: {width_mm:.0f} mm x {height_mm:.0f} mm | "
        f"Cell size: {cell_size_mm:.0f} mm"
    )
    lines.append(
        "Origin: bottom-left | +X right | +Y up | "
        "Rover heading 0 deg = up"
    )
    lines.append("")

    # X-axis labels
    header = "y\\x ".ljust(8)
    for col in range(cols):
        x_label = int(col * cell_size_mm)
        header += f"{x_label:>4}"
    lines.append(header)

    for row in range(rows):
        y_bottom = int((rows - 1 - row) * cell_size_mm)
        line = f"{y_bottom:>4} ".ljust(8)
        for col in range(cols):
            line += f"{grid[row][col]:>4}"
        lines.append(line)

    lines.append("")
    lines.append("Legend:")
    lines.append("  R = red block")
    lines.append("  G = green block")
    lines.append("  Y = yellow block")
    lines.append("  B = blue block")
    lines.append("  ^ > v < / \\ = rover heading")
    lines.append("  * = multiple objects in same cell")
    lines.append("  . = empty")
    lines.append("")

    rover = world_state["rover"]
    if rover["available"] and rover["pose"] is not None:
        pose = rover["pose"]
        lines.append(
            "Rover pose: "
            f"({pose['x_mm']:.1f}, {pose['y_mm']:.1f}) mm, "
            f"heading {pose['heading_deg']:.1f} deg, "
            f"confidence {rover['confidence']:.3f}"
        )
    else:
        lines.append("Rover pose unavailable.")

    if world_state["obstacles"]:
        lines.append("Detected blocks:")
        for block in world_state["obstacles"]:
            c = block["centre_mm"]
            lines.append(
                f"  - {block['id']}: "
                f"{block['colour']} at "
                f"({c['x']:.1f}, {c['y']:.1f}) mm"
            )
    else:
        lines.append("Detected blocks: none")

    return "\n".join(lines)


# ============================================================
# VISUALISATION
# ============================================================

def draw_heading_arrow(
    image: np.ndarray,
    centre_px: np.ndarray,
    heading_deg: float,
    length_px: float = 100.0
) -> None:

    theta = math.radians(
        heading_deg
    )

    direction = np.array(
        [
            math.sin(theta),
            -math.cos(theta)
        ],
        dtype=np.float64
    )

    end = centre_px + direction * length_px

    cv2.arrowedLine(
        image,
        tuple(np.round(centre_px).astype(int)),
        tuple(np.round(end).astype(int)),
        (0, 255, 255),
        4,
        tipLength=0.25
    )


def draw_world_state(
    birdseye: np.ndarray,
    blocks: list[dict],
    pose: FilteredPose | None
) -> np.ndarray:

    display = birdseye.copy()

    for block in blocks:
        colour = block["colour"]
        bgr = DISPLAY_COLOURS[colour]

        polygon = np.round(block["box_px"]).astype(np.int32)

        cv2.polylines(
            display,
            [polygon],
            True,
            bgr,
            3
        )

        centre = np.array(
            block["centre_px"],
            dtype=np.float64
        )

        cv2.circle(
            display,
            tuple(np.round(centre).astype(int)),
            6,
            bgr,
            -1
        )

        x_mm, y_mm = block["centre_mm"]

        cv2.putText(
            display,
            f"{block['id']} ({x_mm:.0f},{y_mm:.0f})",
            (int(centre[0]) + 8, int(centre[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            bgr,
            1
        )

    if pose is not None:
        rover_px = world_to_birdseye(
            pose.x_mm,
            pose.y_mm
        )

        cv2.circle(
            display,
            tuple(np.round(rover_px).astype(int)),
            10,
            (255, 255, 0),
            -1
        )

        draw_heading_arrow(
            display,
            rover_px,
            pose.heading_deg
        )

    cv2.rectangle(
        display,
        (0, 0),
        (birdseye_width, 78),
        (20, 20, 20),
        -1
    )

    if pose is None:
        rover_status = "Rover pose unavailable"
        status_colour = (0, 220, 255)
    else:
        rover_status = (
            f"Rover: ({pose.x_mm:.0f}, {pose.y_mm:.0f}) mm   "
            f"h={pose.heading_deg:.1f} deg   "
            f"conf={pose.confidence:.2f}"
        )

        status_colour = (
            (0, 255, 0)
            if (pose.valid and pose.confidence >= 0.75)
            else (0, 220, 255)
        )

    cv2.putText(
        display,
        rover_status,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        status_colour,
        2
    )

    cv2.putText(
        display,
        (
            f"Blocks: {len(blocks)}   "
            f"S=save snapshot   "
            f"M=print Markdown   "
            f"J=print JSON   "
            f"T=print text map   "
            f"Q=quit"
        ),
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1
    )

    return display


# ============================================================
# FILE OUTPUT
# ============================================================

def write_latest(
    annotated: np.ndarray,
    world_state: dict,
    markdown: str,
    text_map: str
) -> None:

    cv2.imwrite(
        str(OUTPUT_DIR / "latest_world_state.png"),
        annotated
    )

    with (OUTPUT_DIR / "latest_world_state.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(world_state, file, indent=4)

    with (OUTPUT_DIR / "latest_world_state.md").open(
        "w",
        encoding="utf-8"
    ) as file:
        file.write(markdown)

    with (OUTPUT_DIR / "latest_world_state_map.txt").open(
        "w",
        encoding="utf-8"
    ) as file:
        file.write(text_map)


def save_snapshot(
    annotated: np.ndarray,
    world_state: dict,
    markdown: str,
    text_map: str
) -> None:

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    image_path = OUTPUT_DIR / f"{timestamp}_world_state.png"
    json_path = OUTPUT_DIR / f"{timestamp}_world_state.json"
    md_path = OUTPUT_DIR / f"{timestamp}_world_state.md"
    txt_path = OUTPUT_DIR / f"{timestamp}_world_state_map.txt"

    cv2.imwrite(str(image_path), annotated)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(world_state, file, indent=4)

    with md_path.open("w", encoding="utf-8") as file:
        file.write(markdown)

    with txt_path.open("w", encoding="utf-8") as file:
        file.write(text_map)

    print()
    print("Saved world state:")
    print(f"  {image_path}")
    print(f"  {json_path}")
    print(f"  {md_path}")
    print(f"  {txt_path}")
    print()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("AI LEARNER - STEP 9 WORLD STATE GENERATION")
    print("==========================================")
    print()
    print("Output:")
    print("  processed bird's-eye image")
    print("  structured JSON world state")
    print("  compact Markdown world state")
    print("  ASCII/text-map world state")
    print()
    print("Controls:")
    print("  S : save timestamped world-state snapshot")
    print("  J : print current JSON")
    print("  M : print current Markdown")
    print("  T : print current text map")
    print("  R : reset rover temporal filter")
    print("  Q : quit")
    print()

    cap = open_camera()
    pose_filter = RoverPoseFilter()

    last_annotated = None
    last_world_state = None
    last_markdown = None
    last_text_map = None

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            cap = reconnect_camera(cap)
            continue

        now = time.monotonic()

        # 1. Camera undistortion
        undistorted = undistort_frame(frame)

        # 2. Bird's-eye arena
        birdseye = cv2.warpPerspective(
            undistorted,
            H_birdseye,
            (birdseye_width, birdseye_height)
        )

        # 3. Blocks
        blocks = detect_blocks(birdseye)

        # 4. Rover
        marker_corners = detect_rover_marker(undistorted)

        raw_pose = None
        if marker_corners is not None:
            raw_pose = make_raw_pose_measurement(
                marker_corners,
                now
            )

        filtered_pose = pose_filter.update(
            raw_pose,
            now
        )

        # 5. World state
        world_state = build_world_state(
            filtered_pose,
            blocks
        )

        markdown = world_state_to_markdown(world_state)
        text_map = world_state_to_text_map(world_state)

        annotated = draw_world_state(
            birdseye,
            blocks,
            filtered_pose
        )

        # 6. Publish latest snapshot to disk
        if AUTO_WRITE_LATEST:
            write_latest(
                annotated,
                world_state,
                markdown,
                text_map
            )

        last_annotated = annotated.copy()
        last_world_state = world_state
        last_markdown = markdown
        last_text_map = text_map

        # 7. Display
        cv2.imshow("AI Learner World State", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        elif key in (ord("r"), ord("R")):
            pose_filter.reset()
            print("\nRover temporal filter reset.")

        elif key in (ord("j"), ord("J")):
            if last_world_state is not None:
                print()
                print(json.dumps(last_world_state, indent=4))

        elif key in (ord("m"), ord("M")):
            if last_markdown is not None:
                print()
                print(last_markdown)
                print()

        elif key in (ord("t"), ord("T")):
            if last_text_map is not None:
                print()
                print(last_text_map)
                print()

        elif key in (ord("s"), ord("S")):
            if (
                last_annotated is not None
                and last_world_state is not None
                and last_markdown is not None
                and last_text_map is not None
            ):
                save_snapshot(
                    last_annotated,
                    last_world_state,
                    last_markdown,
                    last_text_map
                )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
