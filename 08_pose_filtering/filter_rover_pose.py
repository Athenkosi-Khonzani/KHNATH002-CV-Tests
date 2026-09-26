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
CAPTURE_DIR = SCRIPT_DIR / "captures"

CAMERA_CALIBRATION_FILE = CALIBRATION_DIR / "camera_calibration.json"
ARENA_CALIBRATION_FILE = CALIBRATION_DIR / "arena_calibration.json"

CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# USER SETTINGS
# ============================================================

CAMERA_URL = "http://196.24.154.126:8080/video"

ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
ROVER_MARKER_ID = 29

# Step 6: heading validation showed the fixed correction was
# negligible, so keep this at zero.
MARKER_HEADING_OFFSET_DEG = 0.0

# Step 7: body-frame compensation from ArUco centre to rover centre.
ROVER_CENTRE_LEFT_MM = 64.0
ROVER_CENTRE_FORWARD_MM = 90.0


# ============================================================
# TEMPORAL FILTER SETTINGS
# ============================================================

# EMA time constants.
# Smaller = more responsive, larger = smoother.
POSITION_FILTER_TAU_S = 0.12
HEADING_FILTER_TAU_S = 0.08

# Generous physical outlier gates.
# These are intentionally much larger than expected rover motion.
MAX_TRANSLATIONAL_SPEED_MM_S = 2500.0
MAX_ANGULAR_SPEED_DEG_S = 900.0

# Allow a little extra measurement movement even at tiny dt.
POSITION_GATE_MARGIN_MM = 30.0
HEADING_GATE_MARGIN_DEG = 8.0

# How long the last good pose may be retained during a short
# marker dropout. After this, the pose becomes invalid.
POSE_HOLD_SECONDS = 0.50

# Recent accepted/missed detections used for operational confidence.
CONFIDENCE_WINDOW_FRAMES = 20

# Marker pixel-size quality.
# Detection can work below FULL_CONF_MARKER_SIDE_PX, but a larger
# marker generally gives more reliable corner localisation.
MIN_USABLE_MARKER_SIDE_PX = 20.0
FULL_CONF_MARKER_SIDE_PX = 60.0

RECONNECT_DELAY_S = 1.0


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
# MATH HELPERS
# ============================================================

def shortest_angle_delta_deg(
    target_deg: float,
    current_deg: float
) -> float:
    """
    Signed shortest rotation from current -> target, in [-180, 180).
    """
    return float(
        ((target_deg - current_deg + 180.0) % 360.0) - 180.0
    )


def ema_alpha(dt_s: float, tau_s: float) -> float:
    """
    Time-aware EMA coefficient:
        alpha = 1 - exp(-dt/tau)
    """
    if tau_s <= 0:
        return 1.0

    dt_s = max(0.0, dt_s)
    return float(
        1.0 - math.exp(-dt_s / tau_s)
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
# CAMERA
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
            f"Camera returned {w}x{h}, but calibration expects "
            f"{camera_width}x{camera_height}."
        )

    return cv2.remap(
        frame,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# ARUCO -> RAW COMPENSATED ROVER MEASUREMENT
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

    # Heading is clockwise from +Y.
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


def make_raw_measurement(
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

    heading = calculate_heading_deg(
        corners_world
    )

    rover_x, rover_y = marker_to_rover_centre(
        float(marker_centre_world[0]),
        float(marker_centre_world[1]),
        heading
    )

    return PoseMeasurement(
        x_mm=rover_x,
        y_mm=rover_y,
        heading_deg=heading,
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

        self.last_measurement: PoseMeasurement | None = None

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
                side_px
                - MIN_USABLE_MARKER_SIDE_PX
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
            measurement.timestamp
            - self.last_filter_time
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

        marker_detected = (
            measurement is not None
        )

        measurement_accepted = False

        if measurement is not None:

            size_score = self.marker_size_score(
                measurement.marker_side_px
            )

            if (
                size_score > 0.0
                and self.is_plausible(measurement)
            ):

                measurement_accepted = True
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
                        * (
                            measurement.x_mm
                            - self.x_mm
                        )
                    )

                    self.y_mm += (
                        alpha_position
                        * (
                            measurement.y_mm
                            - self.y_mm
                        )
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
                self.last_measurement = measurement

        self.recent_acceptance.append(
            1 if measurement_accepted else 0
        )

        self.last_filter_time = now

        if not self.initialised:
            return None

        if self.last_good_time is None:
            return None

        age_s = max(
            0.0,
            now - self.last_good_time
        )

        valid = (
            age_s <= POSE_HOLD_SECONDS
        )

        stale = (
            age_s > 0.05
        )

        hit_ratio = (
            sum(self.recent_acceptance)
            / max(1, len(self.recent_acceptance))
        )

        # Operational confidence:
        #   65% recent accepted-detection ratio
        #   35% marker pixel-size quality
        base_confidence = (
            0.65 * hit_ratio
            + 0.35 * self.last_size_score
        )

        # During a short dropout, confidence decays smoothly.
        if age_s > 0.0:
            decay = math.exp(
                -3.0
                * age_s
                / max(POSE_HOLD_SECONDS, 1e-6)
            )
        else:
            decay = 1.0

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
            measurement_accepted=measurement_accepted
        )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def world_to_birdseye(
    x_mm: float,
    y_mm: float
) -> np.ndarray:

    return np.array(
        [
            x_mm
            / arena_width_mm
            * (birdseye_width - 1),

            (
                arena_height_mm
                - y_mm
            )
            / arena_height_mm
            * (birdseye_height - 1)
        ],
        dtype=np.float64
    )


def draw_heading_arrow(
    image: np.ndarray,
    centre_px: np.ndarray,
    heading_deg: float,
    colour: tuple[int, int, int],
    length_px: float,
    thickness: int
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

    end = (
        centre_px
        + direction * length_px
    )

    cv2.arrowedLine(
        image,
        tuple(np.round(centre_px).astype(int)),
        tuple(np.round(end).astype(int)),
        colour,
        thickness,
        tipLength=0.25
    )


def draw_display(
    birdseye: np.ndarray,
    raw: PoseMeasurement | None,
    filtered: FilteredPose | None,
    corners_px: np.ndarray | None
) -> np.ndarray:

    display = birdseye.copy()

    # Draw detected ArUco quadrilateral.
    if corners_px is not None:

        corners_bird = transform_points(
            corners_px,
            H_birdseye
        )

        cv2.polylines(
            display,
            [
                np.round(
                    corners_bird
                ).astype(np.int32)
            ],
            True,
            (0, 255, 0),
            2
        )

    # Raw compensated pose: magenta.
    if raw is not None:

        raw_point = world_to_birdseye(
            raw.x_mm,
            raw.y_mm
        )

        cv2.circle(
            display,
            tuple(np.round(raw_point).astype(int)),
            7,
            (255, 0, 255),
            -1
        )

        draw_heading_arrow(
            display,
            raw_point,
            raw.heading_deg,
            (255, 0, 255),
            75.0,
            2
        )

    # Filtered pose: cyan centre + yellow heading.
    if filtered is not None:

        filtered_point = world_to_birdseye(
            filtered.x_mm,
            filtered.y_mm
        )

        cv2.circle(
            display,
            tuple(np.round(filtered_point).astype(int)),
            9,
            (255, 255, 0),
            -1
        )

        draw_heading_arrow(
            display,
            filtered_point,
            filtered.heading_deg,
            (0, 255, 255),
            100.0,
            4
        )

    # Status panel.
    cv2.rectangle(
        display,
        (0, 0),
        (birdseye_width, 145),
        (20, 20, 20),
        -1
    )

    if raw is None:
        raw_text = "RAW: marker not detected"
    else:
        raw_text = (
            f"RAW: x={raw.x_mm:.1f}  "
            f"y={raw.y_mm:.1f}  "
            f"h={raw.heading_deg:.1f} deg  "
            f"marker={raw.marker_side_px:.0f}px"
        )

    cv2.putText(
        display,
        raw_text,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 0, 255),
        1
    )

    if filtered is None:

        filtered_text = (
            "FILTERED: waiting for first valid pose"
        )

        filtered_colour = (
            0, 220, 255
        )

        confidence_text = (
            "confidence=0.00  valid=False"
        )

    else:

        filtered_text = (
            f"FILTERED: x={filtered.x_mm:.1f}  "
            f"y={filtered.y_mm:.1f}  "
            f"h={filtered.heading_deg:.1f} deg"
        )

        if filtered.valid and filtered.confidence >= 0.75:
            filtered_colour = (0, 255, 0)
        elif filtered.valid:
            filtered_colour = (0, 220, 255)
        else:
            filtered_colour = (0, 0, 255)

        confidence_text = (
            f"confidence={filtered.confidence:.2f}  "
            f"valid={filtered.valid}  "
            f"stale={filtered.stale}  "
            f"age={filtered.age_s:.2f}s  "
            f"accepted={filtered.measurement_accepted}"
        )

    cv2.putText(
        display,
        filtered_text,
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        filtered_colour,
        2
    )

    cv2.putText(
        display,
        confidence_text,
        (15, 87),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        filtered_colour,
        1
    )

    cv2.putText(
        display,
        (
            "MAGENTA=raw pose   CYAN/YELLOW=filtered pose"
        ),
        (15, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1
    )

    cv2.putText(
        display,
        "R=reset filter   S=save snapshot   Q=quit",
        (15, 137),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (180, 180, 180),
        1
    )

    return display


# ============================================================
# OUTPUT JSON
# ============================================================

def build_pose_document(
    raw: PoseMeasurement | None,
    filtered: FilteredPose | None
) -> dict:

    raw_document = None

    if raw is not None:
        raw_document = {
            "x_mm": round(raw.x_mm, 2),
            "y_mm": round(raw.y_mm, 2),
            "heading_deg": round(
                raw.heading_deg,
                2
            ),
            "marker_side_px": round(
                raw.marker_side_px,
                2
            ),
        }

    filtered_document = None

    if filtered is not None:
        filtered_document = {
            "x_mm": round(filtered.x_mm, 2),
            "y_mm": round(filtered.y_mm, 2),
            "heading_deg": round(
                filtered.heading_deg,
                2
            ),
            "confidence": round(
                filtered.confidence,
                3
            ),
            "valid": filtered.valid,
            "stale": filtered.stale,
            "age_s": round(
                filtered.age_s,
                3
            ),
            "marker_detected":
                filtered.marker_detected,
            "measurement_accepted":
                filtered.measurement_accepted,
        }

    return {
        "marker": {
            "dictionary": "DICT_4X4_50",
            "id": ROVER_MARKER_ID,
        },

        "body_offset_mm": {
            "left": ROVER_CENTRE_LEFT_MM,
            "forward": ROVER_CENTRE_FORWARD_MM,
        },

        "raw_pose": raw_document,

        "filtered_pose": filtered_document,
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
        / f"{timestamp}_filtered_pose.png"
    )

    json_path = (
        CAPTURE_DIR
        / f"{timestamp}_filtered_pose.json"
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("AI LEARNER - STEP 8 POSE FILTERING + CONFIDENCE")
    print("================================================")
    print()
    print("ArUco       : DICT_4X4_50 / ID 29")
    print(
        f"Body offset : {ROVER_CENTRE_LEFT_MM:.0f} mm LEFT, "
        f"{ROVER_CENTRE_FORWARD_MM:.0f} mm FORWARD"
    )
    print()
    print(
        f"Pose hold timeout : {POSE_HOLD_SECONDS:.2f} s"
    )
    print()
    print("Controls:")
    print("  R : reset temporal filter")
    print("  S : save current raw + filtered pose")
    print("  Q : quit")
    print()

    cap = open_camera()
    pose_filter = RoverPoseFilter()

    last_display = None
    last_document = None

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            cap = reconnect_camera(
                cap
            )
            continue

        now = time.monotonic()

        undistorted = undistort_frame(
            frame
        )

        corners = detect_rover_marker(
            undistorted
        )

        raw_measurement = None

        if corners is not None:
            raw_measurement = make_raw_measurement(
                corners,
                now
            )

        filtered_pose = pose_filter.update(
            raw_measurement,
            now
        )

        birdseye = cv2.warpPerspective(
            undistorted,
            H_birdseye,
            (
                birdseye_width,
                birdseye_height
            )
        )

        display = draw_display(
            birdseye,
            raw_measurement,
            filtered_pose,
            corners
        )

        document = build_pose_document(
            raw_measurement,
            filtered_pose
        )

        last_display = display.copy()
        last_document = document

        cv2.imshow(
            "Filtered Rover Pose",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            ord("Q")
        ):
            break

        elif key in (
            ord("r"),
            ord("R")
        ):
            pose_filter.reset()
            print("\nPose filter reset.")

        elif key in (
            ord("s"),
            ord("S")
        ):
            if (
                last_display is not None
                and last_document is not None
            ):
                save_snapshot(
                    last_display,
                    last_document
                )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
