from __future__ import annotations

import cv2
import json
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
COLOUR_CALIBRATION_FILE = CALIBRATION_DIR / "colour_calibration.json"

CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# USER SETTINGS
# ============================================================

CAMERA_URL = "http://196.47.212.136:8080/video"

# ------------------------------------------------------------
# Component-level filtering
# ------------------------------------------------------------
# Small green noise below this is ignored.
MIN_COMPONENT_AREA_MM2 = 50.0

# Components above this are considered "not marker groups".
# This cleanly removes the large green obstacle card.
MAX_COMPONENT_AREA_MM2 = 20_000.0

# ------------------------------------------------------------
# Marker-peak detection inside a component
# ------------------------------------------------------------
# We now detect MARKER CENTRES, not one-marker-per-contour.
# This allows a merged 3-marker rear blob to split into 3 peaks.

DIST_PEAK_THRESHOLD_REL = 0.38
DIST_PEAK_THRESHOLD_ABS_PX = 2.0

LOCAL_MAX_KERNEL_SIZE = 9
PEAK_CENTROID_MERGE_DISTANCE_PX = 8.0

# Final candidate size checks
MIN_MARKER_DIAMETER_MM = 8.0
MAX_MARKER_DIAMETER_MM = 120.0

# ------------------------------------------------------------
# Mask cleanup
# ------------------------------------------------------------
MARKER_CLOSE_KERNEL_SIZE = 3

# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------
PRINT_EVERY_N_FRAMES = 20
RECONNECT_DELAY_S = 1.0
SHOW_REJECTED_BY_DEFAULT = True


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

H_birdseye = np.array(
    arena_data["homography_image_to_birdseye"],
    dtype=np.float64
)

birdseye_width = int(arena_data["birdseye_width"])
birdseye_height = int(arena_data["birdseye_height"])

arena_width_mm = float(arena_data["arena_width_mm"])
arena_height_mm = float(arena_data["arena_height_mm"])


# ============================================================
# COLOUR CALIBRATION
# ============================================================

green_calibration = colour_data["colours"]["green"]


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
# WORLD SCALE
# ============================================================

MM_PER_PX_X = arena_width_mm / max(1, birdseye_width - 1)
MM_PER_PX_Y = arena_height_mm / max(1, birdseye_height - 1)
MM_PER_PX_MEAN = 0.5 * (MM_PER_PX_X + MM_PER_PX_Y)


def birdseye_to_world(
    x_px: float,
    y_px: float
) -> tuple[float, float]:

    x_mm = float(x_px) * MM_PER_PX_X
    y_mm = arena_height_mm - (
        float(y_px) * MM_PER_PX_Y
    )

    return x_mm, y_mm


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

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]

    if (w, h) != (camera_width, camera_height):
        raise RuntimeError(
            f"Camera returned {w}x{h}, but calibration expects "
            f"{camera_width}x{camera_height}."
        )

    undistorted = cv2.remap(
        frame,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR
    )

    birdseye = cv2.warpPerspective(
        undistorted,
        H_birdseye,
        (birdseye_width, birdseye_height)
    )

    return birdseye


# ============================================================
# GREEN MASK
# ============================================================

def create_green_mask(
    hsv_image: np.ndarray
) -> np.ndarray:

    mask = np.zeros(
        hsv_image.shape[:2],
        dtype=np.uint8
    )

    for hue_low, hue_high in green_calibration["hue_ranges"]:

        lower = np.array(
            [
                hue_low,
                green_calibration["sat_min"],
                green_calibration["val_min"]
            ],
            dtype=np.uint8
        )

        upper = np.array(
            [
                hue_high,
                green_calibration["sat_max"],
                green_calibration["val_max"]
            ],
            dtype=np.uint8
        )

        partial = cv2.inRange(
            hsv_image,
            lower,
            upper
        )

        mask = cv2.bitwise_or(mask, partial)

    return mask


def build_marker_mask(
    raw_green_mask: np.ndarray
) -> np.ndarray:
    """
    Marker-friendly cleanup.

    We avoid strong opening because the front markers may be
    weak, and the rear 3 markers may merge. We want to keep
    marker mass and later split touching markers by distance
    transform.
    """

    mask = cv2.medianBlur(raw_green_mask, 3)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MARKER_CLOSE_KERNEL_SIZE, MARKER_CLOSE_KERNEL_SIZE)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    return mask


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def contour_area_mm2(contour: np.ndarray) -> float:
    area_px2 = float(cv2.contourArea(contour))
    return area_px2 * MM_PER_PX_X * MM_PER_PX_Y


def merge_close_points(
    points: list[tuple[float, float, float]],
    merge_distance_px: float
) -> list[tuple[float, float, float]]:
    """
    Merge peak centroids that are extremely close.
    Each point is (x, y, radius_px).
    """

    merged: list[tuple[float, float, float]] = []

    for point in points:
        px, py, pr = point
        attached = False

        for index, existing in enumerate(merged):
            ex, ey, er = existing
            distance = float(
                np.hypot(px - ex, py - ey)
            )

            if distance <= merge_distance_px:
                # Weighted average by radius.
                w1 = max(1e-6, er)
                w2 = max(1e-6, pr)
                total = w1 + w2

                new_x = (ex * w1 + px * w2) / total
                new_y = (ey * w1 + py * w2) / total
                new_r = max(er, pr)

                merged[index] = (
                    new_x,
                    new_y,
                    new_r
                )

                attached = True
                break

        if not attached:
            merged.append(point)

    return merged


# ============================================================
# SPLIT A CONNECTED GREEN COMPONENT INTO MARKER PEAKS
# ============================================================

def split_component_into_peaks(
    component_mask: np.ndarray
) -> list[tuple[float, float, float]]:
    """
    Returns a list of (x_px, y_px, radius_px) peaks.

    The peaks are extracted from the distance transform:
    - touching circles generate multiple local maxima
    - one isolated circle generates one local maximum

    This is the core fix for the merged 3-marker rear blob.
    """

    if np.count_nonzero(component_mask) == 0:
        return []

    distance = cv2.distanceTransform(
        component_mask,
        cv2.DIST_L2,
        5
    )

    max_distance = float(distance.max())

    if max_distance < DIST_PEAK_THRESHOLD_ABS_PX:
        return []

    threshold = max(
        DIST_PEAK_THRESHOLD_ABS_PX,
        DIST_PEAK_THRESHOLD_REL * max_distance
    )

    # Find local maxima in the distance map.
    kernel = np.ones(
        (LOCAL_MAX_KERNEL_SIZE, LOCAL_MAX_KERNEL_SIZE),
        dtype=np.uint8
    )

    dilated = cv2.dilate(distance, kernel)
    local_max = distance >= (dilated - 1e-6)

    peak_mask = np.logical_and(
        local_max,
        distance >= threshold
    )

    peak_mask_u8 = (
        peak_mask.astype(np.uint8) * 255
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        peak_mask_u8,
        connectivity=8
    )

    raw_points: list[tuple[float, float, float]] = []

    for label in range(1, num_labels):
        region = labels == label

        if not np.any(region):
            continue

        cx = float(centroids[label][0])
        cy = float(centroids[label][1])

        # Estimate radius by the strongest distance-transform value
        # inside this local-peak region.
        radius_px = float(distance[region].max())

        raw_points.append(
            (cx, cy, radius_px)
        )

    return merge_close_points(
        raw_points,
        PEAK_CENTROID_MERGE_DISTANCE_PX
    )


# ============================================================
# MARKER DETECTION
# ============================================================

def detect_marker_candidates(
    birdseye: np.ndarray
) -> tuple[
    list[dict],
    list[dict],
    np.ndarray,
    np.ndarray
]:

    hsv = cv2.cvtColor(
        birdseye,
        cv2.COLOR_BGR2HSV
    )

    raw_green_mask = create_green_mask(hsv)
    marker_mask = build_marker_mask(raw_green_mask)

    contours, _ = cv2.findContours(
        marker_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    accepted: list[dict] = []
    rejected: list[dict] = []

    for contour in contours:
        if len(contour) < 3:
            continue

        area_mm2 = contour_area_mm2(contour)

        if area_mm2 < MIN_COMPONENT_AREA_MM2:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        component_info = {
            "component_area_mm2": float(area_mm2),
            "bounding_rect_px": (
                int(x), int(y), int(w), int(h)
            ),
        }

        # Remove the large green block and any unexpectedly huge group.
        if area_mm2 > MAX_COMPONENT_AREA_MM2:
            rejected.append(
                {
                    "type": "rejected_component",
                    "reason": "component_too_large",
                    **component_info,
                }
            )
            continue

        component_mask = np.zeros(
            marker_mask.shape,
            dtype=np.uint8
        )

        cv2.drawContours(
            component_mask,
            [contour],
            -1,
            255,
            thickness=-1
        )

        peaks = split_component_into_peaks(
            component_mask
        )

        if not peaks:
            rejected.append(
                {
                    "type": "rejected_component",
                    "reason": "no_distance_peaks",
                    **component_info,
                }
            )
            continue

        for peak_index, (cx, cy, radius_px) in enumerate(peaks, start=1):
            diameter_mm = (
                2.0 * radius_px * MM_PER_PX_MEAN
            )

            centre_mm = birdseye_to_world(
                cx, cy
            )

            accepted_flag = (
                MIN_MARKER_DIAMETER_MM
                <= diameter_mm
                <= MAX_MARKER_DIAMETER_MM
            )

            peak_info = {
                "type": "green_marker_candidate",
                "component_area_mm2": float(area_mm2),
                "component_peak_count": len(peaks),
                "component_peak_index": peak_index,

                "centre_px": (
                    float(cx), float(cy)
                ),
                "centre_mm": (
                    float(centre_mm[0]),
                    float(centre_mm[1])
                ),
                "radius_px": float(radius_px),
                "diameter_mm": float(diameter_mm),
                "bounding_rect_px": (
                    int(x), int(y), int(w), int(h)
                ),
            }

            if accepted_flag:
                accepted.append(peak_info)
            else:
                rejected.append(
                    {
                        "type": "rejected_peak",
                        "reason": "diameter",
                        **peak_info,
                    }
                )

    # Stable ordering for consistent numbering:
    # top-to-bottom, then left-to-right.
    accepted.sort(
        key=lambda item: (
            item["centre_px"][1],
            item["centre_px"][0]
        )
    )

    rejected.sort(
        key=lambda item: (
            item["bounding_rect_px"][1],
            item["bounding_rect_px"][0]
        )
    )

    return accepted, rejected, raw_green_mask, marker_mask


# ============================================================
# VISUALISATION
# ============================================================

def draw_candidate(
    image: np.ndarray,
    marker: dict,
    index: int
) -> None:

    cx, cy = marker["centre_px"]
    radius_px = marker["radius_px"]

    centre = (
        int(round(cx)),
        int(round(cy))
    )

    radius = max(
        3,
        int(round(radius_px))
    )

    cv2.circle(
        image,
        centre,
        radius + 4,
        (255, 255, 255),
        2
    )

    cv2.circle(
        image,
        centre,
        radius,
        (0, 255, 0),
        2
    )

    cv2.circle(
        image,
        centre,
        3,
        (0, 255, 0),
        -1
    )

    x_mm, y_mm = marker["centre_mm"]

    label_1 = f"M{index} ({x_mm:.0f},{y_mm:.0f})"
    label_2 = (
        f"D={marker['diameter_mm']:.0f} mm  "
        f"G={marker['component_peak_index']}/"
        f"{marker['component_peak_count']}"
    )

    cv2.putText(
        image,
        label_1,
        (
            centre[0] + radius + 6,
            centre[1] - 4
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1
    )

    cv2.putText(
        image,
        label_2,
        (
            centre[0] + radius + 6,
            centre[1] + 13
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        (180, 255, 180),
        1
    )


def draw_rejected(
    image: np.ndarray,
    item: dict
) -> None:

    x, y, w, h = item["bounding_rect_px"]

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 165, 255),
        1
    )

    reason = item.get("reason", "reject")

    cv2.putText(
        image,
        reason,
        (x, max(12, y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        (0, 165, 255),
        1
    )


def build_mask_view(
    raw_green_mask: np.ndarray,
    marker_mask: np.ndarray
) -> np.ndarray:

    raw_panel = cv2.cvtColor(
        raw_green_mask,
        cv2.COLOR_GRAY2BGR
    )

    marker_panel = cv2.cvtColor(
        marker_mask,
        cv2.COLOR_GRAY2BGR
    )

    cv2.putText(
        raw_panel,
        "RAW GREEN MASK",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        marker_panel,
        "MARKER MASK",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    combined = np.hstack(
        (raw_panel, marker_panel)
    )

    return cv2.resize(
        combined,
        (birdseye_width, birdseye_height),
        interpolation=cv2.INTER_NEAREST
    )


# ============================================================
# JSON OUTPUT
# ============================================================

def build_detection_document(
    accepted: list[dict],
    rejected: list[dict]
) -> dict:

    accepted_output = []

    for index, marker in enumerate(accepted, start=1):
        accepted_output.append(
            {
                "id": f"green_marker_candidate_{index:02d}",
                "centre_mm": {
                    "x": round(marker["centre_mm"][0], 2),
                    "y": round(marker["centre_mm"][1], 2),
                },
                "diameter_mm": round(marker["diameter_mm"], 2),
                "radius_px": round(marker["radius_px"], 2),
                "component_peak_count": marker["component_peak_count"],
                "component_peak_index": marker["component_peak_index"],
            }
        )

    rejected_output = []

    for item in rejected:
        output = {
            "reason": item.get("reason", "reject"),
            "bounding_rect_px": list(item["bounding_rect_px"]),
        }

        if "centre_mm" in item:
            output["centre_mm"] = {
                "x": round(item["centre_mm"][0], 2),
                "y": round(item["centre_mm"][1], 2),
            }

        if "diameter_mm" in item:
            output["diameter_mm"] = round(item["diameter_mm"], 2)

        if "component_area_mm2" in item:
            output["component_area_mm2"] = round(item["component_area_mm2"], 2)

        rejected_output.append(output)

    return {
        "accepted_marker_candidates": accepted_output,
        "rejected_green_components": rejected_output,
    }


# ============================================================
# SAVE SNAPSHOT
# ============================================================

def save_snapshot(
    annotated: np.ndarray,
    raw_green_mask: np.ndarray,
    marker_mask: np.ndarray,
    document: dict
) -> None:

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    annotated_path = CAPTURE_DIR / f"{timestamp}_markers_v2.png"
    raw_mask_path = CAPTURE_DIR / f"{timestamp}_raw_green_mask_v2.png"
    marker_mask_path = CAPTURE_DIR / f"{timestamp}_marker_mask_v2.png"
    json_path = CAPTURE_DIR / f"{timestamp}_markers_v2.json"

    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(raw_mask_path), raw_green_mask)
    cv2.imwrite(str(marker_mask_path), marker_mask)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(document, file, indent=4)

    print()
    print("Saved:")
    print(f"  {annotated_path}")
    print(f"  {raw_mask_path}")
    print(f"  {marker_mask_path}")
    print(f"  {json_path}")
    print()


# ============================================================
# CONSOLE REPORT
# ============================================================

def print_report(
    accepted: list[dict],
    rejected: list[dict]
) -> None:

    print()
    print(f"Accepted marker candidates: {len(accepted)}")

    for index, marker in enumerate(accepted, start=1):
        x_mm, y_mm = marker["centre_mm"]
        print(
            f"  M{index}: "
            f"({x_mm:7.1f}, {y_mm:7.1f}) mm   "
            f"D={marker['diameter_mm']:5.1f} mm   "
            f"group={marker['component_peak_index']}/"
            f"{marker['component_peak_count']}"
        )

    if rejected:
        print(f"Rejected components/peaks: {len(rejected)}")
        for item in rejected:
            print(
                f"  reject reason={item.get('reason','reject')} "
                f"box={item['bounding_rect_px']}"
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print()
    print("AI LEARNER - STEP 5 ROVER MARKER DETECTION (V2)")
    print("===============================================")
    print()
    print(f"Arena: {arena_width_mm:.0f} x {arena_height_mm:.0f} mm")
    print(f"Bird's-eye: {birdseye_width} x {birdseye_height} px")
    print()
    print("This version splits touching marker blobs using a distance transform.")
    print("Controls:")
    print("  Q : quit")
    print("  S : save annotated frame + masks + JSON")
    print("  D : toggle rejected diagnostics")
    print("  M : toggle mask window")
    print()

    cap = open_camera()

    show_rejected = SHOW_REJECTED_BY_DEFAULT
    show_masks = True
    frame_counter = 0

    last_annotated = None
    last_raw_mask = None
    last_marker_mask = None
    last_document = None

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            cap = reconnect_camera(cap)
            continue

        birdseye = preprocess_frame(frame)

        accepted, rejected, raw_green_mask, marker_mask = detect_marker_candidates(
            birdseye
        )

        annotated = birdseye.copy()

        for index, marker in enumerate(accepted, start=1):
            draw_candidate(annotated, marker, index)

        if show_rejected:
            for item in rejected:
                draw_rejected(annotated, item)

        status = (
            f"Marker candidates: {len(accepted)}   "
            f"Rejected green components: {len(rejected)}"
        )

        cv2.rectangle(
            annotated,
            (0, 0),
            (birdseye_width, 45),
            (20, 20, 20),
            -1
        )

        status_colour = (
            (0, 255, 0) if len(accepted) == 5 else (0, 220, 255)
        )

        cv2.putText(
            annotated,
            status,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            status_colour,
            2
        )

        document = build_detection_document(accepted, rejected)

        last_annotated = annotated
        last_raw_mask = raw_green_mask
        last_marker_mask = marker_mask
        last_document = document

        cv2.imshow("Rover Marker Detection", annotated)

        if show_masks:
            cv2.imshow(
                "Green Marker Masks",
                build_mask_view(raw_green_mask, marker_mask)
            )

        frame_counter += 1

        if frame_counter % PRINT_EVERY_N_FRAMES == 0:
            print_report(accepted, rejected)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        elif key in (ord("d"), ord("D")):
            show_rejected = not show_rejected

        elif key in (ord("m"), ord("M")):
            show_masks = not show_masks

            if not show_masks:
                try:
                    cv2.destroyWindow("Green Marker Masks")
                except cv2.error:
                    pass

        elif key in (ord("s"), ord("S")):
            if (
                last_annotated is not None
                and last_raw_mask is not None
                and last_marker_mask is not None
                and last_document is not None
            ):
                save_snapshot(
                    last_annotated,
                    last_raw_mask,
                    last_marker_mask,
                    last_document
                )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
