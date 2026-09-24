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
COLOUR_CALIBRATION_FILE = CALIBRATION_DIR / "colour_calibration.json"

CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_URL = "http://196.47.212.136:8080/video"

MIN_BLOCK_AREA_MM2 = 10_000.0
MIN_BLOCK_RECTANGULARITY = 0.70
MIN_BLOCK_SIDE_MM = 40.0

MIN_MARKER_AREA_MM2 = 150.0
MAX_MARKER_AREA_MM2 = 6_000.0
MIN_MARKER_CIRCULARITY = 0.55

MORPH_KERNEL_SIZE = 5
PRINT_EVERY_N_FRAMES = 30
RECONNECT_DELAY_S = 1.0

DISPLAY_COLOURS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
}

COLOUR_ORDER = ["red", "green", "yellow", "blue"]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


camera_data = load_json(CAMERA_CALIBRATION_FILE)
arena_data = load_json(ARENA_CALIBRATION_FILE)
colour_data = load_json(COLOUR_CALIBRATION_FILE)

camera_matrix = np.array(camera_data["camera_matrix"], dtype=np.float64)
dist_coeffs = np.array(camera_data["distortion_coefficients"], dtype=np.float64)
camera_width = int(camera_data["image_width"])
camera_height = int(camera_data["image_height"])

H_birdseye = np.array(
    arena_data["homography_image_to_birdseye"],
    dtype=np.float64
)

birdseye_width = int(arena_data["birdseye_width"])
birdseye_height = int(arena_data["birdseye_height"])
arena_width_mm = float(arena_data["arena_width_mm"])
arena_height_mm = float(arena_data["arena_height_mm"])

colour_calibrations = colour_data["colours"]

map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix,
    dist_coeffs,
    None,
    camera_matrix,
    (camera_width, camera_height),
    cv2.CV_32FC1
)


def birdseye_to_world(x_px: float, y_px: float) -> tuple[float, float]:
    x_mm = float(x_px) / max(1, birdseye_width - 1) * arena_width_mm
    y_mm = arena_height_mm - (
        float(y_px) / max(1, birdseye_height - 1) * arena_height_mm
    )
    return x_mm, y_mm


def box_pixels_to_world(box_px: np.ndarray) -> np.ndarray:
    return np.array(
        [birdseye_to_world(x, y) for x, y in box_px],
        dtype=np.float64
    )


def distance_mm(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p2 - p1))


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


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]

    if (width, height) != (camera_width, camera_height):
        raise RuntimeError(
            f"Camera returned {width}x{height}, "
            f"but calibration expects {camera_width}x{camera_height}."
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


def create_mask(hsv_image: np.ndarray, calibration: dict) -> np.ndarray:
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    for hue_low, hue_high in calibration["hue_ranges"]:
        lower = np.array(
            [hue_low, calibration["sat_min"], calibration["val_min"]],
            dtype=np.uint8
        )
        upper = np.array(
            [hue_high, calibration["sat_max"], calibration["val_max"]],
            dtype=np.uint8
        )

        partial = cv2.inRange(hsv_image, lower, upper)
        mask = cv2.bitwise_or(mask, partial)

    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def contour_area_mm2(contour: np.ndarray) -> float:
    area_px2 = float(cv2.contourArea(contour))

    mm_per_px_x = arena_width_mm / max(1, birdseye_width - 1)
    mm_per_px_y = arena_height_mm / max(1, birdseye_height - 1)

    return area_px2 * mm_per_px_x * mm_per_px_y


def contour_circularity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    if perimeter <= 0:
        return 0.0

    return float(4.0 * math.pi * area / (perimeter * perimeter))


def analyse_rectangle(contour: np.ndarray) -> dict | None:
    rect = cv2.minAreaRect(contour)
    centre_px = rect[0]

    box_px = cv2.boxPoints(rect)
    box_px = np.asarray(box_px, dtype=np.float64)
    box_world = box_pixels_to_world(box_px)

    edge_01 = distance_mm(box_world[0], box_world[1])
    edge_12 = distance_mm(box_world[1], box_world[2])

    width_mm = max(edge_01, edge_12)
    height_mm = min(edge_01, edge_12)

    if width_mm <= 0 or height_mm <= 0:
        return None

    rectangle_area_mm2 = width_mm * height_mm
    contour_mm2 = contour_area_mm2(contour)

    rectangularity = (
        contour_mm2 / rectangle_area_mm2
        if rectangle_area_mm2 > 0
        else 0.0
    )

    centre_world = birdseye_to_world(centre_px[0], centre_px[1])

    if edge_01 >= edge_12:
        p_start = box_world[0]
        p_end = box_world[1]
    else:
        p_start = box_world[1]
        p_end = box_world[2]

    vector = p_end - p_start
    angle_deg = math.degrees(math.atan2(vector[1], vector[0])) % 180.0

    return {
        "centre_px": (float(centre_px[0]), float(centre_px[1])),
        "centre_mm": (float(centre_world[0]), float(centre_world[1])),
        "box_px": box_px,
        "box_world_mm": box_world,
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "area_mm2": float(contour_mm2),
        "rectangularity": float(rectangularity),
        "orientation_deg": float(angle_deg),
    }


def is_block(rect_info: dict) -> bool:
    return (
        rect_info["area_mm2"] >= MIN_BLOCK_AREA_MM2
        and rect_info["rectangularity"] >= MIN_BLOCK_RECTANGULARITY
        and rect_info["width_mm"] >= MIN_BLOCK_SIDE_MM
        and rect_info["height_mm"] >= MIN_BLOCK_SIDE_MM
    )


def is_green_marker_candidate(contour: np.ndarray) -> tuple[bool, dict]:
    area_mm2 = contour_area_mm2(contour)
    circularity = contour_circularity(contour)

    valid = (
        MIN_MARKER_AREA_MM2 <= area_mm2 <= MAX_MARKER_AREA_MM2
        and circularity >= MIN_MARKER_CIRCULARITY
    )

    moments = cv2.moments(contour)

    centre_px = None
    centre_mm = None

    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        centre_px = (float(cx), float(cy))
        centre_mm = birdseye_to_world(cx, cy)

    info = {
        "area_mm2": float(area_mm2),
        "circularity": float(circularity),
        "centre_px": centre_px,
        "centre_mm": centre_mm,
    }

    return valid, info


def detect_objects(
    birdseye: np.ndarray
) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:

    hsv = cv2.cvtColor(birdseye, cv2.COLOR_BGR2HSV)

    blocks = []
    marker_candidates = []
    masks = {}

    for colour in COLOUR_ORDER:
        calibration = colour_calibrations[colour]

        mask = create_mask(hsv, calibration)
        mask = clean_mask(mask)
        masks[colour] = mask

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < 3:
                continue

            rect_info = analyse_rectangle(contour)

            if rect_info is None:
                continue

            if is_block(rect_info):
                blocks.append(
                    {
                        "type": "block",
                        "colour": colour,
                        **rect_info
                    }
                )
                continue

            if colour == "green":
                valid, marker_info = is_green_marker_candidate(contour)

                if valid and marker_info["centre_px"] is not None:
                    marker_candidates.append(
                        {
                            "type": "green_marker_candidate",
                            **marker_info
                        }
                    )

    blocks.sort(
        key=lambda item: (
            item["colour"],
            item["centre_mm"][0],
            item["centre_mm"][1]
        )
    )

    marker_candidates.sort(
        key=lambda item: (
            item["centre_mm"][0],
            item["centre_mm"][1]
        )
    )

    return blocks, marker_candidates, masks


def draw_block(
    image: np.ndarray,
    block: dict,
    index: int
) -> None:

    colour = block["colour"]
    bgr = DISPLAY_COLOURS[colour]

    box = np.round(block["box_px"]).astype(np.int32)

    cv2.polylines(
        image,
        [box],
        True,
        bgr,
        3
    )

    cx, cy = block["centre_px"]
    centre_int = (int(round(cx)), int(round(cy)))

    cv2.circle(
        image,
        centre_int,
        6,
        bgr,
        -1
    )

    x_mm, y_mm = block["centre_mm"]

    label_1 = f"{colour.upper()} BLOCK {index}"
    label_2 = f"({x_mm:.0f}, {y_mm:.0f}) mm"
    label_3 = (
        f"{block['width_mm']:.0f} x "
        f"{block['height_mm']:.0f} mm  "
        f"{block['orientation_deg']:.1f} deg"
    )

    x_text = centre_int[0] + 12
    y_text = centre_int[1] - 32

    cv2.putText(
        image,
        label_1,
        (x_text, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        bgr,
        2
    )

    cv2.putText(
        image,
        label_2,
        (x_text, y_text + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        bgr,
        2
    )

    cv2.putText(
        image,
        label_3,
        (x_text, y_text + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        bgr,
        1
    )


def draw_marker_candidate(
    image: np.ndarray,
    marker: dict,
    index: int
) -> None:

    cx, cy = marker["centre_px"]
    centre = (int(round(cx)), int(round(cy)))

    cv2.circle(
        image,
        centre,
        9,
        (255, 255, 255),
        2
    )

    cv2.circle(
        image,
        centre,
        4,
        DISPLAY_COLOURS["green"],
        -1
    )

    x_mm, y_mm = marker["centre_mm"]

    text = f"M{index} ({x_mm:.0f},{y_mm:.0f})"

    cv2.putText(
        image,
        text,
        (centre[0] + 10, centre[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1
    )


def build_mask_grid(masks: dict[str, np.ndarray]) -> np.ndarray:
    panels = []

    for colour in COLOUR_ORDER:
        panel = cv2.cvtColor(
            masks[colour],
            cv2.COLOR_GRAY2BGR
        )

        cv2.putText(
            panel,
            colour.upper(),
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            DISPLAY_COLOURS[colour],
            2
        )

        panels.append(panel)

    top = np.hstack((panels[0], panels[1]))
    bottom = np.hstack((panels[2], panels[3]))
    grid = np.vstack((top, bottom))

    return cv2.resize(
        grid,
        (birdseye_width, birdseye_height),
        interpolation=cv2.INTER_NEAREST
    )


def build_detection_document(
    blocks: list[dict],
    markers: list[dict]
) -> dict:

    colour_running_counts = {
        colour: 0
        for colour in COLOUR_ORDER
    }

    block_output = []

    for block in blocks:
        colour = block["colour"]
        colour_running_counts[colour] += 1

        block_output.append(
            {
                "id": (
                    f"{colour}_block_"
                    f"{colour_running_counts[colour]:02d}"
                ),
                "type": "block",
                "colour": colour,
                "centre_mm": {
                    "x": round(block["centre_mm"][0], 2),
                    "y": round(block["centre_mm"][1], 2)
                },
                "width_mm": round(block["width_mm"], 2),
                "height_mm": round(block["height_mm"], 2),
                "orientation_deg": round(block["orientation_deg"], 2),
                "area_mm2": round(block["area_mm2"], 2),
                "rectangularity": round(block["rectangularity"], 3),
            }
        )

    marker_output = []

    for index, marker in enumerate(markers, start=1):
        marker_output.append(
            {
                "id": f"green_marker_candidate_{index:02d}",
                "type": "green_marker_candidate",
                "centre_mm": {
                    "x": round(marker["centre_mm"][0], 2),
                    "y": round(marker["centre_mm"][1], 2)
                },
                "area_mm2": round(marker["area_mm2"], 2),
                "circularity": round(marker["circularity"], 3),
            }
        )

    return {
        "arena": {
            "width_mm": arena_width_mm,
            "height_mm": arena_height_mm,
        },
        "blocks": block_output,
        "green_marker_candidates": marker_output,
    }


def save_snapshot(
    annotated: np.ndarray,
    birdseye: np.ndarray,
    detection_document: dict
) -> None:

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    annotated_path = CAPTURE_DIR / f"{timestamp}_annotated.png"
    raw_path = CAPTURE_DIR / f"{timestamp}_birdseye.png"
    json_path = CAPTURE_DIR / f"{timestamp}_detections.json"

    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(raw_path), birdseye)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            detection_document,
            file,
            indent=4
        )

    print()
    print("Saved test capture:")
    print(f"  {annotated_path}")
    print(f"  {raw_path}")
    print(f"  {json_path}")
    print()


def print_summary(
    blocks: list[dict],
    markers: list[dict]
) -> None:

    print()
    print(
        f"Detected {len(blocks)} block(s), "
        f"{len(markers)} green marker candidate(s)"
    )

    for block in blocks:
        x_mm, y_mm = block["centre_mm"]

        print(
            f"  {block['colour'].upper():6s} "
            f"centre=({x_mm:7.1f}, {y_mm:7.1f}) mm   "
            f"size={block['width_mm']:6.1f}x"
            f"{block['height_mm']:6.1f} mm   "
            f"rect={block['rectangularity']:.2f}"
        )

    for index, marker in enumerate(markers, start=1):
        x_mm, y_mm = marker["centre_mm"]

        print(
            f"  MARKER {index}: "
            f"({x_mm:.1f}, {y_mm:.1f}) mm   "
            f"area={marker['area_mm2']:.0f} mm^2   "
            f"circ={marker['circularity']:.2f}"
        )


def main() -> None:
    print()
    print("AI LEARNER - BLOCK DETECTION TEST")
    print("=================================")
    print()
    print(
        f"Arena: {arena_width_mm:.0f} x "
        f"{arena_height_mm:.0f} mm"
    )
    print(
        f"Bird's-eye: {birdseye_width} x "
        f"{birdseye_height} px"
    )
    print()
    print("Controls:")
    print("  Q : quit")
    print("  S : save current annotated frame + JSON")
    print("  M : toggle mask window")
    print()

    cap = open_camera()

    frame_counter = 0
    show_masks = True

    last_annotated = None
    last_birdseye = None
    last_document = None

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            cap = reconnect_camera(cap)
            continue

        birdseye = preprocess_frame(frame)

        blocks, markers, masks = detect_objects(birdseye)

        annotated = birdseye.copy()

        colour_counts = {
            colour: 0
            for colour in COLOUR_ORDER
        }

        for block in blocks:
            colour_counts[block["colour"]] += 1

            draw_block(
                annotated,
                block,
                colour_counts[block["colour"]]
            )

        for index, marker in enumerate(markers, start=1):
            draw_marker_candidate(
                annotated,
                marker,
                index
            )

        status = (
            f"Blocks: {len(blocks)}   "
            f"Green marker candidates: {len(markers)}"
        )

        cv2.rectangle(
            annotated,
            (0, 0),
            (birdseye_width, 45),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            annotated,
            status,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (230, 230, 230),
            2
        )

        detection_document = build_detection_document(
            blocks,
            markers
        )

        last_annotated = annotated
        last_birdseye = birdseye
        last_document = detection_document

        cv2.imshow(
            "Block Detection",
            annotated
        )

        if show_masks:
            cv2.imshow(
                "Detection Masks",
                build_mask_grid(masks)
            )

        frame_counter += 1

        if frame_counter % PRINT_EVERY_N_FRAMES == 0:
            print_summary(
                blocks,
                markers
            )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        elif key in (ord("m"), ord("M")):
            show_masks = not show_masks

            if not show_masks:
                try:
                    cv2.destroyWindow(
                        "Detection Masks"
                    )
                except cv2.error:
                    pass

        elif key in (ord("s"), ord("S")):
            if (
                last_annotated is not None
                and last_birdseye is not None
                and last_document is not None
            ):
                save_snapshot(
                    last_annotated,
                    last_birdseye,
                    last_document
                )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
