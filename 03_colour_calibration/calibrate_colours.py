import cv2
import json
import math
import time
import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

# Replace with the actual IP/MJPEG camera stream URL.
CAMERA_URL = "http://196.47.212.136:8080/video"

CAMERA_CALIBRATION_FILE = "camera_calibration.json"
ARENA_CALIBRATION_FILE = "arena_calibration.json"
COLOUR_CALIBRATION_FILE = "colour_calibration.json"

# Each click samples a square patch around the selected point.
# PATCH_RADIUS = 10 gives a 21 x 21 px patch.
PATCH_RADIUS = 10

# Minimum number of independent sample clicks required per colour.
MIN_SAMPLES_PER_COLOUR = 5

# Hue tolerance. OpenCV hue is 0..179.
HUE_MARGIN = 3

# Prevent an unrealistically narrow learned hue range.
MIN_HUE_HALF_WIDTH = 8

# Extra tolerance for saturation and value.
SATURATION_MARGIN = 40
VALUE_MARGIN = 40

# Pixels with very low saturation do not carry reliable hue
# information and are ignored when estimating colour.
MIN_COLOUR_SATURATION = 40

COLOURS = ["red", "green", "yellow", "blue"]

# Display colours in BGR.
DISPLAY_COLOURS = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
}


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

with open(CAMERA_CALIBRATION_FILE, "r") as file:
    camera_data = json.load(file)

camera_matrix = np.array(camera_data["camera_matrix"], dtype=np.float64)
dist_coeffs = np.array(camera_data["distortion_coefficients"], dtype=np.float64)
camera_width = int(camera_data["image_width"])
camera_height = int(camera_data["image_height"])


# ============================================================
# LOAD ARENA CALIBRATION
# ============================================================

with open(ARENA_CALIBRATION_FILE, "r") as file:
    arena_data = json.load(file)

H_birdseye = np.array(
    arena_data["homography_image_to_birdseye"],
    dtype=np.float64,
)

birdseye_width = int(arena_data["birdseye_width"])
birdseye_height = int(arena_data["birdseye_height"])

print(f"Camera calibration loaded: {camera_width} x {camera_height}")
print(f"Bird's-eye output: {birdseye_width} x {birdseye_height}")


# ============================================================
# PRECOMPUTE UNDISTORTION
# ============================================================

map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix,
    dist_coeffs,
    None,
    camera_matrix,
    (camera_width, camera_height),
    cv2.CV_32FC1,
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_frame(frame):
    """Raw frame -> undistortion -> bird's-eye arena."""
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
        interpolation=cv2.INTER_LINEAR,
    )

    birdseye = cv2.warpPerspective(
        undistorted,
        H_birdseye,
        (birdseye_width, birdseye_height),
    )

    return birdseye


# ============================================================
# CIRCULAR HUE HELPERS
# ============================================================

def circular_hue_difference(hue, centre):
    """Signed shortest distance on OpenCV hue circle [0,179]."""
    return ((hue - centre + 90.0) % 180.0) - 90.0


def build_wrapped_hue_ranges(centre, half_width):
    """Return one or two [low, high] hue ranges, handling wraparound."""
    low = centre - half_width
    high = centre + half_width
    ranges = []

    if low < 0:
        ranges.append([0, int(min(179, math.ceil(high)))])
        ranges.append([int(max(0, math.floor(180 + low))), 179])

    elif high > 179:
        ranges.append([int(max(0, math.floor(low))), 179])
        ranges.append([0, int(min(179, math.ceil(high - 180)))])

    else:
        ranges.append(
            [
                int(max(0, math.floor(low))),
                int(min(179, math.ceil(high))),
            ]
        )

    return ranges


# ============================================================
# CALCULATE CALIBRATION FOR ONE COLOUR
# ============================================================

def calculate_colour_calibration(colour_name, sample_patches):
    """Build robust HSV thresholds from multiple sampled patches."""
    pixels = np.concatenate(sample_patches, axis=0).astype(np.float64)

    hue = pixels[:, 0]
    saturation = pixels[:, 1]
    value = pixels[:, 2]

    # Ignore white/grey pixels for hue estimation.
    valid_colour_mask = saturation >= MIN_COLOUR_SATURATION

    hue_valid = hue[valid_colour_mask]
    saturation_valid = saturation[valid_colour_mask]
    value_valid = value[valid_colour_mask]

    if len(hue_valid) == 0:
        raise RuntimeError(
            f"No valid coloured pixels found for {colour_name}. "
            "Re-sample farther from the white border."
        )

    # Circular mean hue.
    angles = hue_valid * 2.0 * math.pi / 180.0
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))
    centre_angle = math.atan2(mean_sin, mean_cos)

    if centre_angle < 0:
        centre_angle += 2.0 * math.pi

    centre_hue = centre_angle * 180.0 / (2.0 * math.pi)

    # Measured hue spread.
    differences = circular_hue_difference(hue_valid, centre_hue)
    low_difference, high_difference = np.percentile(differences, [1, 99])

    measured_half_width = max(abs(low_difference), abs(high_difference))
    half_width = max(
        measured_half_width + HUE_MARGIN,
        MIN_HUE_HALF_WIDTH,
    )

    hue_ranges = build_wrapped_hue_ranges(centre_hue, half_width)

    # Lower S/V limits. Upper limits stay at 255 for robustness.
    sat_low = int(
        max(
            0,
            np.percentile(saturation_valid, 1) - SATURATION_MARGIN,
        )
    )

    val_low = int(
        max(
            0,
            np.percentile(value_valid, 1) - VALUE_MARGIN,
        )
    )

    return {
        "hue_ranges": hue_ranges,
        "sat_min": sat_low,
        "sat_max": 255,
        "val_min": val_low,
        "val_max": 255,
        "samples": len(sample_patches),
        "statistics": {
            "hue_circular_centre": float(centre_hue),
            "hue_half_width": float(half_width),
            "saturation_median": float(np.median(saturation_valid)),
            "value_median": float(np.median(value_valid)),
            "pixel_count": int(len(pixels)),
            "valid_colour_pixels": int(len(hue_valid)),
        },
    }


# ============================================================
# GENERATE MASK
# ============================================================

def create_mask(hsv_image, calibration):
    """Create one binary mask, supporting wrapped hue ranges."""
    final_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    for hue_low, hue_high in calibration["hue_ranges"]:
        lower = np.array(
            [hue_low, calibration["sat_min"], calibration["val_min"]],
            dtype=np.uint8,
        )

        upper = np.array(
            [hue_high, calibration["sat_max"], calibration["val_max"]],
            dtype=np.uint8,
        )

        partial_mask = cv2.inRange(hsv_image, lower, upper)
        final_mask = cv2.bitwise_or(final_mask, partial_mask)

    return final_mask


# ============================================================
# SMALL MORPHOLOGICAL CLEANUP
# ============================================================

def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# ============================================================
# CAMERA HELPERS
# ============================================================

def open_camera():
    """Open the MJPEG stream with finite timeouts when supported."""
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000,
    ]

    try:
        cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG, params)
    except cv2.error:
        cap = cv2.VideoCapture(CAMERA_URL)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open camera stream.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_frame_with_reconnect(cap, retries=3):
    """Read one frame and reconnect if the HTTP/MJPEG stream ends."""
    for attempt in range(1, retries + 1):
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap, frame

        print(f"Camera read failed ({attempt}/{retries}). Reconnecting...")
        cap.release()
        time.sleep(0.5)

        try:
            cap = open_camera()
        except RuntimeError:
            if attempt == retries:
                raise
            time.sleep(0.5)

    return cap, None


cap = open_camera()


# ============================================================
# CAPTURE CALIBRATION FRAME
# ============================================================

print()
print("COLOUR CALIBRATION")
print("==================")
print()
print("SPACE : freeze current arena frame")
print("Q     : quit")
print()

frozen_frame = None

while True:
    cap, frame = read_frame_with_reconnect(cap)

    if frame is None:
        continue

    birdseye = preprocess_frame(frame)
    preview = birdseye.copy()

    cv2.putText(
        preview,
        "SPACE = freeze for colour calibration",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Live Bird's-Eye Arena", preview)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        frozen_frame = birdseye.copy()
        break

    if key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit

cv2.destroyWindow("Live Bird's-Eye Arena")

# The manual sampling stage may take minutes. Do not leave the MJPEG
# connection open and unread during that time: many camera servers
# terminate idle/stalled streams. We will open a fresh connection for
# live validation after the calibration has been saved.
cap.release()
cap = None


# ============================================================
# HSV VERSION OF CALIBRATION FRAME
# ============================================================

frozen_hsv = cv2.cvtColor(frozen_frame, cv2.COLOR_BGR2HSV)


# ============================================================
# SAMPLE STORAGE
# ============================================================

samples = {colour: [] for colour in COLOURS}
sample_points = {colour: [] for colour in COLOURS}
selected_colour = "red"


# ============================================================
# SAMPLE PATCH
# ============================================================

def get_patch_pixels(x, y):
    x1 = max(0, x - PATCH_RADIUS)
    x2 = min(birdseye_width, x + PATCH_RADIUS + 1)
    y1 = max(0, y - PATCH_RADIUS)
    y2 = min(birdseye_height, y + PATCH_RADIUS + 1)

    patch = frozen_hsv[y1:y2, x1:x2]
    return patch.reshape(-1, 3)


# ============================================================
# CALIBRATION UI / MOUSE CALLBACK
# ============================================================

WINDOW_NAME = "Colour Calibration"

# Clickable colour selector buttons. This avoids depending on
# OpenCV keyboard focus, which can be unreliable on some Windows
# OpenCV builds after switching/destroying windows.
BUTTON_HEIGHT = 58
BUTTON_GAP = 8
BUTTON_WIDTH = 150
BUTTON_Y1 = 8
BUTTON_Y2 = BUTTON_Y1 + BUTTON_HEIGHT
BUTTONS = {}

for i, colour in enumerate(COLOURS):
    x1 = 8 + i * (BUTTON_WIDTH + BUTTON_GAP)
    x2 = x1 + BUTTON_WIDTH
    BUTTONS[colour] = (x1, BUTTON_Y1, x2, BUTTON_Y2)


def select_colour(colour):
    global selected_colour
    selected_colour = colour
    print(f"Selected colour: {selected_colour.upper()}")


def get_clicked_button(x, y):
    for colour, (x1, y1, x2, y2) in BUTTONS.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return colour
    return None


def mouse_callback(event, x, y, flags, param):
    global selected_colour

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # First check whether one of the colour-selector buttons was clicked.
    button_colour = get_clicked_button(x, y)
    if button_colour is not None:
        select_colour(button_colour)
        return

    # Otherwise this is a sample click on the frozen arena image.
    patch_pixels = get_patch_pixels(x, y)

    samples[selected_colour].append(patch_pixels)
    sample_points[selected_colour].append((x, y))

    patch_median = np.median(patch_pixels, axis=0)

    print(
        f"{selected_colour.upper()} sample "
        f"{len(samples[selected_colour])}: "
        f"position=({x},{y})  "
        f"patch median HSV="
        f"({patch_median[0]:.1f}, "
        f"{patch_median[1]:.1f}, "
        f"{patch_median[2]:.1f})"
    )


cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print()
print("Sampling controls")
print("-----------------")
print("BEST METHOD: click RED / GREEN / YELLOW / BLUE buttons in the window")
print("Then click inside that coloured object to add samples.")
print()
print("Keyboard alternatives:")
print("R/G/Y/B or 1/2/3/4 : select colour")
print("U                 : undo last sample")
print("X                 : clear selected colour")
print("ENTER             : calculate + save calibration")
print("Q / ESC           : quit")
print()
print(f"Take at least {MIN_SAMPLES_PER_COLOUR} samples per colour.")
print("Click WELL INSIDE each coloured area, not on the white border.")


# ============================================================
# DRAW CALIBRATION SCREEN
# ============================================================

def draw_calibration_screen():
    display = frozen_frame.copy()

    # Draw every saved sample point.
    for colour in COLOURS:
        bgr = DISPLAY_COLOURS[colour]

        for index, (x, y) in enumerate(sample_points[colour]):
            cv2.rectangle(
                display,
                (x - PATCH_RADIUS, y - PATCH_RADIUS),
                (x + PATCH_RADIUS, y + PATCH_RADIUS),
                bgr,
                2,
            )

            cv2.circle(display, (x, y), 4, bgr, -1)

            cv2.putText(
                display,
                str(index + 1),
                (x + 12, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bgr,
                2,
            )

    # Semi-opaque-looking dark toolbar (solid for reliable readability).
    cv2.rectangle(
        display,
        (0, 0),
        (birdseye_width, 105),
        (20, 20, 20),
        -1,
    )

    # Clickable colour selector buttons.
    for colour in COLOURS:
        x1, y1, x2, y2 = BUTTONS[colour]
        bgr = DISPLAY_COLOURS[colour]

        # Fill selected button more strongly, otherwise dark fill.
        fill = bgr if colour == selected_colour else (45, 45, 45)
        text_colour = (0, 0, 0) if colour == selected_colour else bgr

        cv2.rectangle(display, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 255, 255) if colour == selected_colour else bgr,
            2,
        )

        label = colour.upper()
        cv2.putText(
            display,
            label,
            (x1 + 16, y1 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            text_colour,
            2,
        )

    count_text = (
        f"Selected: {selected_colour.upper()}   "
        f"R:{len(samples['red'])}   "
        f"G:{len(samples['green'])}   "
        f"Y:{len(samples['yellow'])}   "
        f"B:{len(samples['blue'])}"
    )

    cv2.putText(
        display,
        count_text,
        (20, 93),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        2,
    )

    return display


# ============================================================
# ROBUST KEY HANDLING
# ============================================================

def normalise_key(raw_key):
    """Return a lowercase ASCII character where possible."""
    if raw_key < 0:
        return None

    # waitKeyEx can include platform-specific high bits.
    key = raw_key & 0xFF

    if 0 <= key <= 255:
        try:
            return chr(key).lower()
        except ValueError:
            pass

    return None


# ============================================================
# SAMPLING LOOP
# ============================================================

while True:
    display = draw_calibration_screen()
    cv2.imshow(WINDOW_NAME, display)

    # waitKeyEx is more robust than waitKey for Windows keyboard handling.
    raw_key = cv2.waitKeyEx(20)
    key_char = normalise_key(raw_key)
    key_byte = raw_key & 0xFF if raw_key >= 0 else -1

    if key_char in ("r", "1"):
        select_colour("red")

    elif key_char in ("g", "2"):
        select_colour("green")

    elif key_char in ("y", "3"):
        select_colour("yellow")

    elif key_char in ("b", "4"):
        select_colour("blue")

    elif key_char == "u":
        if samples[selected_colour]:
            samples[selected_colour].pop()
            sample_points[selected_colour].pop()
            print(f"Removed last {selected_colour} sample.")

    elif key_char == "x":
        samples[selected_colour].clear()
        sample_points[selected_colour].clear()
        print(f"Cleared all {selected_colour} samples.")

    # ENTER can appear as 10 or 13 depending on platform/backend.
    elif key_byte in (10, 13):
        missing = [
            colour
            for colour in COLOURS
            if len(samples[colour]) < MIN_SAMPLES_PER_COLOUR
        ]

        if missing:
            print("Need more samples for: " + ", ".join(missing))
            continue

        break

    elif key_char == "q" or key_byte == 27:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        raise SystemExit

cv2.destroyWindow(WINDOW_NAME)


# ============================================================
# BUILD FINAL CALIBRATION
# ============================================================

colour_calibration = {
    "colour_space": "OpenCV HSV",
    "hue_range": [0, 179],
    "saturation_range": [0, 255],
    "value_range": [0, 255],
    "patch_radius_px": PATCH_RADIUS,
    "minimum_samples_per_colour": MIN_SAMPLES_PER_COLOUR,
    "colours": {},
}

print()
print("Calculated HSV ranges")
print("=====================")

for colour in COLOURS:
    result = calculate_colour_calibration(colour, samples[colour])
    colour_calibration["colours"][colour] = result

    print()
    print(colour.upper())
    print(f"  H centre : {result['statistics']['hue_circular_centre']:.2f}")
    print(f"  H width  : +/-{result['statistics']['hue_half_width']:.2f}")
    print(f"  H ranges : {result['hue_ranges']}")
    print(f"  S        : {result['sat_min']} - {result['sat_max']}")
    print(f"  V        : {result['val_min']} - {result['val_max']}")


# ============================================================
# SAVE JSON
# ============================================================

with open(COLOUR_CALIBRATION_FILE, "w") as file:
    json.dump(colour_calibration, file, indent=4)

print()
print(f"Saved: {COLOUR_CALIBRATION_FILE}")


# ============================================================
# LIVE MASK VALIDATION
# ============================================================

print()
print("LIVE VALIDATION")
print("===============")
print()
print("Move the coloured objects around the arena and verify each mask.")
print("Q = quit")
print()


def make_mask_panel(mask, label, display_colour):
    panel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        panel,
        label.upper(),
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        display_colour,
        2,
    )

    return panel


# Re-open the stream from scratch for live validation. This avoids
# resuming an MJPEG connection that sat unread during manual sampling.
print("Opening fresh camera connection for live validation...")
cap = open_camera()
print("Camera connected. Live validation running.")


while True:
    try:
        cap, frame = read_frame_with_reconnect(cap)
    except RuntimeError as exc:
        print(f"Camera reconnect failed: {exc}")
        print("Retrying in 1 second...")
        time.sleep(1.0)
        try:
            cap = open_camera()
        except RuntimeError:
            continue
        continue

    if frame is None:
        continue

    birdseye = preprocess_frame(frame)
    hsv = cv2.cvtColor(birdseye, cv2.COLOR_BGR2HSV)

    masks = {}

    for colour in COLOURS:
        calibration = colour_calibration["colours"][colour]
        mask = create_mask(hsv, calibration)
        mask = clean_mask(mask)
        masks[colour] = mask

    # --------------------------------------------------------
    # COLOUR OVERLAY
    # --------------------------------------------------------

    overlay = birdseye.copy()

    for colour in COLOURS:
        mask = masks[colour]
        display_colour = np.array(DISPLAY_COLOURS[colour], dtype=np.float32)
        selected = mask > 0

        if np.any(selected):
            original_pixels = overlay[selected].astype(np.float32)
            overlay[selected] = (
                0.55 * original_pixels
                + 0.45 * display_colour
            ).astype(np.uint8)

    # --------------------------------------------------------
    # INDIVIDUAL MASK GRID
    # --------------------------------------------------------

    red_panel = make_mask_panel(
        masks["red"], "Red", DISPLAY_COLOURS["red"]
    )
    green_panel = make_mask_panel(
        masks["green"], "Green", DISPLAY_COLOURS["green"]
    )
    yellow_panel = make_mask_panel(
        masks["yellow"], "Yellow", DISPLAY_COLOURS["yellow"]
    )
    blue_panel = make_mask_panel(
        masks["blue"], "Blue", DISPLAY_COLOURS["blue"]
    )

    top_row = np.hstack((red_panel, green_panel))
    bottom_row = np.hstack((yellow_panel, blue_panel))
    mask_grid = np.vstack((top_row, bottom_row))

    mask_grid = cv2.resize(
        mask_grid,
        (birdseye_width, birdseye_height),
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.imshow("Colour Detection Overlay", overlay)
    cv2.imshow("Individual HSV Masks", mask_grid)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
