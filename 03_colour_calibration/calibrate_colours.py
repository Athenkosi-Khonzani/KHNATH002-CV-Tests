import cv2
import json
import math
import numpy as np


# ============================================================
# USER SETTINGS
# ============================================================

# Use your actual MJPEG/IP stream URL.
CAMERA_URL = "http://196.47.212.136:8080/video"

CAMERA_CALIBRATION_FILE = "camera_calibration.json"
ARENA_CALIBRATION_FILE = "arena_calibration.json"
COLOUR_CALIBRATION_FILE = "colour_calibration.json"

# Each mouse click samples a square patch around the clicked point.
PATCH_RADIUS = 10          # 21 x 21 px patch

# Minimum independent click locations per colour.
MIN_SAMPLES_PER_COLOUR = 3

# Extra tolerance added around the measured HSV samples.
HUE_MARGIN = 3
SATURATION_MARGIN = 25
VALUE_MARGIN = 30

COLOURS = ["red", "green", "yellow", "blue"]

# Display colours in BGR.
DISPLAY_COLOURS = {
    "red":    (0, 0, 255),
    "green":  (0, 255, 0),
    "yellow": (0, 255, 255),
    "blue":   (255, 0, 0),
}


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

with open(CAMERA_CALIBRATION_FILE, "r") as file:
    camera_data = json.load(file)

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
# LOAD ARENA CALIBRATION
# ============================================================

with open(ARENA_CALIBRATION_FILE, "r") as file:
    arena_data = json.load(file)

H_birdseye = np.array(
    arena_data["homography_image_to_birdseye"],
    dtype=np.float64
)

birdseye_width = int(
    arena_data["birdseye_width"]
)

birdseye_height = int(
    arena_data["birdseye_height"]
)


print(
    f"Camera calibration: "
    f"{camera_width} x {camera_height}"
)

print(
    f"Bird's-eye output: "
    f"{birdseye_width} x {birdseye_height}"
)


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
# IMAGE PREPROCESSING
# ============================================================

def preprocess_frame(frame):
    """
    Raw camera frame
        -> undistort
        -> arena bird's-eye image
    """

    h, w = frame.shape[:2]

    if (w, h) != (camera_width, camera_height):
        raise RuntimeError(
            f"Camera returned {w}x{h}, "
            f"but calibration expects "
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
# HUE CALCULATION
# ============================================================

def calculate_hue_ranges(hues):
    """
    OpenCV hue is circular:

        0 ------------------------------ 179
        ^                                  ^
        red                                red

    Therefore ordinary min/max fails for colours around the
    red wrap boundary.

    This function determines the circular centre and builds
    either one hue range or two wrapped ranges automatically.
    """

    hues = np.asarray(
        hues,
        dtype=np.float64
    )

    # Convert OpenCV hue [0,179] into angles [0,2*pi].
    angles = hues * (2.0 * math.pi / 180.0)

    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))

    mean_angle = math.atan2(
        mean_sin,
        mean_cos
    )

    if mean_angle < 0:
        mean_angle += 2.0 * math.pi

    centre_hue = (
        mean_angle
        * 180.0
        / (2.0 * math.pi)
    )

    # Circular difference in range [-90,+90).
    differences = (
        (hues - centre_hue + 90.0)
        % 180.0
    ) - 90.0

    low_diff, high_diff = np.percentile(
        differences,
        [2, 98]
    )

    low = (
        centre_hue
        + low_diff
        - HUE_MARGIN
    )

    high = (
        centre_hue
        + high_diff
        + HUE_MARGIN
    )

    ranges = []

    # The range crosses H = 0.
    if low < 0:

        ranges.append(
            [
                0,
                int(
                    min(
                        179,
                        math.ceil(high)
                    )
                )
            ]
        )

        ranges.append(
            [
                int(
                    max(
                        0,
                        math.floor(180 + low)
                    )
                ),
                179
            ]
        )

    # The range crosses H = 179.
    elif high > 179:

        ranges.append(
            [
                int(
                    max(
                        0,
                        math.floor(low)
                    )
                ),
                179
            ]
        )

        ranges.append(
            [
                0,
                int(
                    min(
                        179,
                        math.ceil(high - 180)
                    )
                )
            ]
        )

    else:

        ranges.append(
            [
                int(max(0, math.floor(low))),
                int(min(179, math.ceil(high)))
            ]
        )

    return ranges, centre_hue


# ============================================================
# CALCULATE CALIBRATION FOR ONE COLOUR
# ============================================================

def calculate_colour_calibration(
    colour_name,
    sample_patches
):

    # Combine all independent sampled patches.
    pixels = np.concatenate(
        sample_patches,
        axis=0
    )

    hue = pixels[:, 0]
    saturation = pixels[:, 1]
    value = pixels[:, 2]

    hue_ranges, hue_centre = (
        calculate_hue_ranges(hue)
    )

    # We mainly care about sensible LOWER S/V limits.
    #
    # Maximum values remain at 255 because rejecting a pixel
    # merely because it became brighter/more saturated tends
    # to make masks unnecessarily fragile.

    saturation_low = int(
        max(
            0,
            np.percentile(
                saturation,
                2
            ) - SATURATION_MARGIN
        )
    )

    value_low = int(
        max(
            0,
            np.percentile(
                value,
                2
            ) - VALUE_MARGIN
        )
    )

    calibration = {

        "hue_ranges": hue_ranges,

        "sat_min": saturation_low,
        "sat_max": 255,

        "val_min": value_low,
        "val_max": 255,

        "samples": len(sample_patches),

        "statistics": {
            "hue_circular_centre": float(
                hue_centre
            ),

            "saturation_median": float(
                np.median(saturation)
            ),

            "value_median": float(
                np.median(value)
            ),

            "pixel_count": int(
                len(pixels)
            )
        }
    }

    return calibration


# ============================================================
# GENERATE MASK
# ============================================================

def create_mask(hsv_image, calibration):

    final_mask = np.zeros(
        hsv_image.shape[:2],
        dtype=np.uint8
    )

    for hue_low, hue_high in calibration[
        "hue_ranges"
    ]:

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

        partial_mask = cv2.inRange(
            hsv_image,
            lower,
            upper
        )

        final_mask = cv2.bitwise_or(
            final_mask,
            partial_mask
        )

    return final_mask


# ============================================================
# SMALL MORPHOLOGICAL CLEANUP
# ============================================================

def clean_mask(mask):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    # Remove tiny isolated pixels.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small holes.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open camera stream."
    )


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

    ret, frame = cap.read()

    if not ret:
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
        2
    )

    cv2.imshow(
        "Live Bird's-Eye Arena",
        preview
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):

        frozen_frame = birdseye.copy()
        break

    elif key == ord("q"):

        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit


cv2.destroyWindow(
    "Live Bird's-Eye Arena"
)


# ============================================================
# HSV VERSION OF CALIBRATION FRAME
# ============================================================

frozen_hsv = cv2.cvtColor(
    frozen_frame,
    cv2.COLOR_BGR2HSV
)


# ============================================================
# SAMPLE STORAGE
# ============================================================

samples = {
    colour: []
    for colour in COLOURS
}

sample_points = {
    colour: []
    for colour in COLOURS
}

selected_colour = "red"


# ============================================================
# SAMPLE PATCH
# ============================================================

def get_patch_pixels(x, y):

    x1 = max(
        0,
        x - PATCH_RADIUS
    )

    x2 = min(
        birdseye_width,
        x + PATCH_RADIUS + 1
    )

    y1 = max(
        0,
        y - PATCH_RADIUS
    )

    y2 = min(
        birdseye_height,
        y + PATCH_RADIUS + 1
    )

    patch = frozen_hsv[
        y1:y2,
        x1:x2
    ]

    pixels = patch.reshape(
        -1,
        3
    )

    return pixels


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global selected_colour

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    patch_pixels = get_patch_pixels(
        x,
        y
    )

    samples[selected_colour].append(
        patch_pixels
    )

    sample_points[selected_colour].append(
        (x, y)
    )

    hsv_centre = frozen_hsv[y, x]

    print(
        f"{selected_colour.upper()} "
        f"sample "
        f"{len(samples[selected_colour])}: "
        f"position=({x},{y})  "
        f"HSV="
        f"({hsv_centre[0]}, "
        f"{hsv_centre[1]}, "
        f"{hsv_centre[2]})"
    )


# ============================================================
# CALIBRATION WINDOW
# ============================================================

WINDOW_NAME = "Colour Calibration"

cv2.namedWindow(
    WINDOW_NAME
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


print()
print("Sampling controls")
print("-----------------")
print("R : select RED")
print("G : select GREEN")
print("Y : select YELLOW")
print("B : select BLUE")
print()
print("Left click : add sample")
print("U          : undo last sample")
print("X          : clear selected colour")
print("ENTER      : calculate + save calibration")
print("Q          : quit")
print()
print(
    "Take at least "
    f"{MIN_SAMPLES_PER_COLOUR} "
    "samples per colour."
)
print()
print(
    "CLICK WELL INSIDE EACH COLOURED AREA, "
    "NOT ON THE WHITE BORDER."
)


# ============================================================
# DRAW CALIBRATION SCREEN
# ============================================================

def draw_calibration_screen():

    display = frozen_frame.copy()

    # Draw every saved sample point.
    for colour in COLOURS:

        bgr = DISPLAY_COLOURS[colour]

        for index, (x, y) in enumerate(
            sample_points[colour]
        ):

            cv2.rectangle(
                display,
                (
                    x - PATCH_RADIUS,
                    y - PATCH_RADIUS
                ),
                (
                    x + PATCH_RADIUS,
                    y + PATCH_RADIUS
                ),
                bgr,
                2
            )

            cv2.circle(
                display,
                (x, y),
                4,
                bgr,
                -1
            )

            cv2.putText(
                display,
                str(index + 1),
                (x + 12, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bgr,
                2
            )

    # Top status panel.
    cv2.rectangle(
        display,
        (0, 0),
        (birdseye_width, 75),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        display,
        (
            f"Selected: "
            f"{selected_colour.upper()}"
        ),
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        DISPLAY_COLOURS[selected_colour],
        2
    )

    count_text = (
        f"R:{len(samples['red'])}   "
        f"G:{len(samples['green'])}   "
        f"Y:{len(samples['yellow'])}   "
        f"B:{len(samples['blue'])}"
    )

    cv2.putText(
        display,
        count_text,
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        2
    )

    return display


# ============================================================
# SAMPLING LOOP
# ============================================================

while True:

    display = draw_calibration_screen()

    cv2.imshow(
        WINDOW_NAME,
        display
    )

    key = cv2.waitKey(20) & 0xFF

    if key == ord("r"):

        selected_colour = "red"

    elif key == ord("g"):

        selected_colour = "green"

    elif key == ord("y"):

        selected_colour = "yellow"

    elif key == ord("b"):

        selected_colour = "blue"

    elif key == ord("u"):

        if samples[selected_colour]:

            samples[selected_colour].pop()
            sample_points[
                selected_colour
            ].pop()

            print(
                f"Removed last "
                f"{selected_colour} sample."
            )

    elif key == ord("x"):

        samples[selected_colour].clear()

        sample_points[
            selected_colour
        ].clear()

        print(
            f"Cleared all "
            f"{selected_colour} samples."
        )

    elif key in (10, 13):

        missing = []

        for colour in COLOURS:

            if (
                len(samples[colour])
                < MIN_SAMPLES_PER_COLOUR
            ):

                missing.append(colour)

        if missing:

            print()
            print(
                "Need more samples for: "
                + ", ".join(missing)
            )

            continue

        break

    elif key == ord("q"):

        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit


cv2.destroyWindow(
    WINDOW_NAME
)


# ============================================================
# BUILD FINAL CALIBRATION
# ============================================================

colour_calibration = {
    "colour_space": "OpenCV HSV",
    "hue_range": [0, 179],
    "saturation_range": [0, 255],
    "value_range": [0, 255],
    "patch_radius_px": PATCH_RADIUS,
    "colours": {}
}


print()
print("Calculated HSV ranges")
print("=====================")


for colour in COLOURS:

    result = calculate_colour_calibration(
        colour,
        samples[colour]
    )

    colour_calibration[
        "colours"
    ][colour] = result

    print()
    print(colour.upper())
    print(
        f"  H ranges : "
        f"{result['hue_ranges']}"
    )
    print(
        f"  S        : "
        f"{result['sat_min']} - "
        f"{result['sat_max']}"
    )
    print(
        f"  V        : "
        f"{result['val_min']} - "
        f"{result['val_max']}"
    )


# ============================================================
# SAVE JSON
# ============================================================

with open(
    COLOUR_CALIBRATION_FILE,
    "w"
) as file:

    json.dump(
        colour_calibration,
        file,
        indent=4
    )


print()
print(
    f"Saved: "
    f"{COLOUR_CALIBRATION_FILE}"
)


# ============================================================
# LIVE MASK VALIDATION
# ============================================================

print()
print("LIVE VALIDATION")
print("===============")
print()
print("Move the coloured objects around the arena.")
print("Q = quit")
print()


def make_mask_panel(
    mask,
    label,
    display_colour
):

    panel = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR
    )

    cv2.putText(
        panel,
        label.upper(),
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        display_colour,
        2
    )

    return panel


while True:

    ret, frame = cap.read()

    if not ret:
        continue

    birdseye = preprocess_frame(
        frame
    )

    hsv = cv2.cvtColor(
        birdseye,
        cv2.COLOR_BGR2HSV
    )

    masks = {}

    for colour in COLOURS:

        calibration = (
            colour_calibration[
                "colours"
            ][colour]
        )

        mask = create_mask(
            hsv,
            calibration
        )

        mask = clean_mask(mask)

        masks[colour] = mask


    # --------------------------------------------------------
    # BUILD COLOUR OVERLAY
    # --------------------------------------------------------

    overlay = birdseye.copy()

    for colour in COLOURS:

        mask = masks[colour]

        display_colour = np.array(
            DISPLAY_COLOURS[colour],
            dtype=np.float32
        )

        selected = mask > 0

        if np.any(selected):

            original_pixels = overlay[
                selected
            ].astype(np.float32)

            overlay[selected] = (
                0.55 * original_pixels
                + 0.45 * display_colour
            ).astype(np.uint8)


    # --------------------------------------------------------
    # MASK GRID
    # --------------------------------------------------------

    red_panel = make_mask_panel(
        masks["red"],
        "Red",
        DISPLAY_COLOURS["red"]
    )

    green_panel = make_mask_panel(
        masks["green"],
        "Green",
        DISPLAY_COLOURS["green"]
    )

    yellow_panel = make_mask_panel(
        masks["yellow"],
        "Yellow",
        DISPLAY_COLOURS["yellow"]
    )

    blue_panel = make_mask_panel(
        masks["blue"],
        "Blue",
        DISPLAY_COLOURS["blue"]
    )

    top_row = np.hstack(
        (
            red_panel,
            green_panel
        )
    )

    bottom_row = np.hstack(
        (
            yellow_panel,
            blue_panel
        )
    )

    mask_grid = np.vstack(
        (
            top_row,
            bottom_row
        )
    )

    # Resize grid so it isn't enormous.
    mask_grid = cv2.resize(
        mask_grid,
        (
            birdseye_width,
            birdseye_height
        )
    )


    # --------------------------------------------------------
    # SHOW RESULTS
    # --------------------------------------------------------

    cv2.imshow(
        "Colour Detection Overlay",
        overlay
    )

    cv2.imshow(
        "Individual HSV Masks",
        mask_grid
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()