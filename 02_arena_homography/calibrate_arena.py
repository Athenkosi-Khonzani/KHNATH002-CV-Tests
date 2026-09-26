import cv2
import json
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

CAMERA_URL = "http://196.24.154.126:8080/video"

CAMERA_CALIBRATION_FILE = "camera_calibration.json"
ARENA_CALIBRATION_FILE = "arena_calibration.json"

ARENA_WIDTH_MM = 1710.0
ARENA_HEIGHT_MM = 1026.0

# Bird's-eye display resolution
BIRDSEYE_WIDTH = 1000
BIRDSEYE_HEIGHT = 1000


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

cal_width = camera_data["image_width"]
cal_height = camera_data["image_height"]


print("Camera calibration loaded")
print(f"Expected resolution: {cal_width} x {cal_height}")


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    raise RuntimeError("Could not open camera stream.")


# ============================================================
# PRECOMPUTE UNDISTORTION MAP
# ============================================================

map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix,
    dist_coeffs,
    None,
    camera_matrix,
    (cal_width, cal_height),
    cv2.CV_32FC1
)


# ============================================================
# CLICK STATE
# ============================================================

clicked_points = []

point_names = [
    "TOP LEFT",
    "TOP RIGHT",
    "BOTTOM RIGHT",
    "BOTTOM LEFT"
]

frozen_frame = None


# ============================================================
# DRAW HELPERS
# ============================================================

def draw_points(image):

    display = image.copy()

    for i, point in enumerate(clicked_points):

        x, y = point

        cv2.circle(
            display,
            (x, y),
            8,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            display,
            f"{i + 1}: {point_names[i]}",
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Draw lines between selected points
    if len(clicked_points) >= 2:

        for i in range(len(clicked_points) - 1):

            cv2.line(
                display,
                clicked_points[i],
                clicked_points[i + 1],
                (255, 200, 0),
                2
            )

    if len(clicked_points) == 4:

        cv2.line(
            display,
            clicked_points[3],
            clicked_points[0],
            (255, 200, 0),
            2
        )

    return display


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(clicked_points) < 4:

            clicked_points.append((x, y))

            print(
                f"Point {len(clicked_points)} "
                f"({point_names[len(clicked_points) - 1]}): "
                f"({x}, {y})"
            )


# ============================================================
# LIVE CAPTURE
# ============================================================

print()
print("Arena calibration")
print("-----------------")
print("SPACE : freeze frame")
print("Q     : quit")


while True:

    ret, frame = cap.read()

    if not ret:
        continue

    h, w = frame.shape[:2]

    if (w, h) != (cal_width, cal_height):

        raise RuntimeError(
            f"Camera is outputting {w}x{h}, "
            f"but calibration expects "
            f"{cal_width}x{cal_height}."
        )

    undistorted = cv2.remap(
        frame,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR
    )

    preview = undistorted.copy()

    cv2.putText(
        preview,
        "SPACE = freeze frame",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Live Undistorted Camera",
        preview
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):

        frozen_frame = undistorted.copy()
        break

    elif key == ord("q"):

        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit


cap.release()
cv2.destroyAllWindows()


# ============================================================
# CORNER SELECTION
# ============================================================

cv2.namedWindow("Select Arena Corners")

cv2.setMouseCallback(
    "Select Arena Corners",
    mouse_callback
)


print()
print("Click arena corners in this EXACT order:")
print()
print("1. TOP LEFT")
print("2. TOP RIGHT")
print("3. BOTTOM RIGHT")
print("4. BOTTOM LEFT")
print()
print("R = reset")
print("S = save calibration once 4 points are selected")


while True:

    display = draw_points(frozen_frame)

    if len(clicked_points) < 4:

        instruction = (
            f"Click {len(clicked_points) + 1}: "
            f"{point_names[len(clicked_points)]}"
        )

    else:

        instruction = "Press S to calculate homography"

    cv2.putText(
        display,
        instruction,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Select Arena Corners",
        display
    )

    key = cv2.waitKey(10) & 0xFF

    if key == ord("r"):

        clicked_points.clear()
        print("Points reset.")

    elif key == ord("s") and len(clicked_points) == 4:

        break

    elif key == ord("q"):

        cv2.destroyAllWindows()
        raise SystemExit


# ============================================================
# COMPUTE WORLD HOMOGRAPHY
# ============================================================

image_points = np.array(
    clicked_points,
    dtype=np.float32
)

# Same order:
# TL, TR, BR, BL
world_points = np.array(
    [
        [0.0, ARENA_HEIGHT_MM],
        [ARENA_WIDTH_MM, ARENA_HEIGHT_MM],
        [ARENA_WIDTH_MM, 0.0],
        [0.0, 0.0]
    ],
    dtype=np.float32
)


H_world = cv2.getPerspectiveTransform(
    image_points,
    world_points
)

H_world_inverse = np.linalg.inv(H_world)


# ============================================================
# BIRD'S-EYE HOMOGRAPHY
# ============================================================

birdseye_points = np.array(
    [
        [0, 0],
        [BIRDSEYE_WIDTH - 1, 0],
        [BIRDSEYE_WIDTH - 1, BIRDSEYE_HEIGHT - 1],
        [0, BIRDSEYE_HEIGHT - 1]
    ],
    dtype=np.float32
)


H_birdseye = cv2.getPerspectiveTransform(
    image_points,
    birdseye_points
)


birdseye = cv2.warpPerspective(
    frozen_frame,
    H_birdseye,
    (BIRDSEYE_WIDTH, BIRDSEYE_HEIGHT)
)


# ============================================================
# SAVE CALIBRATION
# ============================================================

arena_data = {

    "arena_width_mm": ARENA_WIDTH_MM,

    "arena_height_mm": ARENA_HEIGHT_MM,

    "image_width": cal_width,

    "image_height": cal_height,

    "corner_order": [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left"
    ],

    "image_points": image_points.tolist(),

    "world_points_mm": world_points.tolist(),

    "homography_image_to_world":
        H_world.tolist(),

    "homography_world_to_image":
        H_world_inverse.tolist(),

    "birdseye_width": BIRDSEYE_WIDTH,

    "birdseye_height": BIRDSEYE_HEIGHT,

    "homography_image_to_birdseye":
        H_birdseye.tolist()
}


with open(
    ARENA_CALIBRATION_FILE,
    "w"
) as file:

    json.dump(
        arena_data,
        file,
        indent=4
    )


cv2.imwrite(
    "arena_calibration_frame.png",
    draw_points(frozen_frame)
)

cv2.imwrite(
    "arena_birdseye.png",
    birdseye
)


print()
print("Arena calibration complete.")
print()
print("Saved:")
print("  arena_calibration.json")
print("  arena_calibration_frame.png")
print("  arena_birdseye.png")


# ============================================================
# DISPLAY RESULT
# ============================================================

cv2.imshow(
    "Arena Calibration",
    draw_points(frozen_frame)
)

cv2.imshow(
    "Bird's-eye Arena",
    birdseye
)

print()
print("Press any key to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()