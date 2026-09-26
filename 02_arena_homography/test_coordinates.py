# test_coordinates.py

import cv2
import json
import numpy as np


CAMERA_URL = "http://196.24.154.126:8080/video"

CAMERA_CAL_FILE = "camera_calibration.json"
ARENA_CAL_FILE = "arena_calibration.json"


# ============================================================
# LOAD CAMERA CALIBRATION
# ============================================================

with open(CAMERA_CAL_FILE, "r") as file:
    camera_data = json.load(file)

camera_matrix = np.array(
    camera_data["camera_matrix"],
    dtype=np.float64
)

dist_coeffs = np.array(
    camera_data["distortion_coefficients"],
    dtype=np.float64
)

image_width = camera_data["image_width"]
image_height = camera_data["image_height"]


# ============================================================
# LOAD ARENA HOMOGRAPHY
# ============================================================

with open(ARENA_CAL_FILE, "r") as file:
    arena_data = json.load(file)

H_world = np.array(
    arena_data["homography_image_to_world"],
    dtype=np.float64
)


# ============================================================
# UNDISTORTION MAP
# ============================================================

map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix,
    dist_coeffs,
    None,
    camera_matrix,
    (image_width, image_height),
    cv2.CV_32FC1
)


# ============================================================
# PIXEL → WORLD
# ============================================================

def pixel_to_world(x, y):

    pixel = np.array(
        [[[x, y]]],
        dtype=np.float32
    )

    world = cv2.perspectiveTransform(
        pixel,
        H_world
    )

    world_x = float(world[0][0][0])
    world_y = float(world[0][0][1])

    return world_x, world_y


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")


ret, frame = cap.read()

cap.release()

if not ret:
    raise RuntimeError("Could not capture frame.")


frame = cv2.remap(
    frame,
    map1,
    map2,
    interpolation=cv2.INTER_LINEAR
)


display = frame.copy()


# ============================================================
# CLICK CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global display

    if event == cv2.EVENT_LBUTTONDOWN:

        world_x, world_y = pixel_to_world(
            x,
            y
        )

        print()
        print(f"Pixel : ({x}, {y})")
        print(
            f"World : "
            f"({world_x:.1f}, "
            f"{world_y:.1f}) mm"
        )

        cv2.circle(
            display,
            (x, y),
            7,
            (0, 255, 0),
            -1
        )

        text = (
            f"{world_x:.0f}, "
            f"{world_y:.0f} mm"
        )

        cv2.putText(
            display,
            text,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


# ============================================================
# DISPLAY
# ============================================================

cv2.namedWindow("World Coordinate Test")

cv2.setMouseCallback(
    "World Coordinate Test",
    mouse_callback
)


print()
print("Click anywhere on the arena.")
print("The program will report the real-world position.")
print()
print("Q = quit")


while True:

    cv2.imshow(
        "World Coordinate Test",
        display
    )

    key = cv2.waitKey(20) & 0xFF

    if key == ord("q"):
        break


cv2.destroyAllWindows()