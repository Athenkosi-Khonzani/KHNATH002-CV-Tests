# calibrate_camera.py

import cv2
import numpy as np
import glob
import json

# Number of INTERNAL checkerboard corners
CHECKERBOARD = (9, 6)

# Physical checkerboard square size
SQUARE_SIZE_MM = 25.0

image_files = glob.glob("./images/*.png")

if not image_files:
    raise RuntimeError("No calibration images found.")

# ---------------------------------------------------------
# Prepare known checkerboard coordinates
# ---------------------------------------------------------

object_points_template = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

object_points_template[:, :2] = (
    np.mgrid[
        0:CHECKERBOARD[0],
        0:CHECKERBOARD[1]
    ]
    .T
    .reshape(-1, 2)
)

object_points_template *= SQUARE_SIZE_MM


# ---------------------------------------------------------
# Storage
# ---------------------------------------------------------

object_points = []
image_points = []

image_size = None

successful_images = 0


# ---------------------------------------------------------
# Process images
# ---------------------------------------------------------

for filename in image_files:

    image = cv2.imread(filename)

    if image is None:
        print(f"Could not read {filename}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_size = gray.shape[::-1]

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD
    )

    preview = image.copy()

    if found:

        object_points.append(object_points_template)
        image_points.append(corners)

        successful_images += 1

        cv2.drawChessboardCorners(
            preview,
            CHECKERBOARD,
            corners,
            found
        )

        print(f"[OK] {filename}")

    else:

        print(f"[FAILED] {filename}")

    cv2.imshow("Checkerboard Detection", preview)

    cv2.waitKey(200)


cv2.destroyAllWindows()


if successful_images < 10:
    raise RuntimeError(
        "Too few successful calibration images. "
        "Capture more checkerboard views."
    )


# ---------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------

rms_error, camera_matrix, distortion_coeffs, rvecs, tvecs = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )
)


print()
print("Calibration complete")
print("--------------------")

print(f"Successful images: {successful_images}")
print(f"RMS reprojection error: {rms_error}")

print("\nCamera matrix:")
print(camera_matrix)

print("\nDistortion coefficients:")
print(distortion_coeffs)


# ---------------------------------------------------------
# Calculate mean reprojection error
# ---------------------------------------------------------

total_error = 0.0

for i in range(len(object_points)):

    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        distortion_coeffs
    )

    error = cv2.norm(
        image_points[i],
        projected_points,
        cv2.NORM_L2
    ) / len(projected_points)

    total_error += error


mean_error = total_error / len(object_points)

print(f"\nMean reprojection error: {mean_error:.4f} px")


# ---------------------------------------------------------
# Save calibration
# ---------------------------------------------------------

calibration_data = {

    "camera_matrix": camera_matrix.tolist(),

    "distortion_coefficients":
        distortion_coeffs.tolist(),

    "image_width": image_size[0],

    "image_height": image_size[1],

    "checkerboard": {
        "columns": CHECKERBOARD[0],
        "rows": CHECKERBOARD[1],
        "square_size_mm": SQUARE_SIZE_MM
    },

    "rms_error": float(rms_error),

    "mean_reprojection_error":
        float(mean_error)
}


with open("camera_calibration.json", "w") as file:
    json.dump(
        calibration_data,
        file,
        indent=4
    )


print("\nSaved:")
print("camera_calibration.json")