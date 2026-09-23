# capture_calibration.py

import cv2
import os
import time

CAMERA_URL = "http://196.47.212.136:8080/video"

OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    raise RuntimeError("Could not open camera stream.")

image_count = 0

print("Camera calibration capture")
print("--------------------------")
print("C = capture image")
print("Q = quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame.")
        time.sleep(0.1)
        continue

    display = frame.copy()

    cv2.putText(
        display,
        f"Captured: {image_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Calibration Capture", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        filename = os.path.join(
            OUTPUT_DIR,
            f"calibration_{image_count:03d}.png"
        )

        cv2.imwrite(filename, frame)

        print(f"Saved {filename}")

        image_count += 1

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()