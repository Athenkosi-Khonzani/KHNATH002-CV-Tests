# AI Learner Rover Driver — Computer Vision Pipeline

The perception half of the AI Learner Rover Driver project (SW-8). This pipeline
turns a raw overhead-camera frame into a structured, LLM-ready description of the
arena: the rover's pose and the positions of the coloured goal and obstacle
blocks, all in real-world millimetres.

It is built as a sequence of numbered, self-contained stages. Each stage is a
calibration or detection step that writes its result to a JSON file, and each
later stage reads the outputs of the earlier ones. This makes the pipeline easy
to run, inspect and debug one step at a time, and keeps every result on disk as
evidence.

---

## Design principles

- **Classical, deterministic computer vision.** The arena is controlled (known
  lighting, chosen colours, a cooperative rover), so classical OpenCV methods are
  used throughout rather than deep learning. In this setting they are faster,
  fully explainable, and more robust than a learned detector would be.
- **Calibrate once, reuse everywhere.** Camera intrinsics, the arena homography
  and the colour thresholds are each calibrated once and saved to
  `calibration/`. The detection stages load these and do not re-calibrate.
- **Structured symbolic output.** The final world-state is emitted as Markdown
  and JSON with real millimetre coordinates, not as an image. Localisation is
  done by this calibrated pipeline; the language model only reasons over the
  resulting numbers.
- **Measured, not assumed.** Accuracy is validated against ground truth
  (validation stages 06 and 07B), so the pipeline's error is a known quantity.

---

## Pipeline stages

| # | Stage | Purpose | Key output |
|---|---|---|---|
| 01 | `01_camera_calibration` | Correct lens distortion using a checkerboard | `camera_calibration.json` |
| 02 | `02_arena_homography` | Map image pixels to arena millimetres | `arena_calibration.json` |
| 03 | `03_colour_calibration` | Set HSV ranges for each block colour | `colour_calibration.json` |
| 04 | `04_block_detection` | Detect coloured blocks, report centre, size, orientation | detections JSON |
| 05 | `05_aruco_rover` | Recover rover pose (position + heading) from an ArUco marker | pose JSON |
| 06 | `06_heading_validation` | Quantify heading accuracy against known targets | validation CSV/JSON |
| 07 | `07_position_compensation` | Offset the marker pose to the rover's true centre | compensated pose JSON |
| 07B | `07B_position_validation` | Quantify position accuracy against tape-measured points | validation output |
| 08 | `08_pose_filtering` | Temporal smoothing, confidence and staleness flags | filtered pose JSON |
| 09 | `09_world_state` | Fuse rover pose + blocks into the LLM-ready world-state | `world_state.md` / `.json` / map |

The `calibration/` folder holds the current working copies of the three
calibration files, which the detection stages read.

---

## The final output (stage 09)

Stage 09 produces three views of the same world-state:

- **`world_state.md`** — the Markdown description sent to the LLM. It states the
  arena size, the coordinate frame, the rover pose, and each block's centre,
  size and orientation in millimetres.
- **`world_state.json`** — the machine-readable version for the host application.
- **`world_state_map.txt`** — an ASCII top-down map for quick human inspection
  and debugging.

Example world-state (Markdown):

```
# AI Learner World State

Arena: 1710 mm x 1026 mm.
Coordinate frame: origin at bottom-left, +X right, +Y up, heading 0 deg toward +Y.

## Rover
- Position: (952.8, 765.6) mm
- Heading: 290.0 deg
- Pose valid: True
- Confidence: 1.000

## Coloured blocks
- red_block_01: red block, centre=(1200.7, 376.3) mm, size~306.6x197.2 mm, orientation=101.3 deg.
- green_block_01: green block, centre=(285.9, 360.0) mm, size~288.6x202.0 mm, orientation=90.0 deg.
- ...
```

---

## Measured accuracy

The pipeline's accuracy is validated, not assumed:

- **Camera calibration:** 0.47 px RMS reprojection error (1920x1080, 9x6
  checkerboard, 25 mm squares).
- **Rover heading:** mean absolute error ~0.13 deg, standard deviation ~0.04 deg,
  maximum error ~0.19 deg over 30 samples at a known heading (ArUco marker id 29,
  `DICT_4X4_50`).

These figures are the perception error budget the navigation results are
interpreted against.

---

## Coordinate frame

- Origin at the **bottom-left** corner of the arena.
- **+X** to the right, **+Y** up.
- Heading **0 deg** points toward **+Y** (up); 90 is +X (right), 180 down, 270
  left.

All stages and the world-state use this one frame.

---

## Requirements

- Python 3.10+
- `opencv-contrib-python` (the `contrib` build is required for `cv2.aruco`)
- `numpy`

```bash
pip install opencv-contrib-python numpy
```

An overhead camera stream (e.g. an Android phone running IP Webcam, MJPEG) and a
printed ArUco marker (`DICT_4X4_50`, id 29) mounted on the rover.

---

## How to run

Run the stages in order the first time, so each calibration exists before the
stage that needs it.

```bash
# 1. Camera intrinsics (capture checkerboard views, then calibrate)
python 01_camera_calibration/capture_calibration.py
python 01_camera_calibration/calibrate_camera.py

# 2. Arena homography (click the four arena corners)
python 02_arena_homography/calibrate_arena.py
python 02_arena_homography/test_coordinates.py     # verify pixel -> mm

# 3. Colour thresholds (tune HSV per block colour)
python 03_colour_calibration/calibrate_colours.py

# 4-5. Detection
python 04_block_detection/detect_blocks.py
python 05_aruco_rover/detect_aruco_rover.py

# 6-8. Validation, compensation, filtering
python 06_heading_validation/validate_rover_heading.py
python 07_position_compensation/detect_compensated_rover_pose.py
python 07B_position_validation/validate_rover_position.py
python 08_pose_filtering/filter_rover_pose.py

# 9. World-state for the LLM
python 09_world_state/generate_world_state.py
```

After the first full run, the calibration files in `calibration/` are reused, so
day-to-day operation only needs the detection and world-state stages (and a
re-calibration of the homography if the camera is moved).

---

## Integration with the host application

Stage 09 is the boundary with the rest of the system. The host application's
`PerceptionPort` consumes the world-state produced here: `world_state.json` for
the machine-readable scene and `world_state.md` for the text passed to the LLM
planner. Replacing the host's placeholder perception adapter with a call into
this pipeline connects the perception half to the mission controller.

---

## Notes

- If the camera is physically moved, re-run stage 02 (arena homography); the
  other calibrations are unaffected.
- Lock the camera's exposure, focus and white balance before calibrating
  colours, and keep the arena lighting fixed between runs, so the HSV thresholds
  remain valid.