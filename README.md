# camera-track — webcam vision toys

Two real-time webcam apps built with **OpenCV** + **MediaPipe**:

- **`hand_counter.py`** — counts extended fingers (0–10) across up to two
  hands, overlaying the count and hand skeleton. Implements **KAN-15**
  (environment) and **KAN-16 → KAN-21**.
- **`eye_tracker.py`** — tracks gaze direction, blinks (count + rate),
  drowsiness, and no-blink/staring time from the eyes/irises via Face Mesh.
  Implements **KAN-24 → KAN-32**.

## Files

| File | Purpose |
|------|---------|
| `hand_counter.py` | The finger-counting app (KAN-16–21). |
| `eye_tracker.py` | Gaze / blink / drowsiness / no-blink tracker (KAN-24–32). |
| `verify_camera.py` | Environment/webcam smoke-test (KAN-15). |
| `requirements.txt` | Pinned dependencies. |

## Setup

A virtualenv (`.venv/`) is already created with everything installed. To
recreate it from scratch:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Both apps need real camera + display hardware, so run them on your own machine:

```powershell
.\.venv\Scripts\python.exe hand_counter.py     # finger counter
.\.venv\Scripts\python.exe eye_tracker.py      # gaze / blink / drowsiness
```

Press **`q`** or **Esc** in the window to quit.

### `hand_counter.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--camera N` | `0` | Webcam index |
| `--max-hands N` | `2` | Max hands to detect (use `1` for 0–5) |
| `--window N` | `5` | Debounce window in frames |
| `--no-flip` | off | Don't mirror the image |
| `--no-debounce` | off | Show the raw per-frame count |
| `--self-test` | — | Verify finger-detection logic without a camera |

### `eye_tracker.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--camera N` | `0` | Webcam index |
| `--ear-threshold F` | `0.21` | EAR below this counts as a closed eye |
| `--drowsy-seconds F` | `1.5` | Eyes-closed duration that flags drowsiness |
| `--no-blink-seconds F` | `10.0` | No-blink duration that flags staring/eye strain |
| `--no-flip` | off | Don't mirror the image |
| `--no-landmarks` | off | Don't draw the eye/iris mesh overlay |
| `--self-test` | — | Verify gaze/blink/drowsiness logic without a camera |

Each app's `--self-test` mode validates its detection logic against synthetic
landmarks and needs no hardware — handy for CI or a quick sanity check. For the
eye tracker it checks EAR (open vs. shut), gaze classification, single-blink
debounce, the drowsiness duration threshold, and the no-blink timer.

## How it maps to the tickets

- **KAN-16** — `cv2.VideoCapture` opens the webcam; the main loop reads frames.
- **KAN-17** — `mediapipe` Hands returns 21 landmarks per detected hand.
- **KAN-18** — `fingers_up()` judges each finger: non-thumb fingers by tip-vs-PIP
  *y*, the thumb by *x* (orientation taken from MediaPipe's handedness label).
- **KAN-19** — `cv2.putText` overlays the count; `drawing_utils` draws the skeleton.
- **KAN-20** — `max_num_hands=2`; fingers are summed across both hands (0–10).
- **KAN-21** — graceful "No hand" state, a `CountStabilizer` debounce to stop
  flicker, and a `try/finally` that releases the camera and windows on `q`.

### Eye tracker — `eye_tracker.py` (KAN-24–32)

- **KAN-24** — `mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)` adds the
  iris landmarks (468–477) on top of the 468 base face-mesh points.
- **KAN-25** — module-level index constants isolate each eye's EAR points, the
  corner/lid points used for gaze, and the iris rings (`LEFT_IRIS`/`RIGHT_IRIS`).
- **KAN-26** — `estimate_gaze()` maps the iris center's position between the eye
  corners (horizontal) and lids (vertical) to left/right/center/up/down.
- **KAN-27** — `eye_aspect_ratio()` is the standard EAR (two vertical distances
  over twice the horizontal); `average_ear()` means both eyes.
- **KAN-28** — `BlinkCounter` debounces an EAR dip into one blink on the
  recovery edge (needs `min_frames` below threshold) and keeps a rolling-window
  timestamp deque for blinks-per-minute.
- **KAN-29** — `DrowsinessMonitor` uses wall-clock elapsed time below threshold,
  so a sustained closure (> `--drowsy-seconds`) trips the alert while a quick
  blink does not — independent of camera FPS.
- **KAN-30** — `_draw_eye_landmarks()` overlays the iris + contour mesh;
  `draw_overlay()` shows gaze, blink count/rate, EAR, and a red drowsiness banner.
- **KAN-31** — graceful "No face" state, `try/finally` camera/resource release
  reused from the hand tracker, and the robustness notes below.
- **KAN-32** — `NoBlinkMonitor` (the mirror image of `DrowsinessMonitor`) times
  the seconds since the last blink, driven by `BlinkCounter`'s debounced event
  so it resets on exactly the same blinks. The overlay shows the running
  duration and turns amber with a "STARING?" flag past `--no-blink-seconds`
  (default 10s), a proxy for reduced blink rate / eye strain.

#### Robustness notes (KAN-31)

- **Glasses** — frames and lens glare can shift or drop iris/eyelid landmarks.
  Face Mesh usually still tracks, but EAR gets noisier; nudge `--ear-threshold`
  per-user if blinks over- or under-count.
- **Lighting** — strong side light or backlight lowers landmark confidence.
  Even, front-facing light gives the most stable EAR and gaze. In dim light the
  iris ratio drifts toward "center".
- **Distance / angle** — gaze classification assumes a roughly frontal face;
  large head yaw/pitch biases the horizontal/vertical ratios. The thresholds
  (`GAZE_*` constants) are tuned for a centered, arm's-length webcam pose.
- **EAR threshold** — `0.21` is a common default but is somewhat per-person;
  it's exposed as `--ear-threshold` for both blink and drowsiness logic.

## ⚠️ Important: MediaPipe version

Both apps use the **legacy `mediapipe.solutions` API** (`Hands` and
`FaceMesh`). MediaPipe **0.10.30+ removed** the bundled `mp.solutions` wrappers
in favour of the newer *Tasks* API (`HandLandmarker` / `FaceLandmarker`, which
need a separate `.task` model file). The pinned **`mediapipe==0.10.21`** is the
last release that still ships `solutions`. That build requires **numpy < 2**,
which is why `opencv-python` is pinned to `4.11.0.86` (a numpy<2-compatible
wheel). Don't bump these without porting the code to the Tasks API.
