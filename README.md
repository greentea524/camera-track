# camera-track — webcam vision toys

Three real-time webcam apps built with **OpenCV** + **MediaPipe**:

- **`hand_counter.py`** — counts extended fingers (0–10) across up to two
  hands, overlaying the count and hand skeleton. Implements **KAN-15**
  (environment) and **KAN-16 → KAN-21**.
- **`eye_tracker.py`** — tracks gaze direction, blinks (count + rate),
  drowsiness, and no-blink/staring time from the eyes/irises via Face Mesh.
  Implements **KAN-24 → KAN-32**.
- **`rps_game.py`** — play Rock-Paper-Scissors against the computer with hand
  gestures, with a countdown and score tracker. Implements **KAN-33 / KAN-34**.

Or launch any of them from a single menu with **`main.py`** (KAN-35).

## Files

| File               | Purpose                                                   |
| ------------------ | --------------------------------------------------------- |
| `main.py`          | Menu launcher for all the apps (KAN-35).                  |
| `hand_counter.py`  | The finger-counting app (KAN-16–21).                      |
| `eye_tracker.py`   | Gaze / blink / drowsiness / no-blink tracker (KAN-24–32). |
| `rps_game.py`      | Rock-Paper-Scissors gesture game (KAN-33–34).             |
| `verify_camera.py` | Environment/webcam smoke-test (KAN-15).                   |
| `display.py`       | Shared resizable-preview-window helper.                   |
| `requirements.txt` | Pinned dependencies.                                      |

Every app opens a **resizable** preview window (drag to resize) at 1.5× the
camera frame by default; tune the initial size with `--display-scale`.

## Setup

A virtualenv (`.venv/`) is already created with everything installed. To
recreate it from scratch:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

The apps need real camera + display hardware, so run them on your own machine.
The easiest way is the menu launcher:

```powershell
.\.venv\Scripts\python.exe main.py             # pick an app from a menu
```

Or run any app directly (equivalent to picking it from the menu):

```powershell
.\.venv\Scripts\python.exe hand_counter.py     # finger counter
.\.venv\Scripts\python.exe eye_tracker.py      # gaze / blink / drowsiness
.\.venv\Scripts\python.exe rps_game.py         # Rock-Paper-Scissors
```

Press **`q`** or **Esc** in the window to quit (returns to the menu when
launched via `main.py`).

## Troubleshooting

If the camera preview does not appear, try these steps first:

- Run `verify_camera.py` to confirm the webcam is detected.
- If your default camera is wrong, try a different index with `--camera 1` or
  `--camera 2`.
- On Windows, close other apps that may be using the webcam, then re-run the
  app.
- If the view is dark or unstable, improve lighting and make sure the face or
  hand is clearly visible in the frame.

### `main.py` launcher (KAN-35)

`main.py` imports the apps and dispatches to the one you pick — no camera is
touched until an app actually starts, so the menu is instant. You can also skip
the menu and launch by name or number, passing flags straight through to the app:

```powershell
.\.venv\Scripts\python.exe main.py eyes             # jump to the eye tracker
.\.venv\Scripts\python.exe main.py 3 --seed 0       # RPS with a fixed RNG seed
.\.venv\Scripts\python.exe main.py --self-test      # verify menu wiring, no camera
```

App keywords: `hands`, `eyes`, `rps`, `check`.

### `hand_counter.py` options

| Flag                | Default | Description                                      |
| ------------------- | ------- | ------------------------------------------------ |
| `--camera N`        | `0`     | Webcam index                                     |
| `--max-hands N`     | `2`     | Max hands to detect (use `1` for 0–5)            |
| `--window N`        | `5`     | Debounce window in frames                        |
| `--no-flip`         | off     | Don't mirror the image                           |
| `--no-debounce`     | off     | Show the raw per-frame count                     |
| `--display-scale F` | `1.5`   | Initial window size vs. camera frame (resizable) |
| `--self-test`       | —       | Verify finger-detection logic without a camera   |

### `eye_tracker.py` options

| Flag                   | Default | Description                                         |
| ---------------------- | ------- | --------------------------------------------------- |
| `--camera N`           | `0`     | Webcam index                                        |
| `--ear-threshold F`    | `0.21`  | EAR below this counts as a closed eye               |
| `--drowsy-seconds F`   | `1.5`   | Eyes-closed duration that flags drowsiness          |
| `--no-blink-seconds F` | `10.0`  | No-blink duration that flags staring/eye strain     |
| `--no-flip`            | off     | Don't mirror the image                              |
| `--no-landmarks`       | off     | Don't draw the eye/iris mesh overlay                |
| `--display-scale F`    | `1.5`   | Initial window size vs. camera frame (resizable)    |
| `--self-test`          | —       | Verify gaze/blink/drowsiness logic without a camera |

### `rps_game.py` options

| Flag                | Default | Description                                        |
| ------------------- | ------- | -------------------------------------------------- |
| `--camera N`        | `0`     | Webcam index                                       |
| `--countdown F`     | `3.0`   | Countdown seconds before each round                |
| `--result-hold F`   | `2.5`   | Seconds to show each round's result                |
| `--window N`        | `5`     | Frames of gesture history used to lock in          |
| `--seed N`          | —       | Seed the computer's moves for reproducibility      |
| `--manual`          | off     | Hold each result until SPACE starts the next round |
| `--no-flip`         | off     | Don't mirror the image                             |
| `--display-scale F` | `1.5`   | Initial window size vs. camera frame (resizable)   |
| `--self-test`       | —       | Verify game logic without a camera                 |

Press **SPACE** to start the next round (skips the result hold in auto mode; the
only way to advance in `--manual` mode).

Each app's `--self-test` mode validates its detection logic against synthetic
landmarks and needs no hardware — handy for CI or a quick sanity check. For the
eye tracker it checks EAR (open vs. shut), gaze classification, single-blink
debounce, the drowsiness duration threshold, and the no-blink timer. For the RPS
game it checks gesture classification, the winner table, and the round state
machine (scoring once per round, replaying on an unclear gesture).

## How it maps to the tickets

- **KAN-16** — `cv2.VideoCapture` opens the webcam; the main loop reads frames.
- **KAN-17** — `mediapipe` Hands returns 21 landmarks per detected hand.
- **KAN-18** — `fingers_up()` judges each finger: non-thumb fingers by tip-vs-PIP
  _y_, the thumb by _x_ (orientation taken from MediaPipe's handedness label).
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
  (default 10s), a proxy for reduced blink rate / eye strain. It also tracks the
  session's longest no-blink streak (`.longest`) and shows it as a green
  "Best" top score.

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

### RPS game — `rps_game.py` (KAN-33)

- **Gesture detection** — `classify_gesture()` reuses `hand_counter.fingers_up()`
  and looks only at the four fingers (thumb ignored, as it's the noisiest): all
  curled = rock, all extended = paper, index+middle up with ring+pinky down =
  scissors, anything else = unknown.
- **Round loop** — `RPSGame` is a `wait-for-hand -> countdown -> lock-in ->
result -> repeat` state machine. It holds in a "Show your hand to start!"
  state until a hand is detected, so the countdown never runs on an empty frame,
  and each round returns there to re-arm. At the end of the countdown the
  gesture is locked in as the mode of the last few frames (`GestureStabilizer`)
  so a hand mid-transition isn't misread; an unclear/absent gesture replays the
  round instead of scoring.
- **Winner + score** — `decide_winner()` is the standard beats-table;
  `ScoreBoard` tallies wins/losses/draws.
- **Overlay** — `draw_game_overlay()` shows the score band, a big countdown
  number, the live detected gesture, and the round result (You vs. CPU + outcome).
- **Determinism** — the computer's moves come from an injectable `random.Random`,
  so `--seed` (and the self-test) make rounds reproducible.
- **Replay (KAN-34)** — `replay()` (bound to SPACE) re-arms the game from the
  result phase. By default rounds auto-advance after `--result-hold`; `--manual`
  disables that so each result holds until you press SPACE.

## ⚠️ Important: MediaPipe version

All three apps use the **legacy `mediapipe.solutions` API** (`Hands` and
`FaceMesh`; `rps_game.py` reuses the hand pipeline). MediaPipe **0.10.30+
removed** the bundled `mp.solutions` wrappers
in favour of the newer _Tasks_ API (`HandLandmarker` / `FaceLandmarker`, which
need a separate `.task` model file). The pinned **`mediapipe==0.10.21`** is the
last release that still ships `solutions`. That build requires **numpy < 2**,
which is why `opencv-python` is pinned to `4.11.0.86` (a numpy<2-compatible
wheel). Don't bump these without porting the code to the Tasks API.
