# camera-track — hand finger counter

Real-time finger counter built with **OpenCV** + **MediaPipe Hands**. It reads
your webcam, detects up to two hands, decides which fingers are extended, and
overlays the running count (0–10) with the hand skeleton drawn on the video.

Implements **KAN-15** (environment) and **KAN-16 → KAN-21** (the app).

## Files

| File | Purpose |
|------|---------|
| `hand_counter.py` | The finger-counting app (KAN-16–21). |
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

Needs real camera + display hardware, so run it on your own machine:

```powershell
.\.venv\Scripts\python.exe hand_counter.py
```

Press **`q`** or **Esc** in the window to quit.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--camera N` | `0` | Webcam index |
| `--max-hands N` | `2` | Max hands to detect (use `1` for 0–5) |
| `--window N` | `5` | Debounce window in frames |
| `--no-flip` | off | Don't mirror the image |
| `--no-debounce` | off | Show the raw per-frame count |
| `--self-test` | — | Verify finger-detection logic without a camera |

The `--self-test` mode checks the finger math against known poses (fist, open
palm, peace sign, thumbs-up, etc.) and needs no hardware — handy for CI or a
quick sanity check.

## How it maps to the tickets

- **KAN-16** — `cv2.VideoCapture` opens the webcam; the main loop reads frames.
- **KAN-17** — `mediapipe` Hands returns 21 landmarks per detected hand.
- **KAN-18** — `fingers_up()` judges each finger: non-thumb fingers by tip-vs-PIP
  *y*, the thumb by *x* (orientation taken from MediaPipe's handedness label).
- **KAN-19** — `cv2.putText` overlays the count; `drawing_utils` draws the skeleton.
- **KAN-20** — `max_num_hands=2`; fingers are summed across both hands (0–10).
- **KAN-21** — graceful "No hand" state, a `CountStabilizer` debounce to stop
  flicker, and a `try/finally` that releases the camera and windows on `q`.

## ⚠️ Important: MediaPipe version

This app uses the **legacy `mediapipe.solutions.Hands` API**. MediaPipe
**0.10.30+ removed** the bundled `mp.solutions` wrappers in favour of the newer
*Tasks* API (`HandLandmarker`, which needs a separate `.task` model file). The
pinned **`mediapipe==0.10.21`** is the last release that still ships
`solutions`. That build requires **numpy < 2**, which is why `opencv-python` is
pinned to `4.11.0.86` (a numpy<2-compatible wheel). Don't bump these without
porting the code to the Tasks API.
