#!/usr/bin/env python3
"""
eye_tracker.py — real-time gaze, blink, and drowsiness tracker.

Built with OpenCV + MediaPipe Face Mesh (iris refinement). Reads a live
webcam feed, locates the eyes and irises, and reports:

  * gaze direction  (left / right / center / up / down)
  * blink count     (debounced) and blinks-per-minute
  * drowsiness      (eyes closed beyond a duration threshold)
  * no-blink time   (seconds since the last blink; flags staring/eye strain)

This single script implements KAN-24 through KAN-31:

  KAN-24  Set up Face Mesh with iris refinement (refine_landmarks=True).
  KAN-25  Extract left/right eye-contour and iris-center landmarks.
  KAN-26  Estimate gaze from iris-vs-corner and eyelid ratios.
  KAN-27  Compute the Eye Aspect Ratio (EAR) for blink detection.
  KAN-28  Debounce EAR dips into blink events; track blinks/min.
  KAN-29  Flag drowsiness on sustained eye closure (time-based).
  KAN-30  Overlay eye/iris landmarks, gaze, blink count, drowsiness alert.
  KAN-31  Handle the no-face state, document robustness, exit cleanly.
  KAN-32  Track time since the last blink; flag prolonged staring/eye strain.

Usage (run on your own machine — needs camera + display):

    python eye_tracker.py                  # default webcam, mirror view
    python eye_tracker.py --camera 1       # use a different camera index
    python eye_tracker.py --no-flip        # don't mirror the image
    python eye_tracker.py --ear-threshold 0.21
    python eye_tracker.py --self-test      # verify logic, no camera

Press 'q' or Esc in the video window to quit.
"""

import argparse
import collections
import sys
import time

import display

# --- KAN-25: eye + iris landmark indices -----------------------------------
# MediaPipe Face Mesh returns 468 base landmarks; with refine_landmarks=True it
# adds 10 iris points (468-477). Indices below follow MediaPipe's canonical
# face-mesh topology.
#
# EAR uses 6 points per eye in this order:
#   [outer_corner, inner_corner, top1, bottom1, top2, bottom2]
# so horizontal = |p0 - p1| and the two verticals are |p2 - p3|, |p4 - p5|.
LEFT_EYE_EAR = [33, 133, 160, 144, 158, 153]
RIGHT_EYE_EAR = [362, 263, 385, 380, 387, 373]

# Eye corners for the horizontal gaze ratio (outer, inner) and lids for the
# vertical ratio (upper, lower), per eye.
LEFT_EYE_CORNERS = (33, 133)   # outer, inner
RIGHT_EYE_CORNERS = (362, 263)  # inner, outer (mesh ordering)
LEFT_EYE_LIDS = (159, 145)     # upper, lower
RIGHT_EYE_LIDS = (386, 374)    # upper, lower

# Iris landmark rings (centers are the first index of each ring).
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Gaze thresholds on the normalized [0, 1] iris-position ratios. A horizontal
# ratio near 0 means the iris sits toward the outer corner, near 1 the inner.
GAZE_LEFT_MAX = 0.38
GAZE_RIGHT_MIN = 0.62
GAZE_UP_MAX = 0.38
GAZE_DOWN_MIN = 0.62


# --- small geometry helpers ------------------------------------------------
def _xy(landmark):
    return landmark.x, landmark.y


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# --- KAN-27: Eye Aspect Ratio ----------------------------------------------
def eye_aspect_ratio(landmarks, indices):
    """EAR for one eye = (v1 + v2) / (2 * h).

    `indices` is the 6-point layout described above. Returns 0.0 if the
    horizontal span collapses (degenerate / missing landmarks) so callers can
    treat it as a closed/invalid eye without dividing by zero.
    """
    p = [_xy(landmarks[i]) for i in indices]
    horizontal = _dist(p[0], p[1])
    if horizontal == 0:
        return 0.0
    v1 = _dist(p[2], p[3])
    v2 = _dist(p[4], p[5])
    return (v1 + v2) / (2.0 * horizontal)


def average_ear(landmarks):
    """Mean EAR across both eyes — the signal blink/drowsiness logic uses."""
    left = eye_aspect_ratio(landmarks, LEFT_EYE_EAR)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE_EAR)
    return (left + right) / 2.0


# --- KAN-26: gaze estimation -----------------------------------------------
def _ratio(value, low, high):
    """Position of `value` within [low, high], clamped to [0, 1]."""
    span = high - low
    if span == 0:
        return 0.5
    return min(1.0, max(0.0, (value - low) / span))


def _eye_gaze_ratios(landmarks, iris_indices, corners, lids):
    """Return (horizontal, vertical) iris-position ratios in [0, 1] for one eye."""
    cx, cy = _xy(landmarks[iris_indices[0]])  # iris center
    outer = _xy(landmarks[corners[0]])
    inner = _xy(landmarks[corners[1]])
    upper = _xy(landmarks[lids[0]])
    lower = _xy(landmarks[lids[1]])

    h = _ratio(cx, min(outer[0], inner[0]), max(outer[0], inner[0]))
    v = _ratio(cy, min(upper[1], lower[1]), max(upper[1], lower[1]))
    return h, v


def estimate_gaze(landmarks):
    """Classify gaze as left/right/center/up/down from iris position.

    Horizontal takes priority over vertical so a clearly sideways glance reads
    as left/right rather than up/down. Returns 'center' when the iris sits in
    the middle band of both axes.
    """
    lh, lv = _eye_gaze_ratios(landmarks, LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS)
    rh, rv = _eye_gaze_ratios(landmarks, RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS)
    h = (lh + rh) / 2.0
    v = (lv + rv) / 2.0

    if h <= GAZE_LEFT_MAX:
        return "left"
    if h >= GAZE_RIGHT_MIN:
        return "right"
    if v <= GAZE_UP_MAX:
        return "up"
    if v >= GAZE_DOWN_MIN:
        return "down"
    return "center"


# --- KAN-28: blink counting + rate -----------------------------------------
class BlinkCounter:
    """Turn a stream of EAR values into debounced blink events.

    A blink is counted on the *recovery* edge: EAR must dip below `threshold`
    for at least `min_frames` consecutive frames (so a single long blink isn't
    split, and a one-frame jitter isn't counted), then rise back above it.
    Blink timestamps are kept in a rolling window to report blinks-per-minute.
    """

    def __init__(self, threshold=0.21, min_frames=2, window_seconds=60.0):
        self.threshold = threshold
        self.min_frames = max(1, min_frames)
        self.window_seconds = window_seconds
        self.total = 0
        self._below = 0
        self._timestamps = collections.deque()

    def update(self, ear, now=None):
        """Feed one EAR sample. Returns True exactly on the frame a blink ends."""
        if now is None:
            now = time.monotonic()
        blinked = False
        if ear < self.threshold:
            self._below += 1
        else:
            if self._below >= self.min_frames:
                self.total += 1
                self._timestamps.append(now)
                blinked = True
            self._below = 0

        # Drop timestamps outside the rolling window.
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        return blinked

    def blinks_per_minute(self, now=None):
        """Blinks in the rolling window, scaled to a per-minute rate."""
        if now is None:
            now = time.monotonic()
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        if not self._timestamps:
            return 0.0
        return len(self._timestamps) * (60.0 / self.window_seconds)


# --- KAN-29: drowsiness detection ------------------------------------------
class DrowsinessMonitor:
    """Flag drowsiness when EAR stays low for longer than `hold_seconds`.

    Uses wall-clock elapsed time (not a frame count) so the threshold is
    correct regardless of camera FPS, distinguishing a sustained closure from
    a normal quick blink.
    """

    def __init__(self, threshold=0.21, hold_seconds=1.5):
        self.threshold = threshold
        self.hold_seconds = hold_seconds
        self._closed_since = None

    def update(self, ear, now=None):
        """Feed one EAR sample. Returns True while drowsiness is active."""
        if now is None:
            now = time.monotonic()
        if ear < self.threshold:
            if self._closed_since is None:
                self._closed_since = now
            return (now - self._closed_since) >= self.hold_seconds
        self._closed_since = None
        return False


# --- KAN-32: no-blink (staring) duration -----------------------------------
class NoBlinkMonitor:
    """Track how long it's been since the last blink.

    The mirror image of DrowsinessMonitor: instead of eyes staying closed, it
    times eyes staying open without blinking. It's driven by the debounced
    blink event from BlinkCounter rather than raw EAR, so the timer resets on
    exactly the same events that count as a blink. A sustained high duration
    suggests staring / reduced blink rate (eye strain).
    """

    def __init__(self, alert_seconds=10.0):
        self.alert_seconds = alert_seconds
        self.longest = 0.0  # longest no-blink streak this session (a top score)
        self._last_blink_time = None

    def update(self, blinked, now=None):
        """Feed the per-frame blink flag. Returns seconds since the last blink."""
        if now is None:
            now = time.monotonic()
        # Start the clock on the first frame we see, then on every blink.
        if self._last_blink_time is None or blinked:
            self._last_blink_time = now
        duration = now - self._last_blink_time
        # Duration climbs monotonically until a blink resets it, so the running
        # max over every frame is the longest streak's peak.
        if duration > self.longest:
            self.longest = duration
        return duration

    def is_alerting(self, duration):
        """True when the no-blink duration has crossed the alert threshold."""
        return duration >= self.alert_seconds


# --- KAN-24/30: capture + detection loop -----------------------------------
def run(args):
    import cv2
    import mediapipe as mp

    mp_face = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # KAN-24: open the webcam (same pattern as hand_counter.py).
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[FAIL] could not open camera index {args.camera}")
        return 2

    # KAN-24: Face Mesh with iris refinement -> ~478 landmarks incl. irises.
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    blinks = BlinkCounter(threshold=args.ear_threshold)
    drowsy = DrowsinessMonitor(threshold=args.ear_threshold,
                               hold_seconds=args.drowsy_seconds)
    no_blink = NoBlinkMonitor(alert_seconds=args.no_blink_seconds)
    print("Running eye tracker — press 'q' or Esc to quit.")

    window = "Eye Tracker (q/Esc to quit)"
    sized = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[warn] dropped a frame from the camera")
                continue

            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)

            state = None
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                ear = average_ear(landmarks)
                now = time.monotonic()
                blinked = blinks.update(ear, now)
                no_blink_secs = no_blink.update(blinked, now)
                state = {
                    "gaze": estimate_gaze(landmarks),
                    "ear": ear,
                    "blinks": blinks.total,
                    "bpm": blinks.blinks_per_minute(now),
                    "drowsy": drowsy.update(ear, now),
                    "no_blink": no_blink_secs,
                    "staring": no_blink.is_alerting(no_blink_secs),
                    "no_blink_best": no_blink.longest,
                }
                if not args.no_landmarks:
                    _draw_eye_landmarks(cv2, frame, results.multi_face_landmarks[0],
                                        mp_face, mp_draw, mp_styles)

            draw_overlay(cv2, frame, state)

            if not sized:
                display.open_window(cv2, window, frame, args.display_scale)
                sized = True
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # KAN-31: clean exit
                break
    finally:
        # KAN-31: always release hardware and windows.
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()
    return 0


def _draw_eye_landmarks(cv2, frame, face_landmarks, mp_face, mp_draw, mp_styles):
    """KAN-30: draw the iris + eye-contour mesh tesselation."""
    mp_draw.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style(),
    )
    mp_draw.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
    )


def draw_overlay(cv2, frame, state):
    """KAN-30/31: render gaze/blink/drowsiness (or a no-face message)."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

    if state is None:
        # KAN-31: graceful no-face state.
        cv2.putText(frame, "No face", (15, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 200, 255), 3, cv2.LINE_AA)
        return

    cv2.putText(frame, f"Gaze: {state['gaze']}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame,
                f"Blinks: {state['blinks']}  ({state['bpm']:.0f}/min)  EAR {state['ear']:.2f}",
                (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)

    # KAN-32: no-blink duration, turning amber and flagging a stare when it
    # crosses the alert threshold.
    staring = state["staring"]
    nb_text = f"No blink: {state['no_blink']:.1f}s"
    if staring:
        nb_text += "  (STARING?)"
    cv2.putText(frame, nb_text, (w - 340, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 165, 255) if staring else (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Best: {state['no_blink_best']:.1f}s", (w - 340, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # KAN-29/30: prominent drowsiness alert banner.
    if state["drowsy"]:
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 180), -1)
        cv2.putText(frame, "! DROWSINESS ALERT !", (15, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3,
                    cv2.LINE_AA)


# --- self-test (no camera needed) ------------------------------------------
class _LM:
    """Minimal stand-in for a MediaPipe landmark (has .x and .y)."""

    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


def _make_face(open_amount=1.0, gaze_h=0.5, gaze_v=0.5):
    """Build a synthetic landmark list exercising EAR and gaze.

    `open_amount` scales eyelid separation: 1.0 = wide open, 0.0 = shut.
    `gaze_h` / `gaze_v` place the iris center within the eye box (0..1).
    Only the indices the logic reads are positioned meaningfully; the rest are
    filler so the list is long enough to index the iris ring (up to 477).
    """
    lm = [_LM(0.5, 0.5) for _ in range(478)]

    def place_eye(ear_idx, corners, lids, iris, x0):
        outer, inner, t1, b1, t2, b2 = ear_idx
        # Horizontal span of the eye: fixed width centered at x0.
        lm[outer] = _LM(x0 - 0.05, 0.5)
        lm[inner] = _LM(x0 + 0.05, 0.5)
        # Vertical lid separation scales with open_amount.
        half = 0.03 * open_amount
        lm[t1] = _LM(x0 - 0.02, 0.5 - half)
        lm[b1] = _LM(x0 - 0.02, 0.5 + half)
        lm[t2] = _LM(x0 + 0.02, 0.5 - half)
        lm[b2] = _LM(x0 + 0.02, 0.5 + half)
        # Corners / lids used by gaze.
        lm[corners[0]] = lm[outer]
        lm[corners[1]] = lm[inner]
        lm[lids[0]] = _LM(x0, 0.5 - 0.03)
        lm[lids[1]] = _LM(x0, 0.5 + 0.03)
        # Iris center positioned by gaze ratios within the eye box.
        ix = (x0 - 0.05) + gaze_h * 0.10
        iy = (0.5 - 0.03) + gaze_v * 0.06
        for k in iris:
            lm[k] = _LM(ix, iy)

    place_eye(LEFT_EYE_EAR, LEFT_EYE_CORNERS, LEFT_EYE_LIDS, LEFT_IRIS, 0.35)
    place_eye(RIGHT_EYE_EAR, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, RIGHT_IRIS, 0.65)
    return lm


def self_test():
    """Verify EAR, gaze, blink, drowsiness, and no-blink logic. Returns exit code."""
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<34} expected {want}, got {got}")

    # EAR: open eye high, shut eye near zero.
    open_ear = average_ear(_make_face(open_amount=1.0))
    shut_ear = average_ear(_make_face(open_amount=0.0))
    check("EAR open > shut", open_ear > shut_ear, True)
    check("EAR shut below 0.21", shut_ear < 0.21, True)
    check("EAR open above 0.21", open_ear > 0.21, True)

    # Gaze classification.
    check("gaze center", estimate_gaze(_make_face(gaze_h=0.5, gaze_v=0.5)), "center")
    check("gaze left", estimate_gaze(_make_face(gaze_h=0.1)), "left")
    check("gaze right", estimate_gaze(_make_face(gaze_h=0.9)), "right")
    check("gaze up", estimate_gaze(_make_face(gaze_h=0.5, gaze_v=0.1)), "up")
    check("gaze down", estimate_gaze(_make_face(gaze_h=0.5, gaze_v=0.9)), "down")

    # Blink debounce: one dip-and-recover = exactly one blink.
    bc = BlinkCounter(threshold=0.21, min_frames=2)
    t = 0.0
    for ear in [0.30, 0.30, 0.05, 0.05, 0.05, 0.30, 0.30]:
        bc.update(ear, now=t)
        t += 0.05
    check("blink counted once", bc.total, 1)

    # A single-frame dip shorter than min_frames is ignored.
    bc2 = BlinkCounter(threshold=0.21, min_frames=2)
    t = 0.0
    for ear in [0.30, 0.05, 0.30, 0.30]:
        bc2.update(ear, now=t)
        t += 0.05
    check("jitter not counted", bc2.total, 0)

    # Drowsiness: low EAR past hold_seconds fires; a quick blink does not.
    dm = DrowsinessMonitor(threshold=0.21, hold_seconds=1.5)
    check("quick blink not drowsy", dm.update(0.05, now=0.0), False)
    check("sustained closure drowsy", dm.update(0.05, now=2.0), True)
    check("reopen clears drowsy", dm.update(0.30, now=2.1), False)

    # KAN-32: no-blink timer grows while open, resets on a blink, alerts late.
    nb = NoBlinkMonitor(alert_seconds=10.0)
    nb.update(False, now=0.0)                 # first frame anchors the clock
    check("no-blink grows while open", nb.update(False, now=5.0), 5.0)
    check("no-blink resets on blink", nb.update(True, now=6.0), 0.0)
    check("no-blink resumes after reset", nb.update(False, now=6.5), 0.5)
    check("no-blink longest streak retained", nb.longest, 5.0)
    check("no-blink not alerting under threshold", nb.is_alerting(5.0), False)
    check("no-blink alerting over threshold", nb.is_alerting(11.0), True)

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Real-time gaze/blink/drowsiness tracker (OpenCV + MediaPipe)."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--ear-threshold", type=float, default=0.21,
                        help="EAR below this counts as a closed eye (default 0.21).")
    parser.add_argument("--drowsy-seconds", type=float, default=1.5,
                        help="Eyes-closed duration that flags drowsiness (default 1.5s).")
    parser.add_argument("--no-blink-seconds", type=float, default=10.0,
                        help="No-blink duration that flags staring/eye strain (default 10s).")
    parser.add_argument("--no-flip", action="store_true",
                        help="Do not mirror the webcam image.")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="Do not draw the eye/iris mesh overlay.")
    parser.add_argument("--display-scale", type=float, default=1.5,
                        help="Initial window size as a multiple of the camera "
                             "frame (default 1.5). The window is resizable.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run logic checks without a camera, then exit.")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
