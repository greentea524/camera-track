#!/usr/bin/env python3
"""
hand_counter.py — real-time finger counter using OpenCV + MediaPipe.

Counts extended fingers across up to two hands (range 0-10) from a live
webcam feed, drawing the hand skeleton and the running count on the video.

This single script implements KAN-16 through KAN-21:

  KAN-16  Capture webcam feed (cv2.VideoCapture) and loop over frames.
  KAN-17  Run MediaPipe Hands to get 21 landmarks per detected hand.
  KAN-18  Decide per-finger extended/not (thumb by x, others by y).
  KAN-19  Overlay the finger count and draw the landmark skeleton.
  KAN-20  Detect up to two hands and sum fingers for a 0-10 total.
  KAN-21  Handle the no-hand state, debounce the count to stop flicker,
          and exit cleanly on 'q' while releasing the camera.

Usage (run on your own machine — needs camera + display):

    python hand_counter.py                 # default: up to 2 hands, mirror view
    python hand_counter.py --camera 1      # use a different camera index
    python hand_counter.py --max-hands 1   # single-hand mode (0-5)
    python hand_counter.py --no-flip       # don't mirror the image
    python hand_counter.py --no-debounce   # show raw per-frame count
    python hand_counter.py --self-test     # verify finger logic, no camera

Press 'q' or Esc in the video window to quit.
"""

import argparse
import collections
import sys

# MediaPipe Hands landmark indices (per the 21-point hand model).
# Fingertip landmarks and the PIP joint two steps below each tip.
FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
THUMB_TIP = 4
THUMB_IP = 3  # interphalangeal joint, just below the thumb tip


# --- KAN-18: finger-up detection -------------------------------------------
def fingers_up(landmarks, handedness_label):
    """Return a list of 5 booleans: is [thumb, index, middle, ring, pinky] up?

    `landmarks` is a sequence of objects exposing normalized .x / .y in image
    coordinates (y grows downward). `handedness_label` is MediaPipe's "Left"
    or "Right" for the same processed frame, used to orient the thumb.
    """
    states = []

    # Thumb points sideways, so extension is judged on x, not y. The direction
    # depends on which hand it is, so we key off MediaPipe's handedness label.
    if handedness_label == "Right":
        states.append(landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x)
    else:  # "Left"
        states.append(landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x)

    # The other four fingers are extended when the tip sits above (smaller y)
    # the PIP joint below it.
    for name in ("index", "middle", "ring", "pinky"):
        tip = landmarks[FINGER_TIPS[name]]
        pip = landmarks[FINGER_PIPS[name]]
        states.append(tip.y < pip.y)

    return states


# --- KAN-21: debounce / smoothing ------------------------------------------
class CountStabilizer:
    """Smooth a stream of integer counts by taking the mode of a short window.

    This prevents the displayed total from flickering when a borderline finger
    crosses the up/down threshold between frames.
    """

    def __init__(self, window=5):
        self._history = collections.deque(maxlen=max(1, window))

    def update(self, value):
        self._history.append(value)
        # Most common value in the recent window; ties resolve to the most
        # recently most-frequent, which is stable enough for display.
        return collections.Counter(self._history).most_common(1)[0][0]


def count_hands(results, hands_module):
    """Sum extended fingers across all detected hands.

    Returns (total, per_hand) where per_hand is a list of
    (handedness_label, fingers_up_count) for any hands found.
    """
    total = 0
    per_hand = []
    if not results.multi_hand_landmarks:
        return total, per_hand

    handedness_list = results.multi_handedness or []
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        if idx < len(handedness_list):
            label = handedness_list[idx].classification[0].label
        else:
            label = "Right"
        up = fingers_up(hand_landmarks.landmark, label)
        n = sum(up)
        total += n
        per_hand.append((label, n))
    return total, per_hand


def run(args):
    """Live capture + detection loop (KAN-16, 17, 19, 20, 21)."""
    import cv2
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # KAN-16: open the webcam.
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[FAIL] could not open camera index {args.camera}")
        return 2

    # KAN-17 / KAN-20: configure MediaPipe Hands for up to N hands.
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    stabilizer = CountStabilizer(window=args.window)
    print("Running finger counter — press 'q' or Esc to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[warn] dropped a frame from the camera")
                continue

            # Mirror for a natural selfie view. We process the flipped frame so
            # landmarks and handedness stay consistent with what's displayed.
            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            # KAN-17: MediaPipe expects RGB.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            # KAN-18 / KAN-20: count fingers across detected hands.
            raw_total, per_hand = count_hands(results, mp_hands)

            # KAN-21: debounce unless disabled.
            total = raw_total if args.no_debounce else stabilizer.update(raw_total)

            # KAN-19: draw the hand skeletons.
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            draw_overlay(cv2, frame, total, per_hand)

            cv2.imshow("Finger Counter (q/Esc to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # KAN-21: clean exit
                break
    finally:
        # KAN-21: always release hardware and windows.
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
    return 0


def draw_overlay(cv2, frame, total, per_hand):
    """KAN-19/21: render the count (or a no-hand message) onto the frame."""
    h, w = frame.shape[:2]

    # Header band for readability.
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)

    if per_hand:
        text = f"Fingers: {total}"
        color = (0, 255, 0)
    else:
        # KAN-21: graceful no-hand state.
        text = "No hand"
        color = (0, 200, 255)

    cv2.putText(
        frame, text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA
    )

    # Per-hand breakdown in the corner when hands are present.
    for i, (label, n) in enumerate(per_hand):
        cv2.putText(
            frame,
            f"{label}: {n}",
            (w - 200, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


# --- self-test (no camera needed) ------------------------------------------
class _LM:
    """Minimal stand-in for a MediaPipe landmark (has .x and .y)."""

    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y


def _make_hand(thumb=False, index=False, middle=False, ring=False, pinky=False,
               right=True):
    """Build a synthetic 21-landmark hand with the requested fingers extended.

    Uses an upright palm: 'up' fingers get a tip above (smaller y) their PIP.
    The thumb extends outward in x according to handedness.
    """
    lm = [_LM(0.5, 0.5) for _ in range(21)]

    # Thumb: tip at index 4, IP joint at index 3.
    if right:
        # Right hand extended thumb points to smaller x.
        lm[THUMB_IP] = _LM(0.50, 0.5)
        lm[THUMB_TIP] = _LM(0.40 if thumb else 0.55, 0.5)
    else:
        lm[THUMB_IP] = _LM(0.50, 0.5)
        lm[THUMB_TIP] = _LM(0.60 if thumb else 0.45, 0.5)

    wants = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, up in wants.items():
        pip_y = 0.5
        lm[FINGER_PIPS[name]] = _LM(0.5, pip_y)
        # Extended => tip above PIP (smaller y); curled => tip below PIP.
        lm[FINGER_TIPS[name]] = _LM(0.5, pip_y - 0.1 if up else pip_y + 0.1)
    return lm


def self_test():
    """Verify fingers_up against hand-crafted poses. Returns process exit code."""
    cases = [
        # (description, kwargs, handedness, expected_count)
        ("fist (right)", dict(right=True), "Right", 0),
        ("open palm (right)",
         dict(thumb=True, index=True, middle=True, ring=True, pinky=True,
              right=True), "Right", 5),
        ("peace sign (right)",
         dict(index=True, middle=True, right=True), "Right", 2),
        ("thumbs up (right)", dict(thumb=True, right=True), "Right", 1),
        ("open palm (left)",
         dict(thumb=True, index=True, middle=True, ring=True, pinky=True,
              right=False), "Left", 5),
        ("thumb only (left)", dict(thumb=True, right=False), "Left", 1),
        ("three fingers (right)",
         dict(index=True, middle=True, ring=True, right=True), "Right", 3),
    ]

    all_ok = True
    for desc, kwargs, label, expected in cases:
        hand = _make_hand(**kwargs)
        states = fingers_up(hand, label)
        got = sum(states)
        ok = got == expected
        all_ok = all_ok and ok
        flag = "ok  " if ok else "FAIL"
        print(f"[{flag}] {desc:<22} expected {expected}, got {got}  {states}")

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(
        description="Real-time finger counter (OpenCV + MediaPipe). 'q' to quit."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument(
        "--max-hands", type=int, default=2, help="Max hands to detect (default 2)."
    )
    parser.add_argument(
        "--window", type=int, default=5, help="Debounce window size in frames (default 5)."
    )
    parser.add_argument(
        "--no-flip", action="store_true", help="Do not mirror the webcam image."
    )
    parser.add_argument(
        "--no-debounce", action="store_true", help="Show the raw per-frame count."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run finger-detection logic checks without a camera, then exit.",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
