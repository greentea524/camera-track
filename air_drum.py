#!/usr/bin/env python3
"""
air_drum.py — virtual drum kit controlled by hand gestures.

Uses MediaPipe Hands to track both hands. The camera view is divided into
drum-pad zones. When a fingertip "strikes" a zone (moves downward past a
velocity threshold), the corresponding drum sound plays.

Issue: #32

Usage:
    python air_drum.py                 # default settings
    python air_drum.py --camera 1      # different camera index
    python air_drum.py --self-test     # verify game logic, no camera

Press 'q' or Esc in the video window to quit.
"""

import argparse
import math
import sys
import threading
import time

import display

# MediaPipe landmark indices
WRIST = 0
INDEX_TIP = 8
MIDDLE_TIP = 12

# Drum pad note frequencies (Hz) — pentatonic-ish for pleasant sounds
DRUM_PADS = [
    {"name": "Hi-Hat",   "freq": 800, "dur": 80,  "color": (0, 220, 220)},
    {"name": "Snare",    "freq": 400, "dur": 100,  "color": (100, 180, 255)},
    {"name": "Tom",      "freq": 300, "dur": 120,  "color": (180, 100, 255)},
    {"name": "Kick",     "freq": 150, "dur": 150,  "color": (255, 100, 100)},
    {"name": "Crash",    "freq": 600, "dur": 90,   "color": (100, 255, 100)},
    {"name": "Ride",     "freq": 700, "dur": 70,   "color": (255, 200, 50)},
]


# ---------------------------------------------------------------------------
# Sound playback (non-blocking, Windows-native, zero dependencies)
# ---------------------------------------------------------------------------

def _play_tone(freq, duration_ms):
    """Play a tone in a background thread so the camera loop doesn't stall."""
    try:
        import winsound
        winsound.Beep(freq, duration_ms)
    except Exception:
        pass  # Silently skip on non-Windows or if audio fails


def play_sound(freq, duration_ms):
    """Fire-and-forget tone playback."""
    t = threading.Thread(target=_play_tone, args=(freq, duration_ms), daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Strike detection
# ---------------------------------------------------------------------------

class StrikeDetector:
    """Detect a downward 'strike' motion of a fingertip.

    Tracks the y-position of a landmark over frames. A strike is registered
    when the tip moves downward faster than `velocity_threshold` (in
    normalized coordinates per second) and then a cooldown prevents
    repeated triggers.
    """

    def __init__(self, velocity_threshold=0.8, cooldown=0.25):
        self.velocity_threshold = velocity_threshold
        self.cooldown = cooldown
        self._prev_y = None
        self._prev_time = None
        self._last_strike = 0.0

    def update(self, y):
        """Feed a new normalized y value. Returns True if a strike occurred."""
        now = time.time()

        if self._prev_y is None:
            self._prev_y = y
            self._prev_time = now
            return False

        dt = now - self._prev_time
        if dt <= 0:
            return False

        velocity = (y - self._prev_y) / dt   # positive = moving downward
        self._prev_y = y
        self._prev_time = now

        if velocity > self.velocity_threshold and (now - self._last_strike) > self.cooldown:
            self._last_strike = now
            return True

        return False

    def reset(self):
        self._prev_y = None
        self._prev_time = None


# ---------------------------------------------------------------------------
# Pad layout
# ---------------------------------------------------------------------------

def get_pad_regions(frame_w, frame_h, num_pads):
    """Divide the lower portion of the frame into evenly-spaced pad rectangles.

    Returns a list of (x1, y1, x2, y2) tuples.
    """
    pad_zone_top = int(frame_h * 0.35)   # pads occupy the lower 65%
    pad_height = frame_h - pad_zone_top
    pad_width = frame_w // num_pads
    regions = []
    for i in range(num_pads):
        x1 = i * pad_width
        x2 = (i + 1) * pad_width
        regions.append((x1, pad_zone_top, x2, frame_h))
    return regions


def find_pad_at(px, py, regions):
    """Return the index of the pad containing pixel (px, py), or -1."""
    for i, (x1, y1, x2, y2) in enumerate(regions):
        if x1 <= px < x2 and y1 <= py < y2:
            return i
    return -1


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _outlined_text(cv2, frame, text, pos, scale, fg, thickness=2):
    """Draw text with a black outline for readability."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness)


def draw_pads(cv2, frame, regions, flash_indices):
    """Draw the drum pad zones on the frame."""
    for i, (x1, y1, x2, y2) in enumerate(regions):
        pad = DRUM_PADS[i]
        color = pad["color"]

        # Flash brighter when hit
        if i in flash_indices:
            bright = tuple(min(255, c + 80) for c in color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bright, -1)
        else:
            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        text_x = x1 + 8
        text_y = y2 - 15
        _outlined_text(cv2, frame, pad["name"], (text_x, text_y), 0.5, (255, 255, 255), 1)


def draw_hud(cv2, frame, hit_count):
    """Render the top HUD bar."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    _outlined_text(cv2, frame, "Air Drums", (20, 35), 0.9, (0, 220, 220))
    _outlined_text(cv2, frame, f"Hits: {hit_count}", (w - 180, 35), 0.7, (200, 200, 200))


def draw_fingertip(cv2, frame, fx, fy, color=(255, 0, 255)):
    """Draw a dot at the fingertip position."""
    if fx is None:
        return
    cv2.circle(frame, (fx, fy), 10, (0, 0, 0), -1)
    cv2.circle(frame, (fx, fy), 8, color, -1)


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(args):
    """Live capture + drum loop."""
    import cv2
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[FAIL] could not open camera index {args.camera}")
        return 2

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # One strike detector per hand (index 0 = first hand, 1 = second)
    detectors = [StrikeDetector() for _ in range(2)]
    hit_count = 0
    flash_until = {}   # pad_index -> time when flash expires

    window = "Air Drums (q/Esc to quit)"
    sized = False
    print("Air Drums starting — press 'q' or Esc to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            regions = get_pad_regions(w, h, len(DRUM_PADS))

            # Determine which pads are currently flashing
            now = time.time()
            active_flashes = {k for k, v in flash_until.items() if now < v}

            if results.multi_hand_landmarks:
                for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks):
                    if hand_idx >= len(detectors):
                        break

                    # Draw hand skeleton
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    # Track index fingertip
                    tip = hand_lm.landmark[INDEX_TIP]
                    fx = int(tip.x * w)
                    fy = int(tip.y * h)

                    # Choose color per hand
                    tip_color = (255, 0, 255) if hand_idx == 0 else (0, 255, 255)
                    draw_fingertip(cv2, frame, fx, fy, tip_color)

                    # Check for strike
                    if detectors[hand_idx].update(tip.y):
                        pad_idx = find_pad_at(fx, fy, regions)
                        if pad_idx >= 0:
                            pad = DRUM_PADS[pad_idx]
                            play_sound(pad["freq"], pad["dur"])
                            hit_count += 1
                            flash_until[pad_idx] = now + 0.15
                            active_flashes.add(pad_idx)

            draw_pads(cv2, frame, regions, active_flashes)
            draw_hud(cv2, frame, hit_count)

            if not sized:
                display.open_window(cv2, window, frame)
                sized = True
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    """Verify drum logic without a camera."""
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<44} expected {want}, got {got}")

    # Pad regions
    regions = get_pad_regions(600, 400, 6)
    check("6 pad regions created", len(regions), 6)
    check("first pad starts at x=0", regions[0][0], 0)
    check("last pad ends at x=600", regions[5][2], 600)

    # find_pad_at
    check("point in first pad", find_pad_at(50, 300, regions), 0)
    check("point in last pad", find_pad_at(550, 300, regions), 5)
    check("point above pad zone", find_pad_at(50, 50, regions), -1)

    # Strike detector
    sd = StrikeDetector(velocity_threshold=0.5, cooldown=0.0)
    sd._prev_y = 0.3
    sd._prev_time = time.time() - 0.05  # 50ms ago
    # Moving from 0.3 to 0.4 in 50ms => velocity = 0.1/0.05 = 2.0 (above threshold)
    check("fast downward = strike", sd.update(0.4), True)

    sd2 = StrikeDetector(velocity_threshold=0.5, cooldown=0.0)
    sd2._prev_y = 0.3
    sd2._prev_time = time.time() - 0.05
    # Moving from 0.3 to 0.29 => upward motion, not a strike
    check("upward motion = no strike", sd2.update(0.29), False)

    # Cooldown
    sd3 = StrikeDetector(velocity_threshold=0.5, cooldown=10.0)
    sd3._prev_y = 0.3
    sd3._prev_time = time.time() - 0.05
    sd3._last_strike = time.time()  # just struck
    check("cooldown prevents re-strike", sd3.update(0.4), False)

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "--self-test":
        return self_test()

    parser = argparse.ArgumentParser(
        description="Virtual air drums controlled by hand gestures.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0).")
    parser.add_argument("--display-scale", type=float, default=1.5,
                        help="Window scale factor (default 1.5).")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
