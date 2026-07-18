#!/usr/bin/env python3
"""
object_tracker.py — real-time color-based object tracker (GitHub issue #35).

Tracks objects based on their HSV color range. The user can click anywhere on
the video window to pick a color to track in real-time.
"""

import argparse
import collections
import sys

import cv2
import numpy as np

import display

WINDOW = "Color Object Tracker"

# Global variables for mouse callback
clicked_hsv = None

def get_hsv_ranges(hsv_pixel):
    """Calculate lower and upper HSV threshold boundaries around an HSV pixel.

    H is in range 0-180, S and V in range 0-255.
    Returns (lower_bound, upper_bound).
    """
    h, s, v = hsv_pixel

    # Tolerance thresholds
    h_tol = 15
    s_tol = 50
    v_tol = 60

    # Calculate bounds with clamping
    lower_h = (h - h_tol) % 180
    upper_h = (h + h_tol) % 180

    lower_s = max(40, s - s_tol)      # ignore highly desaturated / gray colors
    upper_s = min(255, s + s_tol)

    lower_v = max(40, v - v_tol)      # ignore very dark shadows
    upper_v = min(255, v + v_tol)

    return (lower_h, lower_s, lower_v), (upper_h, upper_s, upper_v)


def mouse_callback(event, x, y, flags, param):
    """Mouse event handler to capture the HSV value of the clicked pixel."""
    global clicked_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param[0]
        # Check boundary just in case
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            pixel_bgr = frame[y, x]
            pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            clicked_hsv = pixel_hsv
            print(f"Tracking new color: BGR={pixel_bgr} -> HSV={pixel_hsv}")


def _draw_label(frame, text, origin, font_scale=0.6, thickness=2,
                 fg=(255, 255, 255), bg=(0, 0, 0)):
    """Draw text with a filled background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x, y - th - baseline), (x + tw, y + baseline), bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, fg, thickness)


def run(args):
    """Main video loop."""
    global clicked_hsv
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {args.camera}.")
        return 1

    # Default to track a bright red color before clicking
    clicked_hsv = np.array([0, 200, 200], dtype=np.uint8) # Default high-sat red
    trail = collections.deque(maxlen=24)

    print("Object Tracker starting. Click on the video window to track that color. Press ESC or 'q' to exit.")

    first_frame = True
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Ignoring empty camera frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if first_frame:
            display.open_window(cv2, WINDOW, frame, scale=args.display_scale)
            # Pass mutable frame reference for coordinates retrieval in callback
            cv2.setMouseCallback(WINDOW, mouse_callback, param=[frame])
            first_frame = False

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get current tracking thresholds
        lower, upper = get_hsv_ranges(clicked_hsv)

        # Handle Hue wrapping around the 0-180 boundary
        if lower[0] <= upper[0]:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        else:
            # Hue wrapped around, combine two ranges: [0, upper_h] and [lower_h, 180]
            lower1 = (0, lower[1], lower[2])
            upper1 = (upper[0], upper[1], upper[2])
            lower2 = (lower[0], lower[1], lower[2])
            upper2 = (180, upper[1], upper[2])
            mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask1, mask2)

        # Clean up mask (reduce noise)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tracked_center = None
        if contours:
            # Find largest contour by area
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 400: # Min area to prevent tracking tiny dust particles
                # Calculate minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(largest)
                tracked_center = (int(x), int(y))

                # Draw track indicators
                cv2.circle(frame, tracked_center, int(radius), (0, 255, 255), 2)
                cv2.circle(frame, tracked_center, 5, (0, 0, 255), -1)

        # Add to trail history
        if tracked_center:
            trail.append(tracked_center)
        else:
            if len(trail) > 0:
                trail.popleft() # slowly fade out the trail when tracking is lost

        # Draw tracking trail lines
        for i in range(1, len(trail)):
            if trail[i - 1] is None or trail[i] is None:
                continue
            thickness = int(np.sqrt(args.display_scale * 8) * float(i) / len(trail)) + 1
            cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), thickness)

        # HUD
        _draw_label(frame, "CLICK ANYWHERE TO TRACK THAT COLOR", (15, 30), font_scale=0.55, fg=(0, 255, 255))
        h_val, s_val, v_val = clicked_hsv
        _draw_label(frame, f"Tracking HSV: {h_val}, {s_val}, {v_val}", (15, 60), font_scale=0.5)

        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def self_test():
    """Verify boundaries generator works correctly and handles hue wrapping."""
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<40} expected {want}, got {got}")

    # Standard mid-range HSV
    l_mid, u_mid = get_hsv_ranges((90, 120, 150))
    check("lower H mid-range", l_mid[0], 75)
    check("upper H mid-range", u_mid[0], 105)
    check("lower S mid-range", l_mid[1], 70)
    check("lower V mid-range", l_mid[2], 90)

    # Low hue wrap test (H=5)
    l_wrap, u_wrap = get_hsv_ranges((5, 120, 150))
    check("lower wrapped H", l_wrap[0], 170) # 5 - 15 = -10 -> 170
    check("upper wrapped H", u_wrap[0], 20)

    # Bounds clamping test
    l_clamp, u_clamp = get_hsv_ranges((90, 10, 10))
    check("S clamped lower bound", l_clamp[1], 40)
    check("V clamped lower bound", l_clamp[2], 40)

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] in ("--self-test",):
        return self_test()

    parser = argparse.ArgumentParser(
        description="Real-time click-to-pick HSV object tracker (OpenCV).")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0).")
    parser.add_argument("--display-scale", type=float, default=1.5,
                        help="Window scale factor (default 1.5).")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
