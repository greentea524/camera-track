#!/usr/bin/env python3
"""
sketch_filter.py — real-time color and grayscale pencil sketch filter (GitHub issue #37).

Applies image processing filters to convert a camera feed into a pencil drawing.
Supports both classic grayscale pencil sketch and colorized sketch. Press SPACE
to toggle modes.
"""

import argparse
import sys

import cv2
import numpy as np

import display

WINDOW = "Pencil Sketch Filter"

def pencil_sketch(frame):
    """Generate a grayscale pencil sketch from a BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    # Perform color dodge blend
    sketch = cv2.divide(gray, 255 - blurred, scale=256.0)
    return sketch


def color_sketch(frame, sketch_gray):
    """Generate a colorized pencil sketch by blending the grayscale sketch with the color frame."""
    # Convert grayscale sketch to 3-channel
    sketch_bgr = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
    # Multiply the original color by the normalized sketch intensity
    result = cv2.multiply(frame, sketch_bgr, scale=1.0/255.0)
    return result


def _draw_label(frame, text, origin, font_scale=0.6, thickness=2,
                 fg=(255, 255, 255), bg=(0, 0, 0)):
    """Draw text with a filled background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x, y - th - baseline), (x + tw, y + baseline), bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, fg, thickness)


def run(args):
    """Main loop."""
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {args.camera}.")
        return 1

    # Modes: 0 = Grayscale Sketch, 1 = Color Sketch
    mode = 0
    print("Sketch Filter starting. Press SPACE to toggle modes. Press ESC or 'q' to exit.")

    first_frame = True
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Ignoring empty camera frame.")
            break

        frame = cv2.flip(frame, 1)

        if first_frame:
            display.open_window(cv2, WINDOW, frame, scale=args.display_scale)
            first_frame = False

        # Apply sketch pipeline
        sketch_gray = pencil_sketch(frame)

        if mode == 0:
            # Grayscale Sketch: Convert to BGR so we can display it consistently
            canvas = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
        else:
            # Color Sketch
            canvas = color_sketch(frame, sketch_gray)

        # Add HUD
        mode_text = "GRAYSCALE PENCIL" if mode == 0 else "COLOR SKETCH"
        _draw_label(canvas, f"MODE: {mode_text} (SPACE TO TOGGLE)", (15, 30), font_scale=0.55, fg=(0, 150, 255))

        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key == ord(" "):
            mode = (mode + 1) % 2
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def self_test():
    """Verify sketch processing results in correct shapes and types."""
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<40} expected {want}, got {got}")

    # Create dummy colorful frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 100, 50), -1)

    try:
        sketch_g = pencil_sketch(frame)
        check("sketch shape matches input", sketch_g.shape, (480, 640))
        check("sketch is single channel", len(sketch_g.shape), 2)

        sketch_c = color_sketch(frame, sketch_g)
        check("color sketch shape matches BGR", sketch_c.shape, (480, 640, 3))
        check("color sketch is 3-channel", len(sketch_c.shape), 3)

        # Check drawing result (should have some sketches or colors)
        check("sketch contains drawing values", bool(np.any(sketch_g)), True)
    except Exception as e:
        check("sketch function raised exception", str(e), "no exception")

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] in ("--self-test",):
        return self_test()

    parser = argparse.ArgumentParser(
        description="Real-time camera color / pencil sketch filter (OpenCV).")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0).")
    parser.add_argument("--display-scale", type=float, default=1.5,
                        help="Window scale factor (default 1.5).")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
