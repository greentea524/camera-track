#!/usr/bin/env python3
"""
ascii_filter.py — real-time ASCII art filter (GitHub issue #36).

Converts the live camera feed into retro ASCII art characters. Supports
matrix green mode and full colorized mode. Press SPACE to toggle modes.
"""

import argparse
import sys

import cv2
import numpy as np

import display

WINDOW = "ASCII Art Filter"

# Standard character ramp from dark/empty to bright/filled.
# For black background, high pixel values (bright) map to denser characters.
CHAR_RAMP = " .:-=+*#%@"

def map_pixel_to_char(val):
    """Map a grayscale pixel value (0-255) to a character in CHAR_RAMP."""
    idx = int(val * (len(CHAR_RAMP) - 1) / 255)
    return CHAR_RAMP[idx]


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

    # Modes: 0 = Matrix Green, 1 = Full Colorized
    mode = 0
    print("ASCII Filter starting. Press SPACE to toggle modes. Press ESC or 'q' to exit.")

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
            first_frame = False

        # Parameters for the ASCII grid mapping
        # Larger cell size = fewer cv2.putText calls = higher FPS
        cell_w = 10
        cell_h = 14

        cols = w // cell_w
        rows = h // cell_h

        # Grayscale version for mapping brightness to characters
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downsample to the size of the grid
        small_gray = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)

        if mode == 1:
            # Colorized mode needs the downsampled color frame too
            small_color = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)

        # Create a black background canvas
        canvas = np.zeros_like(frame)

        # Draw ASCII characters onto the canvas
        font = cv2.FONT_HERSHEY_MONO_PALETTE if hasattr(cv2, 'FONT_HERSHEY_MONO_PALETTE') else cv2.FONT_HERSHEY_PLAIN
        font_scale = 0.75 if font == cv2.FONT_HERSHEY_PLAIN else 0.45
        thickness = 1

        for r in range(rows):
            for c in range(cols):
                val = small_gray[r, c]
                char = map_pixel_to_char(val)

                # Skip spaces to optimize rendering speed
                if char == ' ':
                    continue

                # Position of character
                x = c * cell_w
                y = r * cell_h + cell_h - 2

                # Set color depending on mode
                if mode == 0:
                    # Matrix green (with brightness matching pixel value)
                    color = (0, int(max(70, val)), 0)
                else:
                    # Colorized mode (use BGR color from the downsampled color frame)
                    color = tuple(map(int, small_color[r, c]))

                cv2.putText(canvas, char, (x, y), font, font_scale, color, thickness)

        # Add HUD
        mode_text = "MATRIX GREEN" if mode == 0 else "COLORIZED"
        _draw_label(canvas, f"MODE: {mode_text} (SPACE TO TOGGLE)", (15, 30), font_scale=0.55, fg=(0, 255, 0))

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
    """Verify char mapping works correctly across 0-255 range."""
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<40} expected {want}, got {got}")

    # Grayscale to ASCII mapping tests
    check("Min value maps to space", map_pixel_to_char(0), " ")
    check("Max value maps to @", map_pixel_to_char(255), "@")
    check("Mid-value maps to middle of ramp", map_pixel_to_char(127), "=")

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] in ("--self-test",):
        return self_test()

    parser = argparse.ArgumentParser(
        description="Real-time live ASCII art camera filter (OpenCV).")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0).")
    parser.add_argument("--display-scale", type=float, default=1.5,
                        help="Window scale factor (default 1.5).")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
