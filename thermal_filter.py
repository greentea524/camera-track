#!/usr/bin/env python3
"""
thermal_filter.py — real-time thermal/infrared imaging simulation (GitHub issue #38).

Simulates a thermal camera HUD by applying the JET colormap to a grayscale camera
feed. Renders a thermal colorbar scale and a target crosshair.
"""

import argparse
import sys

import cv2
import numpy as np

import display

WINDOW = "Thermal Simulation Filter"

def apply_thermal_filter(frame):
    """Convert BGR frame to thermal simulation view using COLORMAP_JET."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal


def _draw_colorbar(frame):
    """Draw a thermal colorbar scale on the right side of the frame."""
    h, w = frame.shape[:2]

    # Target colorbar size and coordinates
    bar_w = 20
    bar_h = h // 2
    x_offset = w - 50
    y_offset = (h - bar_h) // 2

    # Create a vertical color gradient (from 255 down to 0)
    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1)
    # Replicate horizontally to match bar width
    gradient_bar = np.repeat(gradient, bar_w, axis=1)

    # Map the gradient to JET colors
    colored_bar = cv2.applyColorMap(gradient_bar, cv2.COLORMAP_JET)

    # Insert into the frame
    frame[y_offset:y_offset + bar_h, x_offset:x_offset + bar_w] = colored_bar

    # Draw border around the bar
    cv2.rectangle(frame, (x_offset, y_offset), (x_offset + bar_w, y_offset + bar_h), (255, 255, 255), 1)

    # Draw text scale labels next to the bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (255, 255, 255)

    cv2.putText(frame, "37 C", (x_offset + bar_w + 5, y_offset + 10), font, font_scale, color, thickness)
    cv2.putText(frame, "28 C", (x_offset + bar_w + 5, y_offset + bar_h // 2), font, font_scale, color, thickness)
    cv2.putText(frame, "18 C", (x_offset + bar_w + 5, y_offset + bar_h - 5), font, font_scale, color, thickness)


def _draw_crosshairs(frame):
    """Draw a target crosshair and bounding indicators in the center of the frame."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Draw center crosshair
    color = (0, 255, 255) # yellow
    cv2.drawMarker(frame, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)

    # Draw corner bracket lines
    bracket_len = 25
    margin = 80
    x1, x2 = cx - margin, cx + margin
    y1, y2 = cy - margin, cy + margin

    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, 1)
    cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, 1)

    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, 1)
    cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, 1)

    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, 1)
    cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, 1)

    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, 1)
    cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, 1)


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

    print("Thermal simulation filter starting. Press ESC or 'q' to exit.")

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

        # Generate thermal camera frame
        thermal_frame = apply_thermal_filter(frame)

        # Draw HUD elements
        _draw_colorbar(thermal_frame)
        _draw_crosshairs(thermal_frame)
        _draw_label(thermal_frame, "THERMAL IR CAM SIMULATION", (15, 30), font_scale=0.55, fg=(0, 0, 255))

        cv2.imshow(WINDOW, thermal_frame)

        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def self_test():
    """Verify applying the thermal colormap generator output."""
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<40} expected {want}, got {got}")

    # Create dummy black frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        thermal = apply_thermal_filter(frame)
        check("thermal shape matches input", thermal.shape, (480, 640, 3))
        check("thermal output has 3 channels", len(thermal.shape), 3)

        # Thermal image shouldn't be black since JET colormap maps 0 to dark blue (BGR: [128, 0, 0] or similar)
        check("thermal maps black to non-black", bool(np.any(thermal)), True)
    except Exception as e:
        check("thermal function raised exception", str(e), "no exception")

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] in ("--self-test",):
        return self_test()

    parser = argparse.ArgumentParser(
        description="Real-time thermal/infrared camera HUD simulation (OpenCV).")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0).")
    parser.add_argument("--display-scale", type=float, default=1.5,
                        help="Window scale factor (default 1.5).")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
