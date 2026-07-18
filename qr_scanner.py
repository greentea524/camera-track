#!/usr/bin/env python3
"""
qr_scanner.py — real-time QR / barcode scanner (GitHub issue #33).

Reads from the default webcam, decodes any visible QR codes or barcodes
using pyzbar, and draws the decoded data on the preview window.  Press
ESC or 'q' to quit.

Supported symbologies (via pyzbar / ZBar):
  QR Code, EAN-13, EAN-8, UPC-A, Code-128, Code-39, I2of5, and more.
"""

import sys

import cv2
import numpy as np

import display


# ── colours (BGR) ────────────────────────────────────────────────────
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)

WINDOW = "QR / Barcode Scanner"

# Keep a running log of unique scans so the user can see a history.
_MAX_HISTORY = 10


def _draw_label(frame, text, origin, font_scale=0.6, thickness=2,
                fg=WHITE, bg=BLACK):
    """Draw *text* with a filled background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x, y - th - baseline), (x + tw, y + baseline), bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, fg, thickness)


def _draw_detection(frame, barcode):
    """Draw the bounding polygon and decoded text for one barcode."""
    points = barcode.polygon
    if len(points) == 4:
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, GREEN, 3)
    else:
        # Fallback: use the bounding rect
        x, y, w, h = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 3)

    data = barcode.data.decode("utf-8", errors="replace")
    btype = barcode.type

    # Label above the polygon
    x, y = barcode.rect.left, barcode.rect.top - 10
    _draw_label(frame, f"{btype}: {data}", (x, max(y, 15)), fg=GREEN)

    return f"{btype}: {data}"


def _draw_history(frame, history):
    """Show the last few unique scans in the top-right corner."""
    if not history:
        return
    x = frame.shape[1] - 10
    y = 30
    _draw_label(frame, "Scan History", (x - 160, y - 15), font_scale=0.5,
                fg=YELLOW, bg=BLACK)
    for i, entry in enumerate(history[-_MAX_HISTORY:]):
        label = entry if len(entry) <= 35 else entry[:32] + "..."
        ty = y + 5 + i * 25
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        _draw_label(frame, label, (x - tw - 5, ty), font_scale=0.45,
                    thickness=1, fg=WHITE, bg=(50, 50, 50))


def _draw_hud(frame, fps, count):
    """Draw a minimal heads-up display: FPS and total scans."""
    _draw_label(frame, f"FPS: {fps:.0f}", (10, 25), font_scale=0.55,
                thickness=1, fg=WHITE, bg=(50, 50, 50))
    _draw_label(frame, f"Scans: {count}", (10, 50), font_scale=0.55,
                thickness=1, fg=WHITE, bg=(50, 50, 50))


import argparse
import collections

def run(args):
    """Live capture + QR scanning loop."""
    from pyzbar import pyzbar
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {args.camera}.")
        return 1

    history: list[str] = []
    seen: set[str] = set()
    total_scans = 0

    # FPS tracking
    prev_tick = cv2.getTickCount()
    fps = 0.0

    print("QR / Barcode Scanner starting. Press ESC or 'q' to exit.")

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

        # Decode all barcodes / QR codes in the frame.
        barcodes = pyzbar.decode(frame)

        for bc in barcodes:
            label = _draw_detection(frame, bc)
            total_scans += 1
            if label not in seen:
                seen.add(label)
                history.append(label)

        # HUD
        tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / max(1, tick - prev_tick)
        prev_tick = tick
        _draw_hud(frame, fps, total_scans)
        _draw_history(frame, history)

        # Prompt when no codes are visible.
        if not barcodes:
            h = frame.shape[0]
            _draw_label(frame, "Point camera at a QR code or barcode",
                        (10, h - 20), font_scale=0.6, thickness=1,
                        fg=YELLOW, bg=(50, 50, 50))

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
    """Verify scanner drawing and detection helper functions."""
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<40} expected {want}, got {got}")

    MockBarcode = collections.namedtuple('MockBarcode', ['data', 'type', 'rect', 'polygon'])
    Rect = collections.namedtuple('Rect', ['left', 'top', 'width', 'height'])

    # Test _draw_detection with a mock barcode
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rect = Rect(left=100, top=100, width=50, height=50)
    polygon = [(100, 100), (150, 100), (150, 150), (100, 150)]
    barcode = MockBarcode(data=b"hello-world", type="QRCODE", rect=rect, polygon=polygon)

    try:
        res = _draw_detection(frame, barcode)
        check("detection label correct", res, "QRCODE: hello-world")
        check("frame drawn on", bool(np.any(frame)), True)
    except Exception as e:
        check("draw_detection raised exception", str(e), "no exception")

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] in ("--self-test",):
        return self_test()

    parser = argparse.ArgumentParser(
        description="Real-time QR / barcode scanner (OpenCV + pyzbar).")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0).")
    parser.add_argument("--display-scale", type=float, default=1.5,
                        help="Window scale factor (default 1.5).")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
