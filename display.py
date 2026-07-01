#!/usr/bin/env python3
"""
display.py — shared OpenCV preview-window helpers for the camera-track apps.

By default `cv2.imshow` creates a fixed, non-resizable window sized to the
camera frame (often 640x480). These helpers instead open a *resizable* window
(drag it to any size) at a configurable multiple of the frame size, so the
preview isn't stuck at the raw capture resolution.
"""

import sys


def window_size(frame_w, frame_h, scale):
    """Return the (width, height) for a window at `scale` x the frame size.

    Pure and hardware-free so it can be unit-tested. Never returns a dimension
    below 1 pixel.
    """
    return max(1, int(frame_w * scale)), max(1, int(frame_h * scale))


def open_window(cv2, name, frame, scale=1.5):
    """Create a resizable preview window sized to `scale` x the frame.

    Call this once the first frame is available. `cv2` is passed in so this
    module stays import-light (no OpenCV import at load time), matching how the
    apps import cv2 lazily inside their run loops.
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    h, w = frame.shape[:2]
    width, height = window_size(w, h, scale)
    cv2.resizeWindow(name, width, height)


def _self_test():
    ok = True

    def check(desc, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"[{'ok  ' if good else 'FAIL'}] {desc:<30} expected {want}, got {got}")

    check("1.5x of 640x480", window_size(640, 480, 1.5), (960, 720))
    check("1.0x identity", window_size(640, 480, 1.0), (640, 480))
    check("2.0x of 320x240", window_size(320, 240, 2.0), (640, 480))
    check("floors below 1px to 1", window_size(10, 10, 0.0), (1, 1))

    print("\nSelf-test", "passed." if ok else "FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_self_test())
