#!/usr/bin/env python3
"""
verify_camera.py — environment smoke-test for KAN-15.

Confirms that the OpenCV + MediaPipe environment is set up correctly:

  1. opencv-python imports and reports its version
  2. mediapipe imports and reports its version
  3. The default webcam (camera index 0) opens and yields a frame

Run on your own machine (needs real camera/display hardware):

    python verify_camera.py            # open camera, show a live preview window
    python verify_camera.py --no-window   # headless: grab one frame, no GUI
    python verify_camera.py --camera 1    # use a different camera index

Press 'q' (or Esc) to close the preview window. Exit code is 0 on success,
non-zero if any import fails or the camera cannot be opened.
"""

import argparse
import sys


def check_imports():
    """Import the two dependencies and print their versions."""
    ok = True
    try:
        import cv2

        print(f"[ok]   opencv-python  {cv2.__version__}")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[FAIL] opencv-python import failed: {exc}")
        ok = False

    try:
        import mediapipe as mp

        print(f"[ok]   mediapipe      {mp.__version__}")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[FAIL] mediapipe import failed: {exc}")
        ok = False

    return ok


def check_camera(index, show_window):
    """Open the webcam at `index`, read frames, optionally show a preview."""
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[FAIL] could not open camera index {index}")
        return False

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[FAIL] camera {index} opened but returned no frame")
            return False

        h, w = frame.shape[:2]
        print(f"[ok]   camera {index} returned a {w}x{h} frame")

        if not show_window:
            return True

        print("       showing live preview — press 'q' or Esc to close")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] dropped a frame from the camera stream")
                break
            cv2.imshow("KAN-15 camera check (q/Esc to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or Esc
                break
        return True
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify the OpenCV + MediaPipe environment and webcam access."
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index to test (default: 0)."
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Headless check: grab a single frame without opening a GUI window.",
    )
    args = parser.parse_args(argv)

    print("Verifying OpenCV + MediaPipe environment...\n")

    if not check_imports():
        print("\nImport check failed. Run: pip install -r requirements.txt")
        return 1

    if not check_camera(args.camera, show_window=not args.no_window):
        print(
            "\nCamera check failed. Make sure a webcam is connected, not in use "
            "by another app, and that this program has camera permission."
        )
        return 2

    print("\nAll checks passed — environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
