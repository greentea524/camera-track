import argparse
import math
import sys
import time

import cv2
import numpy as np

import display

# MediaPipe Solutions
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def calculate_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def calculate_mood(face_landmarks):
    """Calculate mood score (0-100) and state from face landmarks.

    Uses temple width (234, 454), mouth width (61, 291), and inner eyebrow dist (107, 336).
    Returns (mood_score, mood_text, metrics_dict).
    """
    # Face width (temples: 234, 454) for normalization
    face_width = calculate_distance(face_landmarks[234], face_landmarks[454])
    mouth_width = calculate_distance(face_landmarks[61], face_landmarks[291])
    eyebrow_dist = calculate_distance(face_landmarks[107], face_landmarks[336])

    mood_score = 50  # Default neutral
    mood_text = "Neutral"

    if face_width > 0:
        mouth_ratio = mouth_width / face_width
        eyebrow_ratio = eyebrow_dist / face_width

        if mouth_ratio > 0.40:
            # Smiling (Happy)
            mood_score = 50 + ((mouth_ratio - 0.40) / 0.10) * 50
            mood_text = "Happy"
        elif eyebrow_ratio < 0.20:
            # Frowning (Mad/Sad)
            mood_score = ((eyebrow_ratio - 0.15) / 0.05) * 50
            mood_text = "Mad / Sad"
        else:
            mood_score = 50
            mood_text = "Neutral"

        mood_score = max(0, min(100, int(mood_score)))

    metrics = {
        "face_width": face_width,
        "mouth_width": mouth_width,
        "eyebrow_dist": eyebrow_dist,
    }
    return mood_score, mood_text, metrics


def draw_face_mask(image, face_landmarks, mp_face_mesh, mp_drawing, mp_drawing_styles):
    """Draw a facial mesh mask overlay and highlight the feature calculation lines/dots."""
    h, w = image.shape[:2]

    # Draw subtle face mesh tesselation
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    )

    # Convert landmark indices to pixel coordinates
    def pt(idx):
        lm = face_landmarks.landmark[idx]
        return int(lm.x * w), int(lm.y * h)

    # Highlight calculation points & lines
    # 1. Face Width (Temples: 234 -> 454) in Cyan
    p_t1, p_t2 = pt(234), pt(454)
    cv2.line(image, p_t1, p_t2, (255, 255, 0), 2)
    cv2.circle(image, p_t1, 5, (255, 255, 0), -1)
    cv2.circle(image, p_t2, 5, (255, 255, 0), -1)

    # 2. Mouth Width (Corners: 61 -> 291) in Green
    p_m1, p_m2 = pt(61), pt(291)
    cv2.line(image, p_m1, p_m2, (0, 255, 0), 3)
    cv2.circle(image, p_m1, 6, (0, 255, 0), -1)
    cv2.circle(image, p_m2, 6, (0, 255, 0), -1)

    # 3. Eyebrow Distance (Inner: 107 -> 336) in Magenta/Orange
    p_eb1, p_eb2 = pt(107), pt(336)
    cv2.line(image, p_eb1, p_eb2, (255, 0, 255), 3)
    cv2.circle(image, p_eb1, 6, (255, 0, 255), -1)
    cv2.circle(image, p_eb2, 6, (255, 0, 255), -1)


def run(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[FAIL] could not open camera index {args.camera}")
        return 2

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        print("Mood Meter starting. Press 'ESC' or 'q' to exit.")
        window = "Mood Meter (q/Esc to quit)"
        sized = False

        while cap.isOpened():
            success, image = cap.read()
            if not success or image is None:
                print("Ignoring empty camera frame.")
                break

            image = cv2.flip(image, 1)

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            image.flags.writeable = True

            mood_score = 50  # Default neutral
            mood_text = "Neutral"

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mood_score, mood_text, _metrics = calculate_mood(face_landmarks.landmark)

                    # Draw facial mesh mask and calculation indicators
                    if not args.no_mask:
                        draw_face_mask(image, face_landmarks, mp_face_mesh, mp_drawing, mp_drawing_styles)

            # Draw the Mood Meter bar
            bar_width = 300
            bar_height = 30
            x_offset = 50
            y_offset = 80

            # Background bar
            cv2.rectangle(image, (x_offset, y_offset), (x_offset + bar_width, y_offset + bar_height), (100, 100, 100), -1)

            # Fill bar based on score
            fill_width = int((mood_score / 100) * bar_width)

            # Color gradient: 0 is red, 50 is yellow, 100 is green
            if mood_score < 50:
                r = 255
                g = int((mood_score / 50) * 255)
            else:
                r = int((1 - (mood_score - 50) / 50) * 255)
                g = 255
            b = 0

            cv2.rectangle(image, (x_offset, y_offset), (x_offset + fill_width, y_offset + bar_height), (b, g, r), -1)
            cv2.rectangle(image, (x_offset, y_offset), (x_offset + bar_width, y_offset + bar_height), (255, 255, 255), 2)

            # Text labels
            cv2.putText(image, f"Mood: {mood_score}/100", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(image, f"State: {mood_text}", (x_offset, y_offset + bar_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(image, f"Mood: {mood_score}/100", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"State: {mood_text}", (x_offset, y_offset + bar_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if not sized:
                display.open_window(cv2, window, image, args.display_scale)
                sized = True
            cv2.imshow(window, image)

            key = cv2.waitKey(5) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


class _LM:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x = x
        self.y = y


def self_test():
    """Verify mood calculation logic without a camera."""
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<38} expected {want}, got {got}")

    # Build synthetic landmark map with 468 points
    landmarks = [_LM(0.5, 0.5) for _ in range(468)]
    # Temples 234 (left), 454 (right): span = 0.20
    landmarks[234] = _LM(0.40, 0.5)
    landmarks[454] = _LM(0.60, 0.5)

    # 1. Neutral face: mouth_width = 0.07 -> ratio = 0.35, eyebrow_dist = 0.05 -> ratio = 0.25
    landmarks[61] = _LM(0.465, 0.6)
    landmarks[291] = _LM(0.535, 0.6)
    landmarks[107] = _LM(0.475, 0.4)
    landmarks[336] = _LM(0.525, 0.4)
    s_neu, t_neu, _ = calculate_mood(landmarks)
    check("neutral mood text", t_neu, "Neutral")
    check("neutral mood score", s_neu, 50)

    # 2. Happy face: mouth_width = 0.10 -> ratio = 0.50 (> 0.40)
    landmarks[61] = _LM(0.45, 0.6)
    landmarks[291] = _LM(0.55, 0.6)
    s_hap, t_hap, _ = calculate_mood(landmarks)
    check("happy mood text", t_hap, "Happy")
    check("happy mood score > 50", s_hap > 50, True)

    # 3. Frowning face: mouth_width = 0.06 -> ratio = 0.30, eyebrow_dist = 0.03 -> ratio = 0.15 (< 0.20)
    landmarks[61] = _LM(0.47, 0.6)
    landmarks[291] = _LM(0.53, 0.6)
    landmarks[107] = _LM(0.485, 0.4)
    landmarks[336] = _LM(0.515, 0.4)
    s_mad, t_mad, _ = calculate_mood(landmarks)
    check("frown mood text", t_mad, "Mad / Sad")
    check("frown mood score < 50", s_mad < 50, True)

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "--self-test":
        return self_test()

    parser = argparse.ArgumentParser(description="Mood Meter emotion detector (OpenCV + MediaPipe).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--no-mask", action="store_true", help="Do not draw the face mesh mask overlay.")
    parser.add_argument("--display-scale", type=float, default=1.5, help="Window scale factor (default 1.5).")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())

