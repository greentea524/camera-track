#!/usr/bin/env python3
"""
rubiks_cube.py — Rubik's Cube face tracker (webcam -> sticker colors -> face state).

Detects a single Rubik's Cube face from a live webcam feed, corrects its
perspective into a straight-on square, classifies each of the 9 stickers by
color, and shows the resulting face state live in the preview window.

This single script implements GitHub issues #20-25:

  #20/#21  New tracker module, wired into main.py, with the standard camera /
           display-scale CLI flags and a clean exit on quit keys.
  #22      Detect the cube face in the frame and perspective-warp it into a
           straight-on square; gracefully skip frames with no good detection.
  #23      Segment the warped square into 9 cells and classify each sticker's
           color with a tunable HSV rule set.
  #24      Map the 9 classified colors into a face-state representation (a
           3x3 grid plus a flat 9-character notation string), with basic
           validation for unknown/ambiguous stickers.
  #25      Overlay the detected face outline, a gridded thumbnail, and the
           current state string on the live preview.
  #26      Solve mode: scan all six faces step by step, run the beginner
           solver (cube_solver.py), and show move-by-move guidance on screen.

Solve mode: press 's', then show each prompted face to the camera and press
SPACE to capture it (six faces total). The solver validates the scan and the
overlay then walks you through the solution one move at a time ("RED face:
1/4 turn clockwise"); SPACE advances, 'r' restarts. Moves name faces by their
*center color*, and clockwise means "as if looking straight at that face",
so the instructions work no matter how you hold the cube.

Usage (run on your own machine — needs camera + display):

    python rubiks_cube.py                  # default webcam, no mirroring
    python rubiks_cube.py --camera 1       # use a different camera index
    python rubiks_cube.py --flip           # mirror the image (selfie view)
    python rubiks_cube.py --min-area 0.08  # detect a smaller/farther face
    python rubiks_cube.py --self-test      # verify detection/color/solve logic

Press 'q' or Esc in the video window to quit.
"""

import argparse
import colorsys
import sys

import cube_solver
import display

# Canonical sticker colors, keyed by the single-letter code used in the face
# state string. "?" marks a cell that couldn't be confidently classified.
COLOR_NAMES = {
    "W": "white",
    "Y": "yellow",
    "R": "red",
    "O": "orange",
    "B": "blue",
    "G": "green",
}

# BGR swatches used to draw each color's label in the overlay.
COLOR_SWATCH = {
    "W": (255, 255, 255),
    "Y": (0, 213, 255),
    "R": (0, 0, 200),
    "O": (0, 128, 255),
    "B": (200, 60, 0),
    "G": (0, 160, 0),
    "?": (128, 128, 128),
}

WARP_SIZE = 300  # px, side length of the perspective-corrected face square
GRID_N = 3

# --- #23: HSV thresholds for sticker color classification ------------------
# OpenCV HSV ranges: h is 0-179, s/v are 0-255. Tuned for a standard cube
# color scheme under fairly even lighting; nudge these per lighting setup.
DARK_V_MAX = 50        # below this: shadow / bad exposure, not a real reading
WHITE_S_MAX = 45       # saturation below this, with enough brightness, is white
WHITE_V_MIN = 140
HUE_RED_MAX = 6
HUE_ORANGE_MAX = 18
HUE_YELLOW_MAX = 34
HUE_GREEN_MAX = 85
HUE_BLUE_MAX = 135
HUE_RED_WRAP_MIN = 172  # red also wraps around the high end of the hue wheel


# --- #22: face detection + perspective correction ---------------------------
def order_corners(pts):
    """Order 4 arbitrary (x, y) points as [top-left, top-right, bottom-right, bottom-left].

    Pure and hardware-free. Classic sum/diff trick: the top-left corner has
    the smallest x+y, the bottom-right the largest; the top-right has the
    smallest y-x, the bottom-left the largest.
    """
    pts = list(pts)
    sums = [p[0] + p[1] for p in pts]
    diffs = [p[1] - p[0] for p in pts]
    top_left = pts[sums.index(min(sums))]
    bottom_right = pts[sums.index(max(sums))]
    top_right = pts[diffs.index(min(diffs))]
    bottom_left = pts[diffs.index(max(diffs))]
    return [top_left, top_right, bottom_right, bottom_left]


def detect_face_quad(frame, cv2, np, min_area_ratio=0.12):
    """Locate the cube face in `frame` and return its 4 ordered corners, or None.

    Edge-detects, then dilates to bridge the dark grooves between stickers so
    the whole face becomes one blob, and approximates the largest resulting
    contour to a quadrilateral. Returns None for anything too small or not
    roughly square, so callers can show a graceful "no face" state instead of
    a bogus detection.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    dilated = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidate = max(contours, key=cv2.contourArea)
    if cv2.contourArea(candidate) < min_area_ratio * (w * h):
        return None

    x, y, bw, bh = cv2.boundingRect(candidate)
    aspect = bw / bh if bh else 0
    if not (0.6 <= aspect <= 1.6):
        return None  # too far from square to be a face-on cube view

    peri = cv2.arcLength(candidate, True)
    approx = cv2.approxPolyDP(candidate, 0.03 * peri, True)
    if len(approx) != 4:
        # Not a clean quad (stickers didn't fully merge) — fall back to the
        # rotated bounding box, still good enough to warp a square region.
        approx = cv2.boxPoints(cv2.minAreaRect(candidate))

    pts = [tuple(p) for p in np.array(approx).reshape(-1, 2)]
    return order_corners(pts)


def warp_face(frame, cv2, np, quad, size=WARP_SIZE):
    """Perspective-warp the quadrilateral region of `frame` into a `size`x`size` square."""
    src = np.array(quad, dtype=np.float32)
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (size, size))


def grid_cells(size=WARP_SIZE, n=GRID_N):
    """Return the n*n cell boxes (x1, y1, x2, y2) over a size x size image, row-major.

    Pure and hardware-free.
    """
    step = size / n
    cells = []
    for row in range(n):
        for col in range(n):
            x1, y1 = int(col * step), int(row * step)
            x2, y2 = int((col + 1) * step), int((row + 1) * step)
            cells.append((x1, y1, x2, y2))
    return cells


# --- #23: sticker color classification --------------------------------------
def classify_hsv(h, s, v):
    """Classify a single HSV pixel (OpenCV scale) into a cube sticker color code."""
    if v < DARK_V_MAX:
        return "?"
    if s < WHITE_S_MAX and v > WHITE_V_MIN:
        return "W"
    if s < WHITE_S_MAX:
        return "?"  # low saturation but too dim to call white confidently

    if h < HUE_RED_MAX or h >= HUE_RED_WRAP_MIN:
        return "R"
    if h < HUE_ORANGE_MAX:
        return "O"
    if h < HUE_YELLOW_MAX:
        return "Y"
    if h < HUE_GREEN_MAX:
        return "G"
    if h < HUE_BLUE_MAX:
        return "B"
    return "?"


def classify_color(b, g, r):
    """Classify a mean BGR sticker color (0-255 each) into a cube color code.

    Uses the stdlib `colorsys` for the BGR->HSV conversion (rescaled to
    OpenCV's h:0-179 / s,v:0-255 ranges) so this stays pure Python and
    testable without an image library.
    """
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return classify_hsv(h * 179.0, s * 255.0, v * 255.0)


def sample_cell_bgr(cv2, warped, box, margin=0.28):
    """Mean BGR color of the inset center of a cell (avoids grid-line edges)."""
    x1, y1, x2, y2 = box
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    inner = warped[y1 + my:y2 - my, x1 + mx:x2 - mx]
    return cv2.mean(inner)[:3]


def classify_face(cv2, warped):
    """Segment the warped face into 9 cells and classify each sticker's color.

    Returns a 3x3 list-of-lists of color codes, row-major, top-to-bottom /
    left-to-right as seen on screen.
    """
    size = warped.shape[0]
    codes = [classify_color(*sample_cell_bgr(cv2, warped, box)) for box in grid_cells(size)]
    return [codes[i:i + GRID_N] for i in range(0, len(codes), GRID_N)]


# --- #24: cube face-state representation ------------------------------------
def face_state(grid):
    """Flatten a 3x3 color-code grid into a face-state record.

    Returns a dict with:
      grid   - the original 3x3 list of codes
      flat   - the 9-character notation string, row-major
      valid  - True if every cell resolved to a known color (no "?")
      issues - human-readable problems, empty when valid
    """
    flat = "".join(code for row in grid for code in row)
    issues = []
    unknown = flat.count("?")
    if unknown:
        issues.append(f"{unknown} sticker(s) could not be classified")
    return {"grid": grid, "flat": flat, "valid": unknown == 0, "issues": issues}


# --- #26: six-face scan -> solver state -> human guidance --------------------
# Scan order and how to hold the cube for each capture. The prompts assume the
# standard color scheme (white opposite yellow, green opposite blue, red
# opposite orange); the assembly itself is scheme-agnostic and keyed on the
# captured center colors.
SCAN_STEPS = [
    ("F", "G", "Show the GREEN face to the camera, WHITE on top"),
    ("R", "R", "Turn the cube: show RED to the camera, WHITE on top"),
    ("B", "B", "Keep turning: show BLUE to the camera, WHITE on top"),
    ("L", "O", "Keep turning: show ORANGE to the camera, WHITE on top"),
    ("U", "W", "Back to GREEN in front, then tilt down: WHITE to camera"),
    ("D", "Y", "From GREEN in front, tilt up: YELLOW to camera"),
]


def assemble_cube(captures):
    """Turn six captured 3x3 color grids into a solver state.

    `captures` maps face letters (URFDLB) to 3x3 grids of color codes as
    captured by the scanner. Colors are mapped to faces by each capture's
    center sticker, so any color scheme works as long as centers are right.
    Returns (state, errors): state is {face: [9 face letters]} or None.
    """
    if sorted(captures) != sorted("URFDLB"):
        return None, ["captures must cover exactly the six faces"]
    color_to_face = {}
    for face, grid in captures.items():
        center = grid[1][1]
        if center in color_to_face:
            return None, [f"two faces share the center color {center!r}"]
        color_to_face[center] = face

    errors = []
    counts = {}
    for grid in captures.values():
        for row in grid:
            for code in row:
                counts[code] = counts.get(code, 0) + 1
    for color in color_to_face:
        if counts.get(color, 0) != 9:
            errors.append(f"found {counts.get(color, 0)} '{color}' stickers, expected 9")
    unknown = {c: n for c, n in counts.items() if c not in color_to_face}
    if unknown:
        errors.append(f"unrecognized sticker colors: {unknown}")
    if errors:
        return None, errors

    state = {
        face: [color_to_face[code] for row in grid for code in row]
        for face, grid in captures.items()
    }
    return state, []


def move_text(move, face_colors):
    """Human instruction for one move token, naming the face by its color.

    `face_colors` maps face letters to color codes (from the scan), so 'R2'
    becomes e.g. "RED face: half turn". Clockwise is defined as if looking
    straight at that face, which holds however the cube is held.
    """
    face, suffix = move[0], move[1:]
    color = COLOR_NAMES.get(face_colors.get(face, ""), face).upper()
    amount = {
        "": "1/4 turn clockwise",
        "2": "half turn",
        "'": "1/4 turn counter-clockwise",
    }[suffix]
    return f"{color} face: {amount}"


# --- #25: live overlay -------------------------------------------------------
def annotate_warped(cv2, warped, grid):
    """Draw 3x3 grid lines and each cell's color code onto a copy of the warped face."""
    out = warped.copy()
    size = out.shape[0]
    step = size // GRID_N
    for i in range(1, GRID_N):
        cv2.line(out, (0, i * step), (size, i * step), (0, 0, 0), 2)
        cv2.line(out, (i * step, 0), (i * step, size), (0, 0, 0), 2)
    for r, row in enumerate(grid):
        for c, code in enumerate(row):
            cx, cy = c * step + step // 2, r * step + step // 2
            color = COLOR_SWATCH.get(code, COLOR_SWATCH["?"])
            cv2.circle(out, (cx, cy), step // 4, color, -1)
            cv2.circle(out, (cx, cy), step // 4, (0, 0, 0), 2)
            cv2.putText(
                out, code, (cx - 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 2, cv2.LINE_AA,
            )
    return out


def draw_overlay(cv2, np, frame, quad, warped_annotated, state):
    """Draw the detected face outline, a gridded thumbnail, and the state string."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)

    if quad is None:
        cv2.putText(
            frame, "No cube face detected", (15, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 200, 255), 2, cv2.LINE_AA,
        )
        return

    pts = np.array(quad, dtype=np.int32)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    color = (0, 255, 0) if state["valid"] else (0, 200, 255)
    cv2.putText(
        frame, f"State: {state['flat']}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, color, 2, cv2.LINE_AA,
    )

    thumb_size = max(1, min(160, h - 20, w // 3))
    thumb = cv2.resize(warped_annotated, (thumb_size, thumb_size))
    frame[10:10 + thumb_size, w - thumb_size - 10:w - 10] = thumb


def draw_workflow(cv2, frame, lines):
    """#26: render the solve-workflow instruction band along the bottom."""
    h, w = frame.shape[:2]
    band = 28 + 30 * len(lines)
    cv2.rectangle(frame, (0, h - band), (w, h), (0, 0, 0), -1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame, text, (15, h - band + 32 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, color, 2, cv2.LINE_AA,
        )


# --- main capture loop (#20-22, #26 workflow) ---------------------------------
def run(args):
    """Live capture + detect/classify/overlay loop."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[FAIL] could not open camera index {args.camera}")
        return 2

    print("Running Rubik's Cube tracker — 's' to scan & solve, 'q'/Esc to quit.")
    window = "Rubik's Cube Tracker (q/Esc to quit)"
    sized = False

    # #26 solve-workflow state: track -> scan (6 captures) -> guide -> done.
    mode = "track"
    captures = {}
    face_colors = {}
    solution = []
    move_index = 0
    message = ""

    white = (255, 255, 255)
    green = (0, 255, 0)
    amber = (0, 200, 255)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[warn] dropped a frame from the camera")
                continue

            # Unlike the other apps, we do NOT mirror by default: the face
            # state reads left-to-right off the physical cube, and mirroring
            # would silently reverse that orientation in the output.
            if args.flip:
                frame = cv2.flip(frame, 1)

            quad = detect_face_quad(frame, cv2, np, min_area_ratio=args.min_area)
            state = None
            warped_annotated = None
            grid = None
            if quad is not None:
                warped = warp_face(frame, cv2, np, quad)
                grid = classify_face(cv2, warped)
                state = face_state(grid)
                warped_annotated = annotate_warped(cv2, warped, grid)

            draw_overlay(cv2, np, frame, quad, warped_annotated, state)

            # #26: workflow instruction band.
            if mode == "track":
                draw_workflow(cv2, frame, [
                    ("Press 's' to scan the cube and get solve guidance", white),
                ])
            elif mode == "scan":
                _face, color, prompt = SCAN_STEPS[len(captures)]
                lines = [
                    (f"Scan {len(captures) + 1}/6: {prompt}", white),
                    ("SPACE = capture    r = restart    q = quit", green),
                ]
                if message:
                    lines.append((message, amber))
                draw_workflow(cv2, frame, lines)
            elif mode == "guide":
                lines = [
                    (f"Move {move_index + 1}/{len(solution)}: "
                     f"{move_text(solution[move_index], face_colors)}", green),
                    ("(clockwise = as if looking at that face)", white),
                    ("SPACE = I did it, next    r = restart    q = quit", white),
                ]
                draw_workflow(cv2, frame, lines)
            elif mode == "done":
                draw_workflow(cv2, frame, [
                    ("Cube solved — nice!    r = scan again    q = quit", green),
                ])

            if not sized:
                display.open_window(cv2, window, frame, args.display_scale)
                sized = True
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # clean exit
                break
            if key == ord("r"):
                mode, captures, solution, move_index, message = "track", {}, [], 0, ""
            elif key == ord("s") and mode == "track":
                mode, captures, message = "scan", {}, ""
            elif key == ord(" ") and mode == "scan":
                face, expect_color, _prompt = SCAN_STEPS[len(captures)]
                if state is None or not state["valid"]:
                    message = "No clean face detection — adjust and try again"
                elif grid[1][1] != expect_color:
                    got = COLOR_NAMES.get(grid[1][1], "?")
                    message = (f"Center looks {got.upper()}, expected "
                               f"{COLOR_NAMES[expect_color].upper()} — check the prompt")
                else:
                    captures[face] = grid
                    message = ""
                    if len(captures) == 6:
                        cube_state, errors = assemble_cube(captures)
                        if errors:
                            print("[scan] " + "; ".join(errors))
                            captures, message = {}, "Scan inconsistent — starting over"
                            continue
                        try:
                            solution = cube_solver.solve(cube_state)
                        except ValueError as exc:
                            print(f"[scan] {exc}")
                            captures, message = {}, "Scan looks misread — starting over"
                            continue
                        face_colors = {f: g[1][1] for f, g in captures.items()}
                        move_index = 0
                        mode = "done" if not solution else "guide"
                        print(f"[solve] {len(solution)} moves: {' '.join(solution)}")
            elif key == ord(" ") and mode == "guide":
                move_index += 1
                if move_index >= len(solution):
                    mode = "done"
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


# --- self-test (no camera needed) -------------------------------------------
def self_test():
    """Verify the pure detection/color/state logic without a camera."""
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<38} expected {want}, got {got}")

    # order_corners: scrambled square corners resolve to TL, TR, BR, BL.
    scrambled = [(100, 100), (0, 0), (100, 0), (0, 100)]
    check(
        "order_corners on a square",
        order_corners(scrambled),
        [(0, 0), (100, 0), (100, 100), (0, 100)],
    )

    # grid_cells: 300x300 / 3x3 -> 9 cells of 100x100, row-major.
    cells = grid_cells(300, 3)
    check("grid_cells count", len(cells), 9)
    check("grid_cells first cell", cells[0], (0, 0, 100, 100))
    check("grid_cells last cell", cells[-1], (200, 200, 300, 300))

    # classify_color: representative BGR means for each cube color.
    check("classify red", classify_color(20, 25, 200), "R")
    check("classify orange", classify_color(20, 110, 230), "O")
    check("classify yellow", classify_color(30, 210, 220), "Y")
    check("classify green", classify_color(60, 160, 30), "G")
    check("classify blue", classify_color(180, 90, 20), "B")
    check("classify white", classify_color(230, 230, 230), "W")
    check("classify shadow -> unknown", classify_color(10, 10, 10), "?")

    # face_state: a clean face vs. one with an unresolved sticker.
    solved = [["W", "W", "W"], ["W", "W", "W"], ["W", "W", "W"]]
    state = face_state(solved)
    check("face_state flat", state["flat"], "WWWWWWWWW")
    check("face_state valid", state["valid"], True)
    check("face_state issues empty", state["issues"], [])

    messy = [["W", "W", "?"], ["W", "W", "W"], ["W", "W", "W"]]
    messy_state = face_state(messy)
    check("face_state flags unknowns", messy_state["valid"], False)
    check("face_state issue count", len(messy_state["issues"]), 1)

    # --- #26: scan assembly + solve guidance -------------------------------
    scheme = {face: color for face, color, _prompt in SCAN_STEPS}

    def to_captures(cube):
        """Render a solver Cube back into per-face 3x3 color grids."""
        return {
            face: [[scheme[cube.f[face][r * 3 + c]] for c in range(3)] for r in range(3)]
            for face in "URFDLB"
        }

    solved_caps = to_captures(cube_solver.Cube())
    cube_state, errors = assemble_cube(solved_caps)
    check("assemble solved captures", errors, [])
    check("solved scan needs no moves", cube_solver.solve(cube_state), [])

    # Scramble a virtual cube, "scan" it, solve, and apply the guidance moves.
    scrambled = cube_solver.Cube().apply(cube_solver.scramble(25, seed=7))
    cube_state, errors = assemble_cube(to_captures(scrambled))
    check("assemble scrambled captures", errors, [])
    moves = cube_solver.solve(cube_state)
    check("guidance really solves the cube",
          scrambled.apply(moves).is_solved(), True)

    bad_caps = to_captures(cube_solver.Cube())
    bad_caps["F"][0][0] = "W"  # 10 whites, 8 greens
    _state, errors = assemble_cube(bad_caps)
    check("bad sticker counts rejected", len(errors) > 0, True)

    check("move_text names the color",
          move_text("R2", scheme), "RED face: half turn")
    check("move_text counter-clockwise",
          move_text("F'", scheme), "GREEN face: 1/4 turn counter-clockwise")

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rubik's Cube face tracker (OpenCV). Detects one face, "
                     "classifies its 9 stickers, and shows the face state live. "
                     "'q' to quit."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument(
        "--flip", action="store_true",
        help="Mirror the webcam image. Off by default (unlike the other apps) "
             "because mirroring would reverse left/right in the detected face state.",
    )
    parser.add_argument(
        "--min-area", type=float, default=0.12,
        help="Minimum detected-face area as a fraction of the frame (default "
             "0.12). Lower this if the cube is small/far from the camera.",
    )
    parser.add_argument(
        "--display-scale", type=float, default=1.5,
        help="Initial window size as a multiple of the camera frame "
             "(default 1.5). The window is resizable.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Verify detection/color/state logic without a camera, then exit.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
