#!/usr/bin/env python3
"""
fruit_ninja.py — Fruit Ninja-style slicing game using OpenCV + MediaPipe.

Fruits (colored circles) are launched upward from the bottom of the screen
with gravity pulling them back down. Slice them with your index fingertip
by swiping through them. Supports two hands for dual-blade action.

Issue: #44

Usage:
    python fruit_ninja.py                # default settings
    python fruit_ninja.py --camera 1     # different camera index
    python fruit_ninja.py --self-test    # verify game logic, no camera

Press 'q' or Esc in the video window to quit.
"""

import argparse
import collections
import math
import random
import sys
import time

import display

# MediaPipe index-fingertip landmark
INDEX_TIP = 8

# Gravity constant (pixels/s^2) — tuned for a 480p frame
GRAVITY = 900.0

# Fruit types with names and BGR colors
FRUIT_TYPES = [
    {"name": "Apple",      "color": (0, 0, 220),     "radius": 38},
    {"name": "Orange",     "color": (0, 140, 255),    "radius": 36},
    {"name": "Watermelon", "color": (50, 180, 50),    "radius": 44},
    {"name": "Lemon",      "color": (0, 230, 255),    "radius": 30},
    {"name": "Grape",      "color": (180, 0, 160),    "radius": 28},
    {"name": "Blueberry",  "color": (200, 80, 20),    "radius": 26},
]

# Bomb — lose a life if you slice it
BOMB = {"name": "Bomb", "color": (30, 30, 30), "radius": 34}


# ---------------------------------------------------------------------------
# Game objects
# ---------------------------------------------------------------------------

class Fruit:
    """A fruit projectile with gravity physics."""

    def __init__(self, x, y, vx, vy, fruit_type, spawn_time=None):
        self.x = float(x)
        self.y = float(y)
        self.vx = vx          # pixels/s horizontal
        self.vy = vy          # pixels/s vertical (negative = upward)
        self.fruit_type = fruit_type
        self.radius = fruit_type["radius"]
        self.color = fruit_type["color"]
        self.spawn_time = spawn_time or time.time()
        self.sliced = False
        self.is_bomb = (fruit_type is BOMB)

    def update(self, dt):
        """Advance physics by dt seconds."""
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def is_offscreen(self, frame_h):
        """True if the fruit has fallen below the screen."""
        return self.y > frame_h + 100

    def center(self):
        return (int(self.x), int(self.y))


class SlicedHalf:
    """A half of a sliced fruit that tumbles away."""

    def __init__(self, x, y, vx, vy, color, radius, angle):
        self.x = float(x)
        self.y = float(y)
        self.vx = vx
        self.vy = vy
        self.color = color
        self.radius = radius
        self.angle = angle        # rotation for visual effect
        self.spin = random.uniform(-8, 8)
        self.spawn_time = time.time()
        self.lifetime = 1.0       # fade out after 1 second

    def update(self, dt):
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.spin * dt

    def is_expired(self):
        return time.time() - self.spawn_time > self.lifetime

    def opacity(self):
        """1.0 → 0.0 over lifetime."""
        return max(0.0, 1.0 - (time.time() - self.spawn_time) / self.lifetime)


# ---------------------------------------------------------------------------
# Geometry: line-segment vs circle intersection for slice detection
# ---------------------------------------------------------------------------

def segment_intersects_circle(x1, y1, x2, y2, cx, cy, r):
    """Return True if the line segment (x1,y1)-(x2,y2) intersects
    or is inside the circle centered at (cx,cy) with radius r.

    Uses closest-point-on-segment projection.
    """
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy

    a = dx * dx + dy * dy
    if a < 1e-9:
        # Degenerate segment (single point)
        return math.hypot(fx, fy) <= r

    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        return False

    discriminant = math.sqrt(discriminant)
    t1 = (-b - discriminant) / (2.0 * a)
    t2 = (-b + discriminant) / (2.0 * a)

    # Check if either intersection point is within the segment [0, 1]
    if 0 <= t1 <= 1:
        return True
    if 0 <= t2 <= 1:
        return True

    # Check if both ends are inside the circle
    if t1 < 0 and t2 > 1:
        return True

    return False


# ---------------------------------------------------------------------------
# Game engine
# ---------------------------------------------------------------------------

class FruitNinjaGame:
    """Manages fruit spawning, physics, slicing, and scoring."""

    def __init__(self):
        self.score = 0
        self.lives = 3
        self.fruits = []          # active Fruit objects
        self.halves = []          # SlicedHalf debris
        self.game_over = False
        self._last_spawn = 0.0
        self._spawn_interval = 1.2   # seconds between waves
        self._combo = 0
        self._combo_time = 0.0
        self._combo_display = ""

    @property
    def spawn_interval(self):
        """Speed up spawning as score rises."""
        return max(0.5, self._spawn_interval - self.score * 0.02)

    def spawn_wave(self, frame_w, frame_h):
        """Launch a wave of 1-3 fruits from the bottom."""
        count = random.randint(1, min(3, 1 + self.score // 5))

        for _ in range(count):
            # Random horizontal position
            x = random.randint(80, frame_w - 80)
            y = frame_h + 20   # just below screen

            # Launch upward with random angle
            vx = random.uniform(-120, 120)
            vy = random.uniform(-650, -450)  # strong upward thrust

            # Pick fruit type (small chance of bomb after score >= 3)
            if self.score >= 3 and random.random() < 0.15:
                ftype = BOMB
            else:
                ftype = random.choice(FRUIT_TYPES)

            self.fruits.append(Fruit(x, y, vx, vy, ftype))

    def try_slice(self, prev_x, prev_y, cur_x, cur_y):
        """Check if the fingertip swipe from prev to cur slices any fruit.
        Returns list of sliced fruits for visual effects.
        """
        sliced = []
        for fruit in self.fruits:
            if fruit.sliced:
                continue
            if segment_intersects_circle(
                prev_x, prev_y, cur_x, cur_y,
                fruit.x, fruit.y, fruit.radius
            ):
                fruit.sliced = True
                sliced.append(fruit)

                if fruit.is_bomb:
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_over = True
                else:
                    self.score += 1
                    self._combo += 1
                    self._combo_time = time.time()

                # Create split halves
                angle = math.atan2(cur_y - prev_y, cur_x - prev_x)
                perp = angle + math.pi / 2
                speed = 120
                for sign in (-1, 1):
                    half_vx = fruit.vx + math.cos(perp) * speed * sign
                    half_vy = fruit.vy + math.sin(perp) * speed * sign - 80
                    self.halves.append(SlicedHalf(
                        fruit.x, fruit.y, half_vx, half_vy,
                        fruit.color, fruit.radius, angle
                    ))

        return sliced

    def update(self, frame_w, frame_h, dt):
        """Advance the game by dt seconds. Call once per frame."""
        if self.game_over:
            return

        now = time.time()

        # Spawn new wave
        if now - self._last_spawn >= self.spawn_interval:
            self.spawn_wave(frame_w, frame_h)
            self._last_spawn = now

        # Update fruit physics
        for fruit in self.fruits:
            fruit.update(dt)

        # Check for missed fruits (fell off screen without being sliced)
        missed = [f for f in self.fruits if f.is_offscreen(frame_h) and not f.sliced and not f.is_bomb]
        for _ in missed:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True

        # Remove offscreen / sliced fruits
        self.fruits = [f for f in self.fruits if not f.is_offscreen(frame_h) and not f.sliced]

        # Update halves
        for half in self.halves:
            half.update(dt)
        self.halves = [h for h in self.halves if not h.is_expired()]

        # Reset combo after 0.8s of no slicing
        if self._combo > 0 and now - self._combo_time > 0.8:
            if self._combo >= 3:
                self._combo_display = f"{self._combo}x COMBO!"
            self._combo = 0

    def restart(self):
        self.__init__()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _outlined_text(cv2, frame, text, pos, scale, fg, thickness=2):
    """Draw text with a black outline for readability."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness)


def draw_fruit(cv2, frame, fruit):
    """Draw a single fruit with highlight and shadow."""
    cx, cy = fruit.center()
    r = fruit.radius

    # Main body
    cv2.circle(frame, (cx, cy), r, fruit.color, -1)

    if fruit.is_bomb:
        # Draw fuse line and X
        cv2.line(frame, (cx, cy - r), (cx + 8, cy - r - 15), (0, 180, 255), 2)
        cv2.circle(frame, (cx + 8, cy - r - 15), 4, (0, 200, 255), -1)
        _outlined_text(cv2, frame, "X", (cx - 8, cy + 6), 0.5, (0, 0, 200), 2)
    else:
        # Highlight spot
        hx = cx - r // 3
        hy = cy - r // 3
        cv2.circle(frame, (hx, hy), max(3, r // 4), (255, 255, 255), -1)

    # Outline
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 1)


def draw_half(cv2, frame, half):
    """Draw a sliced half-fruit fading out."""
    cx, cy = int(half.x), int(half.y)
    r = half.radius
    alpha = half.opacity()

    # Dim the color based on opacity
    color = tuple(int(c * alpha) for c in half.color)
    cv2.circle(frame, (cx, cy), r, color, -1)
    # Slice line through the middle
    dx = int(r * math.cos(half.angle))
    dy = int(r * math.sin(half.angle))
    cv2.line(frame, (cx - dx, cy - dy), (cx + dx, cy + dy), (200, 200, 200), 1)


def draw_slash_trail(cv2, frame, trail, color=(0, 255, 255)):
    """Draw a fading gradient trail from the fingertip history."""
    for i in range(1, len(trail)):
        if trail[i - 1] is None or trail[i] is None:
            continue
        # Thickness and opacity increase toward the tip
        ratio = float(i) / len(trail)
        thickness = max(1, int(ratio * 5))
        alpha = ratio
        c = tuple(int(ch * alpha) for ch in color)
        cv2.line(frame, trail[i - 1], trail[i], c, thickness)


def draw_hud(cv2, frame, game):
    """Render the score and lives HUD."""
    h, w = frame.shape[:2]

    # Semi-transparent header
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    _outlined_text(cv2, frame, f"Score: {game.score}", (20, 35), 0.9, (0, 255, 200))
    _outlined_text(cv2, frame, f"Lives: {'*' * max(0, game.lives)}", (20, 65), 0.7, (100, 180, 255))

    # Mode label
    _outlined_text(cv2, frame, "FRUIT NINJA", (w - 220, 35), 0.7, (0, 200, 255))

    # Combo display
    if game._combo_display and time.time() - game._combo_time < 1.5:
        _outlined_text(cv2, frame, game._combo_display, (w // 2 - 100, h // 2), 1.2, (0, 255, 255), 3)


def draw_game_over(cv2, frame, game):
    """Overlay game-over screen."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    _outlined_text(cv2, frame, "GAME OVER", (w // 2 - 150, h // 2 - 40), 1.5, (0, 0, 255), 3)
    _outlined_text(cv2, frame, f"Final Score: {game.score}", (w // 2 - 130, h // 2 + 20), 1.0, (255, 255, 255))
    _outlined_text(cv2, frame, "Press SPACE to restart  |  Q to quit", (w // 2 - 250, h // 2 + 70), 0.7, (180, 180, 180))


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(args):
    """Live capture + game loop."""
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

    game = FruitNinjaGame()
    window = "Fruit Ninja (q/Esc to quit)"
    sized = False

    # Trail history per hand (up to 2 hands)
    trails = [collections.deque(maxlen=12) for _ in range(2)]
    prev_tips = [None, None]   # previous frame fingertip positions

    trail_colors = [(0, 255, 255), (255, 100, 255)]  # yellow, pink

    prev_time = time.time()
    print("Fruit Ninja starting — press 'q' or Esc to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            dt = min(now - prev_time, 0.1)  # cap dt to avoid physics explosion
            prev_time = now

            # MediaPipe hand detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            # Extract fingertip positions for up to 2 hands
            cur_tips = [None, None]
            if results.multi_hand_landmarks:
                for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks):
                    if hand_idx >= 2:
                        break
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    tip = hand_lm.landmark[INDEX_TIP]
                    cur_tips[hand_idx] = (int(tip.x * w), int(tip.y * h))

            # Slice detection & trail management for each hand
            if not game.game_over:
                for i in range(2):
                    if cur_tips[i] is not None:
                        trails[i].append(cur_tips[i])

                        if prev_tips[i] is not None:
                            # Check slice
                            px, py = prev_tips[i]
                            cx, cy = cur_tips[i]
                            game.try_slice(px, py, cx, cy)
                    else:
                        if len(trails[i]) > 0:
                            trails[i].popleft()

            prev_tips = list(cur_tips)

            # Update game physics
            game.update(w, h, dt)

            # Draw fruits
            for fruit in game.fruits:
                draw_fruit(cv2, frame, fruit)

            # Draw sliced halves
            for half in game.halves:
                draw_half(cv2, frame, half)

            # Draw slash trails
            for i in range(2):
                if len(trails[i]) > 1:
                    draw_slash_trail(cv2, frame, trails[i], trail_colors[i])

            # Draw fingertip dots
            for i in range(2):
                if cur_tips[i] is not None:
                    cx, cy = cur_tips[i]
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 0), -1)
                    cv2.circle(frame, (cx, cy), 8, trail_colors[i], -1)

            draw_hud(cv2, frame, game)

            if game.game_over:
                draw_game_over(cv2, frame, game)

            if not sized:
                display.open_window(cv2, window, frame)
                sized = True
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" ") and game.game_over:
                game.restart()
                trails = [collections.deque(maxlen=12) for _ in range(2)]
                prev_tips = [None, None]
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------------------
# Self-test (no camera needed)
# ---------------------------------------------------------------------------

def self_test():
    """Verify game logic without a camera."""
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<44} expected {want}, got {got}")

    # Segment-circle intersection
    check("segment through circle center",
          segment_intersects_circle(0, 0, 200, 0, 100, 0, 20), True)
    check("segment misses circle",
          segment_intersects_circle(0, 0, 200, 0, 100, 50, 20), False)
    check("segment endpoint inside circle",
          segment_intersects_circle(90, 0, 100, 0, 100, 0, 20), True)
    check("degenerate segment inside circle",
          segment_intersects_circle(100, 0, 100, 0, 100, 0, 20), True)
    check("degenerate segment outside circle",
          segment_intersects_circle(200, 0, 200, 0, 100, 0, 20), False)
    check("segment tangent to circle",
          segment_intersects_circle(0, 20, 200, 20, 100, 0, 20), True)
    check("segment just outside circle",
          segment_intersects_circle(0, 21, 200, 21, 100, 0, 20), False)

    # Game initialization
    g = FruitNinjaGame()
    check("initial score", g.score, 0)
    check("initial lives", g.lives, 3)
    check("not game over initially", g.game_over, False)

    # Fruit physics
    f = Fruit(100, 100, 50, -200, FRUIT_TYPES[0], spawn_time=time.time())
    f.update(0.1)  # 100ms
    check("fruit moves right", f.x > 100, True)
    check("fruit moves up initially", f.y < 100, True)

    # Slicing
    g2 = FruitNinjaGame()
    g2.fruits.append(Fruit(100, 100, 0, 0, FRUIT_TYPES[0]))
    sliced = g2.try_slice(50, 100, 150, 100)  # horizontal swipe through center
    check("slice registers", len(sliced), 1)
    check("score after slice", g2.score, 1)

    # Bomb slicing
    g3 = FruitNinjaGame()
    g3.fruits.append(Fruit(100, 100, 0, 0, BOMB))
    g3.try_slice(50, 100, 150, 100)
    check("bomb costs a life", g3.lives, 2)

    # Restart
    g2.restart()
    check("restart resets score", g2.score, 0)
    check("restart resets lives", g2.lives, 3)

    # Spawn interval ramp
    g4 = FruitNinjaGame()
    base = g4.spawn_interval
    g4.score = 20
    check("spawn interval decreases with score", g4.spawn_interval < base, True)
    check("spawn interval has a floor", g4.spawn_interval >= 0.5, True)

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "--self-test":
        return self_test()

    parser = argparse.ArgumentParser(
        description="Fruit Ninja-style slicing game (OpenCV + MediaPipe).")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0).")
    parser.add_argument("--display-scale", type=float, default=1.5,
                        help="Window scale factor (default 1.5).")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
