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


class JuiceParticle:
    """A particle burst droplet when a fruit is sliced."""

    def __init__(self, x, y, color):
        self.x = float(x)
        self.y = float(y)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(150, 450)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.radius = random.uniform(3, 8)
        self.spawn_time = time.time()
        self.lifetime = random.uniform(0.3, 0.6)

    def update(self, dt):
        self.vy += GRAVITY * 0.5 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.radius = max(1.0, self.radius - dt * 6)

    def is_expired(self):
        return time.time() - self.spawn_time > self.lifetime

    def opacity(self):
        return max(0.0, 1.0 - (time.time() - self.spawn_time) / self.lifetime)


class SliceFlash:
    """A glowing flash line segment along the cut vector."""

    def __init__(self, x1, y1, x2, y2, color=(255, 255, 255)):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.color = color
        self.spawn_time = time.time()
        self.lifetime = 0.2

    def is_expired(self):
        return time.time() - self.spawn_time > self.lifetime

    def opacity(self):
        return max(0.0, 1.0 - (time.time() - self.spawn_time) / self.lifetime)


class FloatingPopup:
    """A floating text indicator (+1, +3, BOOM!)."""

    def __init__(self, x, y, text, color=(0, 255, 255), scale=0.8):
        self.x = float(x)
        self.y = float(y)
        self.text = text
        self.color = color
        self.scale = scale
        self.vy = -90.0  # float upward
        self.spawn_time = time.time()
        self.lifetime = 0.8

    def update(self, dt):
        self.y += self.vy * dt

    def is_expired(self):
        return time.time() - self.spawn_time > self.lifetime

    def opacity(self):
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
        self.fruits = []          # active Fruit objects
        self.halves = []          # SlicedHalf debris
        self.particles = []       # JuiceParticle objects
        self.flashes = []         # SliceFlash objects
        self.popups = []          # FloatingPopup objects
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

                # Create slice flash effect
                self.flashes.append(SliceFlash(prev_x, prev_y, cur_x, cur_y))

                if fruit.is_bomb:
                    self.score = max(0, self.score - 3)
                    self.popups.append(FloatingPopup(fruit.x, fruit.y, "BOOM! -3", (0, 0, 255), scale=1.0))
                    # Spawn explosive dark/orange particles
                    for _ in range(20):
                        pcolor = random.choice([(0, 100, 255), (0, 0, 255), (50, 50, 50)])
                        self.particles.append(JuiceParticle(fruit.x, fruit.y, pcolor))
                else:
                    self.score += 1
                    self._combo += 1
                    self._combo_time = time.time()

                    pts_text = f"+{self._combo}" if self._combo > 1 else "+1"
                    pts_color = (0, 255, 255) if self._combo > 1 else (0, 255, 0)
                    self.popups.append(FloatingPopup(fruit.x, fruit.y, pts_text, pts_color, scale=0.85))

                    # Spawn juice splatter particles
                    for _ in range(14):
                        self.particles.append(JuiceParticle(fruit.x, fruit.y, fruit.color))

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
        now = time.time()

        # Spawn new wave
        if now - self._last_spawn >= self.spawn_interval:
            self.spawn_wave(frame_w, frame_h)
            self._last_spawn = now

        # Update fruit physics
        for fruit in self.fruits:
            fruit.update(dt)

        # Remove offscreen / sliced fruits
        self.fruits = [f for f in self.fruits if not f.is_offscreen(frame_h) and not f.sliced]

        # Update halves
        for half in self.halves:
            half.update(dt)
        self.halves = [h for h in self.halves if not h.is_expired()]

        # Update particles
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if not p.is_expired()]

        # Update popups
        for pop in self.popups:
            pop.update(dt)
        self.popups = [pop for pop in self.popups if not pop.is_expired()]

        # Update flashes
        self.flashes = [f for f in self.flashes if not f.is_expired()]

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
    """Draw detailed fruit graphics matching each fruit type."""
    cx, cy = fruit.center()
    r = fruit.radius
    name = fruit.fruit_type.get("name", "")

    if fruit.is_bomb:
        # Bomb: Metallic dark sphere
        cv2.circle(frame, (cx, cy), r, (35, 35, 35), -1)
        cv2.circle(frame, (cx - r // 3, cy - r // 3), r // 2, (70, 70, 70), -1)
        cv2.circle(frame, (cx - r // 3, cy - r // 3), r // 4, (120, 120, 120), -1)
        # Fuse wire
        fuse_tip = (cx + 10, cy - r - 14)
        cv2.line(frame, (cx, cy - r + 3), fuse_tip, (40, 80, 120), 3)
        # Animated spark on fuse tip
        t = time.time()
        spark_color = (0, random.randint(180, 255), 255)
        cv2.circle(frame, fuse_tip, random.randint(3, 6), spark_color, -1)
        # Spark rays
        for angle in [0, 1.2, 2.4, 3.6, 4.8]:
            sx = int(fuse_tip[0] + math.cos(angle + t * 10) * 8)
            sy = int(fuse_tip[1] + math.sin(angle + t * 10) * 8)
            cv2.line(frame, fuse_tip, (sx, sy), (0, 255, 255), 1)
        # Bomb outline
        cv2.circle(frame, (cx, cy), r, (150, 150, 150), 2)
        return

    if name == "Apple":
        # Red apple body
        cv2.circle(frame, (cx, cy), r, (10, 10, 220), -1)
        cv2.circle(frame, (cx - r // 3, cy - r // 3), r // 3, (80, 80, 255), -1)
        # Indentation top
        cv2.ellipse(frame, (cx, cy - r + 4), (r // 3, r // 6), 0, 0, 360, (0, 0, 160), -1)
        # Brown stem
        cv2.line(frame, (cx, cy - r + 4), (cx + 4, cy - r - 12), (30, 60, 100), 3)
        # Green leaf
        cv2.ellipse(frame, (cx + 10, cy - r - 8), (8, 4), -30, 0, 360, (20, 180, 40), -1)
        # Gloss spot
        cv2.circle(frame, (cx - r // 3, cy - r // 3), 4, (255, 255, 255), -1)

    elif name == "Orange":
        # Orange sphere
        cv2.circle(frame, (cx, cy), r, (0, 140, 255), -1)
        # Dimple texture dots
        for dx, dy in [(-r//2, 0), (r//3, r//3), (-r//4, r//3), (r//4, -r//4)]:
            cv2.circle(frame, (cx + dx, cy + dy), 2, (0, 100, 220), -1)
        # Small green stem cap
        cv2.circle(frame, (cx, cy - r + 3), 4, (20, 160, 30), -1)
        # Gloss spot
        cv2.circle(frame, (cx - r // 3, cy - r // 3), 5, (255, 255, 255), -1)

    elif name == "Watermelon":
        # Dark green body
        cv2.circle(frame, (cx, cy), r, (30, 120, 30), -1)
        # Dark wavy stripes
        for offset in [-r//2, 0, r//2]:
            cv2.ellipse(frame, (cx + offset, cy), (r // 4, r - 4), 15, 0, 360, (10, 60, 10), 3)
        # Gloss spot
        cv2.circle(frame, (cx - r // 3, cy - r // 3), 6, (120, 255, 120), -1)

    elif name == "Lemon":
        # Yellow oval shape
        cv2.ellipse(frame, (cx, cy), (r + 4, r - 3), -20, 0, 360, (0, 230, 255), -1)
        # Nubs at ends
        cv2.circle(frame, (cx - r - 2, cy + 3), 4, (0, 200, 240), -1)
        cv2.circle(frame, (cx + r + 2, cy - 3), 4, (0, 200, 240), -1)
        # Gloss spot
        cv2.circle(frame, (cx - r // 3, cy - r // 3), 5, (255, 255, 255), -1)

    elif name == "Grape":
        # Cluster of purple grapes
        offsets = [(0, 4), (-r//2, -r//3), (r//2, -r//3), (0, -r//2)]
        for dx, dy in offsets:
            cv2.circle(frame, (cx + dx, cy + dy), r // 2 + 2, (160, 20, 140), -1)
            cv2.circle(frame, (cx + dx - 2, cy + dy - 2), 2, (230, 150, 255), -1)
        # Green stem
        cv2.line(frame, (cx, cy - r + 2), (cx + 2, cy - r - 8), (40, 180, 50), 2)

    elif name == "Blueberry":
        # Deep blue sphere
        cv2.circle(frame, (cx, cy), r, (180, 60, 20), -1)
        # Crown top calyx
        cv2.circle(frame, (cx, cy - r + 5), 5, (120, 30, 10), -1)
        cv2.circle(frame, (cx, cy - r + 5), 2, (220, 120, 80), -1)
        # Gloss spot
        cv2.circle(frame, (cx - r // 3, cy - r // 3), 4, (255, 200, 180), -1)

    else:
        # Fallback circle
        cv2.circle(frame, (cx, cy), r, fruit.color, -1)
        cv2.circle(frame, (cx - r // 3, cy - r // 3), max(3, r // 4), (255, 255, 255), -1)

    # Outer crisp white outline
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 1)


def draw_half(cv2, frame, half):
    """Draw a sliced half-fruit fading out, showing inner pulp/seeds."""
    cx, cy = int(half.x), int(half.y)
    r = half.radius
    alpha = half.opacity()

    # Base color faded by opacity
    color = tuple(int(c * alpha) for c in half.color)
    cv2.circle(frame, (cx, cy), r, color, -1)

    # Inner pulp detail for specific fruits
    pulp_alpha = alpha * 0.9
    if half.color == (50, 180, 50):  # Watermelon -> red pulp with seeds
        pulp_color = (int(40 * pulp_alpha), int(40 * pulp_alpha), int(230 * pulp_alpha))
        cv2.circle(frame, (cx, cy), int(r * 0.75), pulp_color, -1)
        # Tiny black seeds
        for sx, sy in [(-r//4, 0), (r//4, -r//4), (0, r//4)]:
            cv2.circle(frame, (cx + sx, cy + sy), 2, (0, 0, 0), -1)
    elif half.color == (10, 10, 220):  # Apple -> pale yellow core
        core_color = (int(180 * pulp_alpha), int(240 * pulp_alpha), int(255 * pulp_alpha))
        cv2.circle(frame, (cx, cy), int(r * 0.7), core_color, -1)

    # Slice cut line through the middle
    dx = int(r * math.cos(half.angle))
    dy = int(r * math.sin(half.angle))
    cv2.line(frame, (cx - dx, cy - dy), (cx + dx, cy + dy), (240, 240, 240), 2)


def draw_particle(cv2, frame, particle):
    """Draw a juice splatter particle fading out."""
    cx, cy = int(particle.x), int(particle.y)
    r = max(1, int(particle.radius))
    alpha = particle.opacity()
    color = tuple(int(c * alpha) for c in particle.color)
    cv2.circle(frame, (cx, cy), r, color, -1)


def draw_flash(cv2, frame, flash):
    """Draw a glowing slice line flash."""
    alpha = flash.opacity()
    color = tuple(int(c * alpha) for c in flash.color)
    cv2.line(frame, (flash.x1, flash.y1), (flash.x2, flash.y2), color, 4)
    cv2.line(frame, (flash.x1, flash.y1), (flash.x2, flash.y2), (255, 255, 255), 2)


def draw_popup(cv2, frame, popup):
    """Draw a floating text popup."""
    cx, cy = int(popup.x), int(popup.y)
    alpha = popup.opacity()
    color = tuple(int(c * alpha) for c in popup.color)
    _outlined_text(cv2, frame, popup.text, (cx - 20, cy), popup.scale, color, 2)


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

    # Mode label
    _outlined_text(cv2, frame, "FRUIT NINJA", (w - 220, 35), 0.7, (0, 200, 255))

    # Combo display
    if game._combo_display and time.time() - game._combo_time < 1.5:
        _outlined_text(cv2, frame, game._combo_display, (w // 2 - 100, h // 2), 1.2, (0, 255, 255), 3)





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
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
    )

    game = FruitNinjaGame()
    window = "Fruit Ninja (q/Esc to quit)"
    sized = False

    # Trail history per hand (up to 2 hands)
    trails = [collections.deque(maxlen=12) for _ in range(2)]
    prev_tips = [None, None]   # previous frame fingertip positions
    prev2_tips = [None, None]  # 2 frames ago position for velocity extrapolation

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

            # Velocity extrapolation for 1 frame if hand was missed due to fast motion blur
            for i in range(2):
                if cur_tips[i] is None and prev_tips[i] is not None and prev2_tips[i] is not None:
                    vx = prev_tips[i][0] - prev2_tips[i][0]
                    vy = prev_tips[i][1] - prev2_tips[i][1]
                    # Only extrapolate if the hand was moving fast (> 15px/frame)
                    if math.hypot(vx, vy) > 15:
                        extrapolated = (prev_tips[i][0] + vx, prev_tips[i][1] + vy)
                        cur_tips[i] = extrapolated

            # Slice detection & trail management for each hand
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

            prev2_tips = list(prev_tips)
            prev_tips = list(cur_tips)

            # Update game physics
            game.update(w, h, dt)

            # Draw fruits
            for fruit in game.fruits:
                draw_fruit(cv2, frame, fruit)

            # Draw sliced halves
            for half in game.halves:
                draw_half(cv2, frame, half)

            # Draw juice particles
            for particle in game.particles:
                draw_particle(cv2, frame, particle)

            # Draw slice flashes
            for flash in game.flashes:
                draw_flash(cv2, frame, flash)

            # Draw floating popups
            for popup in game.popups:
                draw_popup(cv2, frame, popup)

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

            if not sized:
                display.open_window(cv2, window, frame)
                sized = True
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
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
    g3.score = 5
    g3.fruits.append(Fruit(100, 100, 0, 0, BOMB))
    g3.try_slice(50, 100, 150, 100)
    check("bomb deducts 3 from score", g3.score, 2)

    # Restart
    g2.restart()
    check("restart resets score", g2.score, 0)

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
