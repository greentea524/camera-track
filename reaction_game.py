#!/usr/bin/env python3
"""
reaction_game.py — hand-gesture reaction game using OpenCV + MediaPipe.

Targets spawn at random positions on the webcam feed. Touch them with your
index fingertip to score points before the timer runs out. The game gets
progressively faster as your score climbs, and both hands are tracked so you
can play two-handed (or duel a friend).

Issue: #31 (original), #43 (multi-hand tracking + collision detection)

Usage:
    python reaction_game.py                # default settings (2 hands)
    python reaction_game.py --max-hands 1  # single-hand play
    python reaction_game.py --camera 1     # different camera index
    python reaction_game.py --self-test    # verify game logic, no camera

Press 'q' or Esc in the video window to quit.
"""

import argparse
import math
import random
import sys
import time

import display

# MediaPipe index-fingertip landmark
INDEX_TIP = 8


# ---------------------------------------------------------------------------
# Game objects
# ---------------------------------------------------------------------------

class Target:
    """A single circular target that the player must touch."""

    def __init__(self, x, y, radius, lifetime):
        self.x = x
        self.y = y
        self.radius = radius
        self.lifetime = lifetime          # seconds before it expires
        self.spawn_time = time.time()

    def is_expired(self):
        return time.time() - self.spawn_time >= self.lifetime

    def time_remaining_ratio(self):
        """Return 1.0 (fresh) → 0.0 (about to expire)."""
        elapsed = time.time() - self.spawn_time
        return max(0.0, 1.0 - elapsed / self.lifetime)

    def contains(self, px, py):
        """Return True if point (px, py) is inside this target."""
        return math.hypot(px - self.x, py - self.y) <= self.radius


class ReactionGame:
    """Manages scoring, difficulty, and target lifecycle."""

    def __init__(self):
        self.score = 0
        self.misses = 0
        self.lives = 5
        self.target = None
        self.game_over = False
        self._last_spawn = 0.0
        self._spawn_delay = 0.6          # brief pause between targets

    # -- difficulty ramp -----------------------------------------------------
    @property
    def target_radius(self):
        """Shrink targets as score rises (min 30 px)."""
        return max(30, 55 - self.score * 2)

    @property
    def target_lifetime(self):
        """Reduce lifetime as score rises (min 1.0 s)."""
        return max(1.0, 3.0 - self.score * 0.12)

    # -- logic ---------------------------------------------------------------
    def update(self, frame_w, frame_h, finger_x=None, finger_y=None, fingers=None):
        """Call once per frame. Returns a status string for the HUD."""
        if self.game_over:
            return "GAME OVER"

        now = time.time()

        # Spawn a new target if there isn't one and the cooldown has passed.
        if self.target is None:
            if now - self._last_spawn >= self._spawn_delay:
                margin = self.target_radius + 10
                x = random.randint(margin, max(margin, frame_w - margin))
                y = random.randint(margin + 120, max(margin + 120, frame_h - margin))
                self.target = Target(x, y, self.target_radius, self.target_lifetime)
            return "Get ready..."

        # Check expiry.
        if self.target.is_expired():
            self.misses += 1
            self.lives -= 1
            self.target = None
            self._last_spawn = now
            if self.lives <= 0:
                self.game_over = True
                return "GAME OVER"
            return "Missed!"

        # Combine single fingertip params into fingers list
        all_fingers = list(fingers) if fingers else []
        if finger_x is not None and finger_y is not None:
            all_fingers.append((finger_x, finger_y))

        # Check hit across all active fingertips.
        if all_fingers:
            for fx, fy in all_fingers:
                if fx is not None and fy is not None and self.target.contains(fx, fy):
                    self.score += 1
                    self.target = None
                    self._last_spawn = now
                    return "Hit!"

        return ""

    def restart(self):
        self.__init__()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _outlined_text(cv2, frame, text, pos, scale, fg, thickness=2):
    """Draw text with a black outline for readability on any background."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness)


def draw_hud(cv2, frame, game, status):
    """Render the score, lives, and status bar."""
    h, w = frame.shape[:2]

    # Semi-transparent header band
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    _outlined_text(cv2, frame, f"Score: {game.score}", (20, 40), 0.9, (0, 255, 200))
    _outlined_text(cv2, frame, f"Lives: {'*' * game.lives}", (20, 80), 0.8, (100, 180, 255))

    # Difficulty indicator
    _outlined_text(cv2, frame, f"Level: {game.score // 5 + 1}", (w - 200, 40), 0.7, (200, 200, 200))

    if status:
        color = (0, 255, 0) if status == "Hit!" else (0, 100, 255) if "Miss" in status else (255, 255, 255)
        _outlined_text(cv2, frame, status, (w // 2 - 80, h - 30), 0.9, color)


def draw_target(cv2, frame, target):
    """Draw the target with a shrinking ring to show remaining time."""
    if target is None:
        return
    ratio = target.time_remaining_ratio()

    # Color fades from green → yellow → red as time runs out
    if ratio > 0.5:
        g = 255
        r = int((1.0 - ratio) * 2 * 255)
    else:
        r = 255
        g = int(ratio * 2 * 255)

    # Filled circle
    cv2.circle(frame, (target.x, target.y), target.radius, (0, g, r), -1)
    # Outer ring showing time left
    ring_radius = int(target.radius + 8 * ratio)
    cv2.circle(frame, (target.x, target.y), ring_radius, (255, 255, 255), 2)
    # Inner dot
    cv2.circle(frame, (target.x, target.y), 5, (255, 255, 255), -1)


HAND_COLORS = [(255, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0)]

def draw_fingertip(cv2, frame, fx, fy, hand_idx=0):
    """Draw a crosshair at a detected index fingertip."""
    if fx is None or fy is None:
        return
    color = HAND_COLORS[hand_idx % len(HAND_COLORS)]
    cv2.circle(frame, (fx, fy), 14, color, 2)
    cv2.line(frame, (fx - 18, fy), (fx + 18, fy), color, 1)
    cv2.line(frame, (fx, fy - 18), (fx, fy + 18), color, 1)


def draw_game_over(cv2, frame, game):
    """Overlay a game-over screen with final score and restart prompt."""
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
        max_num_hands=args.max_hands,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    game = ReactionGame()
    window = "Reaction Game (q/Esc to quit)"
    sized = False
    print(f"Reaction Game starting (max {args.max_hands} hands) — press 'q' or Esc to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # MediaPipe hand detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            fingers = []
            if results.multi_hand_landmarks:
                for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks):
                    if hand_idx >= args.max_hands:
                        break
                    # Draw hand skeleton
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    # Get index fingertip in pixel coords
                    tip = hand_lm.landmark[INDEX_TIP]
                    fx = int(tip.x * w)
                    fy = int(tip.y * h)
                    fingers.append((fx, fy))

            # Update game state
            status = game.update(w, h, fingers=fingers)

            # Draw everything
            draw_target(cv2, frame, game.target)
            for idx, (fx, fy) in enumerate(fingers):
                draw_fingertip(cv2, frame, fx, fy, hand_idx=idx)
            draw_hud(cv2, frame, game, status)

            if game.game_over:
                draw_game_over(cv2, frame, game)

            if not sized:
                display.open_window(cv2, window, frame, args.display_scale)
                sized = True
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" ") and game.game_over:
                game.restart()
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
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<40} expected {want}, got {got}")

    # Target hit detection
    t = Target(100, 100, 40, 5.0)
    check("point inside target", t.contains(110, 105), True)
    check("point outside target", t.contains(200, 200), False)
    check("point on edge", t.contains(140, 100), True)

    # Game scoring
    g = ReactionGame()
    check("initial score", g.score, 0)
    check("initial lives", g.lives, 5)
    check("not game over initially", g.game_over, False)

    # Difficulty ramp
    check("initial radius", g.target_radius, 55)
    g.score = 10
    check("radius shrinks with score", g.target_radius, 35)
    g.score = 20
    check("radius has a floor of 30", g.target_radius, 30)

    # Multi-hand update test
    g.score = 0
    g.target = Target(100, 100, 40, 5.0)
    # Hand 1 misses at (200, 200), Hand 2 hits at (105, 105)
    st = g.update(640, 480, fingers=[(200, 200), (105, 105)])
    check("multi-hand hit registered", st, "Hit!")
    check("score incremented by multi-hand hit", g.score, 1)

    # Multi-hand miss: no fingertip inside, target survives, score unchanged.
    g.score = 0
    g.target = Target(100, 100, 40, 5.0)
    st = g.update(640, 480, fingers=[(300, 300), (400, 300)])
    check("multi-hand miss leaves target", st, "")
    check("score unchanged on multi-hand miss", g.score, 0)
    check("target still alive on miss", g.target is not None, True)

    # Backward-compatible single-fingertip args still work.
    st = g.update(640, 480, finger_x=105, finger_y=105)
    check("single-fingertip args still hit", st, "Hit!")
    check("score incremented via legacy args", g.score, 1)

    # Restart
    g.restart()
    check("restart resets score", g.score, 0)
    check("restart resets lives", g.lives, 5)

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
        description="Hand-gesture reaction game (OpenCV + MediaPipe).")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0).")
    parser.add_argument("--max-hands", type=int, default=2,
                        help="Max hands to track (default 2 for dual-hand play).")
    parser.add_argument("--display-scale", type=float, default=1.5,
                        help="Window scale factor (default 1.5).")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
