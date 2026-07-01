#!/usr/bin/env python3
"""
rps_game.py — Rock-Paper-Scissors you play with your hand on the webcam.

A game mode built on the same MediaPipe Hands pipeline as hand_counter.py: it
reuses that module's `fingers_up()` to read which fingers are extended, maps
that to a gesture, runs a countdown, picks a computer move, and keeps score.

This implements KAN-33:

  * Gesture detection — Rock (fingers curled), Paper (fingers extended),
    Scissors (index + middle extended, ring + pinky curled). The thumb is
    ignored because it is the noisiest finger to classify.
  * Round loop — waits for a hand to appear, then a countdown, then the
    player's gesture is locked in (mode of the last few frames), the computer
    picks randomly, and the winner shows. Each round re-arms by waiting again.
  * Score tracker — running wins / losses / draws.
  * Countdown timer before each round so you know when to show your hand.

Usage (run on your own machine — needs camera + display):

    python rps_game.py                   # default 3s countdown
    python rps_game.py --countdown 5     # longer countdown
    python rps_game.py --seed 0          # deterministic computer moves
    python rps_game.py --self-test       # verify game logic, no camera

Press 'q' or Esc in the window to quit.
"""

import argparse
import collections
import math
import random
import sys
import time

from hand_counter import fingers_up

GESTURES = ("rock", "paper", "scissors")

# Which gesture each gesture beats (key beats value).
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


# --- KAN-33: gesture classification ----------------------------------------
def classify_gesture(states):
    """Map a [thumb, index, middle, ring, pinky] up/down list to a gesture.

    Classification uses only the four fingers (thumb ignored): all curled is
    rock, all extended is paper, index+middle extended with ring+pinky curled
    is scissors. Anything else is "unknown" so callers can ask for a redo.
    """
    _thumb, index, middle, ring, pinky = states
    four = (index, middle, ring, pinky)
    if not any(four):
        return "rock"
    if all(four):
        return "paper"
    if index and middle and not ring and not pinky:
        return "scissors"
    return "unknown"


# --- KAN-33: winner logic --------------------------------------------------
def decide_winner(player, computer):
    """Return 'win' / 'lose' / 'draw' from the player's perspective."""
    if player == computer:
        return "draw"
    return "win" if BEATS[player] == computer else "lose"


class ScoreBoard:
    """Running tally of round outcomes from the player's perspective."""

    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def record(self, result):
        if result == "win":
            self.wins += 1
        elif result == "lose":
            self.losses += 1
        else:
            self.draws += 1


class GestureStabilizer:
    """Smooth a stream of per-frame gestures by taking the mode of a window.

    Ignores None / "unknown" frames so a hand mid-transition doesn't drag the
    locked-in gesture toward garbage. Returns "unknown" only when the whole
    window had nothing valid.
    """

    def __init__(self, window=5):
        self._history = collections.deque(maxlen=max(1, window))

    def update(self, gesture):
        self._history.append(gesture)
        return self.current()

    def current(self):
        valid = [g for g in self._history if g in GESTURES]
        if not valid:
            return "unknown"
        return collections.Counter(valid).most_common(1)[0][0]

    def clear(self):
        self._history.clear()


# --- KAN-33: round state machine -------------------------------------------
class RPSGame:
    """Drives countdown -> lock-in -> result -> repeat, and keeps score.

    `update(gesture, now)` is fed the current per-frame gesture (a string, or
    None when no hand is visible) and a monotonic timestamp. It returns a view
    dict describing what to render. Injecting an `rng` (random.Random) makes the
    computer's moves deterministic, which is what the self-test relies on.
    """

    WAITING = "waiting"
    COUNTDOWN = "countdown"
    RESULT = "result"

    def __init__(self, countdown_seconds=3.0, result_seconds=2.5, window=5,
                 rng=None):
        self.countdown_seconds = countdown_seconds
        self.result_seconds = result_seconds
        self.stabilizer = GestureStabilizer(window)
        self.score = ScoreBoard()
        self._rng = rng or random.Random()
        # Hold until a hand shows up, so the countdown never runs on an empty
        # frame; each round returns here to re-arm.
        self.phase = self.WAITING
        self._phase_start = None
        self.round = None          # last scored round: player/computer/result
        self.message = ""

    def update(self, gesture, now):
        if self._phase_start is None:
            self._phase_start = now
        elapsed = now - self._phase_start

        if self.phase == self.WAITING:
            # Start the round as soon as a hand is detected (any gesture).
            if gesture is not None:
                self._start_countdown(now)
                self.stabilizer.update(gesture)
                return self._view(count=max(1, math.ceil(self.countdown_seconds)),
                                  live=gesture)
            return self._view()

        if self.phase == self.COUNTDOWN:
            self.stabilizer.update(gesture)
            remaining = self.countdown_seconds - elapsed
            if remaining > 0:
                return self._view(count=max(1, math.ceil(remaining)), live=gesture)

            # Countdown hit zero: lock in the stabilized gesture.
            player = self.stabilizer.current()
            if player not in GESTURES:
                # Nothing clear to score — replay the round.
                self.message = "No clear gesture — try again!"
                self._start_countdown(now)
                return self._view(count=self.countdown_seconds, live=gesture)

            computer = self._rng.choice(GESTURES)
            result = decide_winner(player, computer)
            self.score.record(result)
            self.round = {"player": player, "computer": computer, "result": result}
            self.message = ""
            self.phase = self.RESULT
            self._phase_start = now
            return self._view()

        # RESULT phase: hold the outcome on screen, then wait for a hand again.
        remaining = self.result_seconds - elapsed
        if remaining <= 0:
            self._start_waiting(now)
            return self._view(live=gesture)
        return self._view()

    def _start_countdown(self, now):
        self.phase = self.COUNTDOWN
        self._phase_start = now
        self.stabilizer.clear()

    def _start_waiting(self, now):
        self.phase = self.WAITING
        self._phase_start = now
        self.stabilizer.clear()
        self.message = ""

    def _view(self, count=None, live=None):
        return {
            "phase": self.phase,
            "count": count,
            "live": live,
            "round": self.round,
            "score": self.score,
            "message": self.message,
        }


# --- capture + game loop ---------------------------------------------------
def run_game(args):
    import cv2
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[FAIL] could not open camera index {args.camera}")
        return 2

    # One hand for the game.
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    rng = random.Random(args.seed) if args.seed is not None else None
    game = RPSGame(countdown_seconds=args.countdown,
                   result_seconds=args.result_hold,
                   window=args.window, rng=rng)
    print("Rock-Paper-Scissors — press 'q' or Esc to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[warn] dropped a frame from the camera")
                continue

            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            gesture = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                label = "Right"
                if results.multi_handedness:
                    label = results.multi_handedness[0].classification[0].label
                gesture = classify_gesture(fingers_up(hand_landmarks.landmark, label))
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            view = game.update(gesture, time.monotonic())
            draw_game_overlay(cv2, frame, view)

            cv2.imshow("Rock Paper Scissors (q/Esc to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
    return 0


def draw_game_overlay(cv2, frame, view):
    """Render the score, countdown, live gesture, and round result."""
    h, w = frame.shape[:2]
    score = view["score"]

    # Score band along the top.
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame,
                f"Wins {score.wins}   Losses {score.losses}   Draws {score.draws}",
                (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)

    if view["phase"] == RPSGame.WAITING:
        cv2.putText(frame, "Show your hand to start!", (w // 2 - 230, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
    elif view["phase"] == RPSGame.COUNTDOWN:
        if view["count"] is not None:
            text = str(int(view["count"]))
            cv2.putText(frame, text, (w // 2 - 30, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 255), 8, cv2.LINE_AA)
        cv2.putText(frame, "Show your hand!", (w // 2 - 170, h // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        if view["live"]:
            cv2.putText(frame, f"Detected: {view['live']}", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2,
                        cv2.LINE_AA)
    else:  # RESULT
        rnd = view["round"]
        if rnd:
            colors = {"win": (0, 255, 0), "lose": (0, 0, 255), "draw": (0, 255, 255)}
            labels = {"win": "YOU WIN", "lose": "YOU LOSE", "draw": "DRAW"}
            cv2.putText(frame, f"You: {rnd['player']}    CPU: {rnd['computer']}",
                        (w // 2 - 240, h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, labels[rnd["result"]], (w // 2 - 140, h // 2 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, colors[rnd["result"]], 4,
                        cv2.LINE_AA)

    if view["message"]:
        cv2.putText(frame, view["message"], (15, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)


# --- self-test (no camera needed) ------------------------------------------
def self_test():
    """Verify gesture classification, winner logic, and the round machine."""
    from hand_counter import _make_hand

    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<34} expected {want}, got {got}")

    def gesture_of(**kwargs):
        return classify_gesture(fingers_up(_make_hand(**kwargs), "Right"))

    # Classify real finger poses (reusing hand_counter's synthetic hands).
    check("rock (fist)", gesture_of(right=True), "rock")
    check("paper (open palm)",
          gesture_of(thumb=True, index=True, middle=True, ring=True, pinky=True,
                     right=True), "paper")
    check("scissors (peace sign)",
          gesture_of(index=True, middle=True, right=True), "scissors")
    # Thumb is ignored, so a thumbs-up (all four fingers curled) reads as rock.
    check("rock (thumbs up, thumb ignored)", gesture_of(thumb=True, right=True), "rock")
    check("unknown (three fingers)",
          gesture_of(index=True, middle=True, ring=True, right=True), "unknown")

    # Winner logic across the board.
    check("rock beats scissors", decide_winner("rock", "scissors"), "win")
    check("rock loses to paper", decide_winner("rock", "paper"), "lose")
    check("paper beats rock", decide_winner("paper", "rock"), "win")
    check("scissors beats paper", decide_winner("scissors", "paper"), "win")
    check("scissors loses to rock", decide_winner("scissors", "rock"), "lose")
    check("same gesture draws", decide_winner("paper", "paper"), "draw")

    # The game waits for a hand before the countdown starts.
    game0 = RPSGame(countdown_seconds=1.0, result_seconds=1.0, rng=random.Random(0))
    game0.update(None, now=0.0)  # no hand yet
    check("waits for a hand at start", game0.phase, RPSGame.WAITING)
    game0.update("rock", now=0.5)  # hand appears
    check("countdown starts once hand seen", game0.phase, RPSGame.COUNTDOWN)

    # A full round: hand detected, countdown, then lock-in scores exactly once.
    game = RPSGame(countdown_seconds=1.0, result_seconds=1.0, rng=random.Random(0))
    game.update("paper", now=0.0)  # hand seen -> countdown begins
    game.update("paper", now=0.5)
    game.update("paper", now=1.0)  # shoot
    scored = game.score.wins + game.score.losses + game.score.draws
    check("one round scored", scored, 1)
    check("phase is result after shoot", game.phase, RPSGame.RESULT)
    check("player gesture locked in", game.round["player"], "paper")

    # An unclear gesture at shoot time replays without scoring.
    game2 = RPSGame(countdown_seconds=1.0, result_seconds=1.0, window=2,
                    rng=random.Random(0))
    game2.update("rock", now=0.0)  # hand seen -> countdown begins
    game2.update(None, now=0.4)    # hand lost
    game2.update(None, now=0.8)    # window fully stale -> unknown
    game2.update(None, now=1.0)    # shoot with nothing clear -> replay
    scored2 = game2.score.wins + game2.score.losses + game2.score.draws
    check("unclear gesture not scored", scored2, 0)
    check("replays countdown on unclear gesture", game2.phase, RPSGame.COUNTDOWN)

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(
        description="Rock-Paper-Scissors with webcam hand gestures (OpenCV + MediaPipe)."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--countdown", type=float, default=3.0,
                        help="Countdown seconds before each round (default 3).")
    parser.add_argument("--result-hold", type=float, default=2.5,
                        help="Seconds to show each round's result (default 2.5).")
    parser.add_argument("--window", type=int, default=5,
                        help="Frames of gesture history used to lock in (default 5).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the computer's moves for reproducibility.")
    parser.add_argument("--no-flip", action="store_true",
                        help="Do not mirror the webcam image.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run game-logic checks without a camera, then exit.")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    return run_game(args)


if __name__ == "__main__":
    sys.exit(main())
