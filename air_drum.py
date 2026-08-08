#!/usr/bin/env python3
"""
air_drum.py — virtual drum kit controlled by hand gestures.

Uses MediaPipe Hands to track both hands. The camera view is divided into
drum-pad zones. When a fingertip "strikes" a zone (moves downward past a
velocity threshold), the corresponding drum sound plays.

Issue: #32

Usage:
    python air_drum.py                 # default settings
    python air_drum.py --camera 1      # different camera index
    python air_drum.py --self-test     # verify game logic, no camera

Press 'q' or Esc in the video window to quit.
"""

import argparse
import collections
import math
import sys
import threading
import time

import numpy as np

import display

# MediaPipe landmark indices
WRIST = 0
INDEX_TIP = 8
MIDDLE_TIP = 12

# Audio synthesis settings
SAMPLE_RATE = 44100

# Drum pads. "voice" selects the synthesised percussion sample; "freq"/"dur"
# are only used by the winsound fallback, which can emit tones but not noise.
DRUM_PADS = [
    {"name": "Hi-Hat", "voice": "hihat", "freq": 800, "dur": 80,  "color": (0, 220, 220)},
    {"name": "Snare",  "voice": "snare", "freq": 400, "dur": 100, "color": (100, 180, 255)},
    {"name": "Tom",    "voice": "tom",   "freq": 300, "dur": 120, "color": (180, 100, 255)},
    {"name": "Kick",   "voice": "kick",  "freq": 150, "dur": 150, "color": (255, 100, 100)},
    {"name": "Crash",  "voice": "crash", "freq": 600, "dur": 90,  "color": (100, 255, 100)},
    {"name": "Ride",   "voice": "ride",  "freq": 700, "dur": 70,  "color": (255, 200, 50)},
]

# Piano key pads (7 diatonic notes C4-B4)
PIANO_PADS = [
    {"name": "C4 (Do)",  "voice": "piano_c4", "freq": 261.6, "dur": 300, "color": (255, 100, 100)},
    {"name": "D4 (Re)",  "voice": "piano_d4", "freq": 293.7, "dur": 300, "color": (255, 180, 100)},
    {"name": "E4 (Mi)",  "voice": "piano_e4", "freq": 329.6, "dur": 300, "color": (255, 255, 100)},
    {"name": "F4 (Fa)",  "voice": "piano_f4", "freq": 349.2, "dur": 300, "color": (100, 255, 100)},
    {"name": "G4 (Sol)", "voice": "piano_g4", "freq": 392.0, "dur": 300, "color": (100, 220, 255)},
    {"name": "A4 (La)",  "voice": "piano_a4", "freq": 440.0, "dur": 300, "color": (150, 100, 255)},
    {"name": "B4 (Ti)",  "voice": "piano_b4", "freq": 493.9, "dur": 300, "color": (255, 100, 255)},
]

# Strike velocity (normalized units/sec) mapped onto playback gain. A strike at
# the detection threshold plays at MIN_GAIN; VELOCITY_CEILING and above plays
# at full volume.
VELOCITY_CEILING = 4.0
MIN_GAIN = 0.35


# ---------------------------------------------------------------------------
# Drum sample synthesis
#
# Samples are generated with numpy at startup rather than shipped as .wav
# assets, so the repo stays free of binary files and the kit stays tweakable.
# Each synth returns a mono float32 array in [-1, 1].
# ---------------------------------------------------------------------------

def _envelope(n, decay, attack=64):
    """Exponential decay envelope with a short attack ramp (declicks the onset)."""
    t = np.arange(n) / SAMPLE_RATE
    env = np.exp(-t / decay)
    a = min(attack, n)
    if a > 0:
        env[:a] *= np.linspace(0.0, 1.0, a)
    return env


def _swept_sine(n, f_start, f_end, sweep_time):
    """Sine whose frequency decays exponentially from f_start to f_end."""
    t = np.arange(n) / SAMPLE_RATE
    freq = f_end + (f_start - f_end) * np.exp(-t / sweep_time)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    return np.sin(phase)


def _noise(n, rng, brightness=1):
    """White noise, optionally differentiated to tilt energy toward high frequencies."""
    sig = rng.standard_normal(n)
    for _ in range(brightness):
        sig = np.diff(sig, prepend=0.0)
    return sig


def _normalize(sig, peak=0.95):
    """Scale to a target peak so pads sit at comparable loudness."""
    m = float(np.max(np.abs(sig)))
    if m < 1e-9:
        return sig.astype(np.float32)
    return (sig * (peak / m)).astype(np.float32)


def _synth_kick(rng):
    n = int(SAMPLE_RATE * 0.38)
    t = np.arange(n) / SAMPLE_RATE
    body = _swept_sine(n, 120.0, 45.0, 0.06) * _envelope(n, 0.13)
    click = _noise(n, rng) * np.exp(-t / 0.004) * 0.35
    return _normalize(body + click)


def _synth_snare(rng):
    """Noise rattle over a tuned drum body.

    The body carries most of the level: a rattle-only snare reads as hiss
    rather than a drum, so the 185/330 Hz pair is kept prominent.
    """
    n = int(SAMPLE_RATE * 0.22)
    t = np.arange(n) / SAMPLE_RATE
    rattle = _noise(n, rng, brightness=1) * _envelope(n, 0.070) * 0.30
    body = (np.sin(2 * np.pi * 185 * t) * 1.0 + np.sin(2 * np.pi * 330 * t) * 0.55)
    body *= _envelope(n, 0.085)
    return _normalize(rattle + body)


def _synth_hihat(rng):
    n = int(SAMPLE_RATE * 0.07)
    return _normalize(_noise(n, rng, brightness=2) * _envelope(n, 0.018))


def _synth_tom(rng):
    n = int(SAMPLE_RATE * 0.30)
    body = _swept_sine(n, 180.0, 95.0, 0.08) * _envelope(n, 0.11)
    skin = _noise(n, rng) * _envelope(n, 0.012) * 0.2
    return _normalize(body + skin)


def _synth_crash(rng):
    n = int(SAMPLE_RATE * 1.40)
    t = np.arange(n) / SAMPLE_RATE
    wash = _noise(n, rng, brightness=1) * _envelope(n, 0.55)
    shimmer = _noise(n, rng, brightness=2) * np.exp(-t / 0.20) * 0.6
    return _normalize(wash + shimmer)


def _synth_ride(rng):
    n = int(SAMPLE_RATE * 0.90)
    t = np.arange(n) / SAMPLE_RATE
    ping = (np.sin(2 * np.pi * 720 * t) + np.sin(2 * np.pi * 1180 * t) * 0.6)
    ping *= _envelope(n, 0.10)
    wash = _noise(n, rng, brightness=2) * _envelope(n, 0.35) * 0.55
    return _normalize(ping * 0.8 + wash)


def _synth_piano_note(freq):
    """Synthesize acoustic piano note with harmonics and natural decay."""
    n = int(SAMPLE_RATE * 1.0)
    t = np.arange(n) / SAMPLE_RATE
    sound = (
        np.sin(2 * np.pi * freq * t) * 1.0 +
        np.sin(2 * np.pi * freq * 2 * t) * 0.40 +
        np.sin(2 * np.pi * freq * 3 * t) * 0.18 +
        np.sin(2 * np.pi * freq * 4 * t) * 0.06
    )
    return _normalize(sound * _envelope(n, decay=0.35, attack=32))


_SYNTHS = {
    "kick": _synth_kick,
    "snare": _synth_snare,
    "hihat": _synth_hihat,
    "tom": _synth_tom,
    "crash": _synth_crash,
    "ride": _synth_ride,
    "piano_c4": lambda rng: _synth_piano_note(261.63),
    "piano_d4": lambda rng: _synth_piano_note(293.66),
    "piano_e4": lambda rng: _synth_piano_note(329.63),
    "piano_f4": lambda rng: _synth_piano_note(349.23),
    "piano_g4": lambda rng: _synth_piano_note(392.00),
    "piano_a4": lambda rng: _synth_piano_note(440.00),
    "piano_b4": lambda rng: _synth_piano_note(493.88),
}


def build_samples(seed=7):
    """Render every drum voice. Returns {voice_name: float32 mono array}."""
    rng = np.random.default_rng(seed)
    return {name: fn(rng) for name, fn in _SYNTHS.items()}


def velocity_to_gain(velocity, threshold):
    """Map a strike velocity onto a 0..1 playback gain.

    Anything at or below the detection threshold plays at MIN_GAIN; the gain
    rises linearly to 1.0 at VELOCITY_CEILING so harder hits are louder.
    """
    if velocity is None or velocity <= threshold:
        return MIN_GAIN
    span = max(VELOCITY_CEILING - threshold, 1e-6)
    frac = min(1.0, (velocity - threshold) / span)
    return MIN_GAIN + (1.0 - MIN_GAIN) * frac


# ---------------------------------------------------------------------------
# Playback backends
#
# pygame.mixer is preferred: it is cross-platform and mixes overlapping hits,
# which matters for a drum kit. winsound is a Windows-only fallback that can
# only manage single square-wave beeps. If neither is available the kit still
# runs silently, but says so in the HUD instead of failing quietly.
# ---------------------------------------------------------------------------

class AudioEngine:
    """Plays drum samples with velocity-scaled volume."""

    def __init__(self):
        self.backend = "silent"
        self.status = "no audio backend"
        self._sounds = {}
        self._mixer = None

    def start(self):
        if self._start_pygame():
            return self
        if self._start_winsound():
            return self
        self.backend = "silent"
        self.status = "silent (install pygame for sound)"
        return self

    def _start_pygame(self):
        try:
            import pygame
        except Exception:
            return False
        try:
            # Buffer size 128 (~2.9ms latency at 44.1kHz) for instantaneous response
            pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=128)
            pygame.mixer.init()
            # Enough channels that fast fills and both hands never cut each other off.
            pygame.mixer.set_num_channels(32)
        except Exception as exc:
            print(f"[warn] pygame mixer unavailable ({exc}); trying fallback.")
            return False

        for voice, mono in build_samples().items():
            stereo = np.repeat((np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)[:, None], 2, axis=1)
            self._sounds[voice] = pygame.sndarray.make_sound(np.ascontiguousarray(stereo))

        self._mixer = pygame
        self.backend = "pygame"
        self.status = "pygame mixer"
        return True

    def _start_winsound(self):
        try:
            import winsound  # noqa: F401  (Windows only)
        except Exception:
            return False
        self.backend = "winsound"
        self.status = "winsound beeps (no samples)"
        return True

    def play(self, pad, velocity=None, threshold=0.0):
        """Fire-and-forget playback of one pad at a velocity-scaled volume."""
        gain = velocity_to_gain(velocity, threshold)

        if self.backend == "pygame":
            snd = self._sounds.get(pad["voice"])
            if snd is None:
                return
            ch = snd.play()
            if ch and gain < 1.0:
                ch.set_volume(gain)
        elif self.backend == "winsound":
            threading.Thread(
                target=self._beep, args=(pad["freq"], pad["dur"]), daemon=True
            ).start()

    @staticmethod
    def _beep(freq, duration_ms):
        try:
            import winsound
            winsound.Beep(int(freq), int(duration_ms))
        except Exception as exc:
            print(f"[warn] beep failed: {exc}")

    def close(self):
        if self.backend == "pygame" and self._mixer is not None:
            try:
                self._mixer.mixer.quit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Strike detection
# ---------------------------------------------------------------------------

class StrikeDetector:
    """Detect a downward 'strike' motion of a fingertip.

    Tracks the y-position of a landmark over frames. A strike is registered
    when the tip moves downward faster than `velocity_threshold` (in
    normalized coordinates per second) and then a cooldown prevents
    repeated triggers.
    """

    def __init__(self, velocity_threshold=0.8, cooldown=0.25):
        self.velocity_threshold = velocity_threshold
        self.cooldown = cooldown
        # Velocity of the most recent strike, so callers can scale hit volume.
        self.last_velocity = 0.0
        self._prev_y = None
        self._prev_time = None
        self._last_strike = 0.0

    def update(self, y):
        """Feed a new normalized y value. Returns True if a strike occurred.

        On a strike, `last_velocity` holds the downward speed that triggered it.
        """
        now = time.time()

        if self._prev_y is None:
            self._prev_y = y
            self._prev_time = now
            return False

        dt = now - self._prev_time
        if dt <= 0:
            return False

        velocity = (y - self._prev_y) / dt   # positive = moving downward
        self._prev_y = y
        self._prev_time = now

        if velocity > self.velocity_threshold and (now - self._last_strike) > self.cooldown:
            self._last_strike = now
            self.last_velocity = velocity
            return True

        return False

    def reset(self):
        self._prev_y = None
        self._prev_time = None


# ---------------------------------------------------------------------------
# Pad layout
# ---------------------------------------------------------------------------

def get_pad_regions(frame_w, frame_h, num_pads):
    """Divide the lower portion of the frame into evenly-spaced pad rectangles.

    Returns a list of (x1, y1, x2, y2) tuples.
    """
    pad_zone_top = int(frame_h * 0.35)   # pads occupy the lower 65%
    pad_height = frame_h - pad_zone_top
    pad_width = frame_w // num_pads
    regions = []
    for i in range(num_pads):
        x1 = i * pad_width
        x2 = (i + 1) * pad_width
        regions.append((x1, pad_zone_top, x2, frame_h))
    return regions


def find_pad_at(px, py, regions):
    """Return the index of the pad containing pixel (px, py), or -1."""
    for i, (x1, y1, x2, y2) in enumerate(regions):
        if x1 <= px < x2 and y1 <= py < y2:
            return i
    return -1


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

class FloatingNote:
    """A floating note text indicator above hit pads."""

    def __init__(self, x, y, text, color=(255, 255, 255)):
        self.x = float(x)
        self.y = float(y)
        self.text = text
        self.color = color
        self.spawn_time = time.time()
        self.lifetime = 0.7
        self.vy = -70.0  # float upward

    def update(self, dt):
        self.y += self.vy * dt

    def is_expired(self):
        return time.time() - self.spawn_time > self.lifetime

    def opacity(self):
        return max(0.0, 1.0 - (time.time() - self.spawn_time) / self.lifetime)


def _outlined_text(cv2, frame, text, pos, scale, fg, thickness=2):
    """Draw text with a black outline for readability."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness)


def draw_pads(cv2, frame, regions, flash_indices, active_pads):
    """Draw the drum / piano pad zones on the frame."""
    for i, (x1, y1, x2, y2) in enumerate(regions):
        if i >= len(active_pads):
            break
        pad = active_pads[i]
        color = pad["color"]

        # Flash brighter when hit
        if i in flash_indices:
            bright = tuple(min(255, c + 80) for c in color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bright, -1)
        else:
            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        text_x = x1 + 8
        text_y = y2 - 15
        _outlined_text(cv2, frame, pad["name"], (text_x, text_y), 0.5, (255, 255, 255), 1)


def draw_hud(cv2, frame, hit_count, audio_status=None, mode_name="Drums", key_history=None):
    """Render the top HUD bar and recent key history."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    title = f"Air Instruments [{mode_name}]"
    _outlined_text(cv2, frame, title, (20, 35), 0.9, (0, 220, 220))
    _outlined_text(cv2, frame, "Press 'M' to switch instrument", (w // 2 - 130, 35), 0.6, (255, 255, 0))
    _outlined_text(cv2, frame, f"Hits: {hit_count}", (w - 150, 35), 0.7, (200, 200, 200))

    # Surface the audio backend so a silent kit is obvious rather than baffling.
    if audio_status:
        muted = audio_status.startswith("silent")
        color = (120, 120, 255) if muted else (160, 200, 160)
        _outlined_text(cv2, frame, f"Audio: {audio_status}", (20, 78), 0.5, color, 1)

    # Key history banner across the top under HUD
    if key_history:
        recent = list(key_history)[-6:]
        history_str = " -> ".join([name for name, _c in recent])
        _outlined_text(cv2, frame, f"Played: {history_str}", (w // 2 - 130, 78), 0.55, (0, 255, 255), 1)


def draw_fingertip(cv2, frame, fx, fy, color=(255, 0, 255)):
    """Draw a dot at the fingertip position."""
    if fx is None:
        return
    cv2.circle(frame, (fx, fy), 10, (0, 0, 0), -1)
    cv2.circle(frame, (fx, fy), 8, color, -1)


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(args):
    """Live capture + drum loop."""
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

    # One strike detector per hand (index 0 = first hand, 1 = second)
    detectors = [StrikeDetector() for _ in range(2)]
    hit_count = 0
    flash_until = {}   # pad_index -> time when flash expires

    modes = [("Drums", DRUM_PADS), ("Piano", PIANO_PADS)]
    current_mode_idx = 0

    key_history = collections.deque(maxlen=8)
    floating_notes = []

    audio = AudioEngine().start()

    window = "Air Instruments (q/Esc to quit, M to toggle Drums/Piano)"
    sized = False
    prev_time = time.time()
    print(f"Air Instruments starting — audio: {audio.status}. Press 'm' to switch instrument, 'q'/Esc to quit.")
    if audio.backend == "silent":
        print("[warn] no audio backend found; install pygame for drum sounds.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            dt = min(now - prev_time, 0.1)
            prev_time = now

            mode_name, active_pads = modes[current_mode_idx]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)

            regions = get_pad_regions(w, h, len(active_pads))

            # Determine which pads are currently flashing
            active_flashes = {k for k, v in flash_until.items() if now < v}

            if results.multi_hand_landmarks:
                for hand_idx, hand_lm in enumerate(results.multi_hand_landmarks):
                    if hand_idx >= len(detectors):
                        break

                    # Draw hand skeleton
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    # Track index fingertip
                    tip = hand_lm.landmark[INDEX_TIP]
                    fx = int(tip.x * w)
                    fy = int(tip.y * h)

                    # Choose color per hand
                    tip_color = (255, 0, 255) if hand_idx == 0 else (0, 255, 255)
                    draw_fingertip(cv2, frame, fx, fy, tip_color)

                    # Check for strike
                    detector = detectors[hand_idx]
                    if detector.update(tip.y):
                        pad_idx = find_pad_at(fx, fy, regions)
                        if 0 <= pad_idx < len(active_pads):
                            pad = active_pads[pad_idx]
                            audio.play(pad, detector.last_velocity,
                                       detector.velocity_threshold)
                            hit_count += 1
                            flash_until[pad_idx] = now + 0.15
                            active_flashes.add(pad_idx)

                            # Record key history & floating note popup
                            key_history.append((pad["name"], pad["color"]))
                            rx1, ry1, rx2, ry2 = regions[pad_idx]
                            floating_notes.append(FloatingNote((rx1 + rx2) // 2, ry1 + 30, pad["name"], pad["color"]))

            # Update floating notes
            for note in floating_notes:
                note.update(dt)
            floating_notes = [n for n in floating_notes if not n.is_expired()]

            draw_pads(cv2, frame, regions, active_flashes, active_pads)

            # Draw floating notes
            for note in floating_notes:
                alpha = note.opacity()
                c = tuple(int(ch * alpha) for ch in note.color)
                _outlined_text(cv2, frame, f"~ {note.text}", (int(note.x) - 30, int(note.y)), 0.7, c, 2)

            draw_hud(cv2, frame, hit_count, audio.status, mode_name, key_history)

            if not sized:
                display.open_window(cv2, window, frame)
                sized = True
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                current_mode_idx = (current_mode_idx + 1) % len(modes)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        hands.close()
        audio.close()
        cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    """Verify drum logic without a camera."""
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<44} expected {want}, got {got}")

    # Pad regions
    regions = get_pad_regions(600, 400, 6)
    check("6 pad regions created", len(regions), 6)
    check("first pad starts at x=0", regions[0][0], 0)
    check("last pad ends at x=600", regions[5][2], 600)

    # find_pad_at
    check("point in first pad", find_pad_at(50, 300, regions), 0)
    check("point in last pad", find_pad_at(550, 300, regions), 5)
    check("point above pad zone", find_pad_at(50, 50, regions), -1)

    # Strike detector
    sd = StrikeDetector(velocity_threshold=0.5, cooldown=0.0)
    sd._prev_y = 0.3
    sd._prev_time = time.time() - 0.05  # 50ms ago
    # Moving from 0.3 to 0.4 in 50ms => velocity = 0.1/0.05 = 2.0 (above threshold)
    check("fast downward = strike", sd.update(0.4), True)

    sd2 = StrikeDetector(velocity_threshold=0.5, cooldown=0.0)
    sd2._prev_y = 0.3
    sd2._prev_time = time.time() - 0.05
    # Moving from 0.3 to 0.29 => upward motion, not a strike
    check("upward motion = no strike", sd2.update(0.29), False)

    # Cooldown
    sd3 = StrikeDetector(velocity_threshold=0.5, cooldown=10.0)
    sd3._prev_y = 0.3
    sd3._prev_time = time.time() - 0.05
    sd3._last_strike = time.time()  # just struck
    check("cooldown prevents re-strike", sd3.update(0.4), False)

    # Strike velocity is recorded for volume scaling
    sd4 = StrikeDetector(velocity_threshold=0.5, cooldown=0.0)
    sd4._prev_y = 0.3
    sd4._prev_time = time.time() - 0.05
    sd4.update(0.4)   # ~2.0 units/sec
    check("strike records velocity", 1.5 < sd4.last_velocity < 2.5, True)

    # Velocity -> gain mapping
    check("at threshold = min gain", velocity_to_gain(0.8, 0.8), MIN_GAIN)
    check("below threshold = min gain", velocity_to_gain(0.1, 0.8), MIN_GAIN)
    check("at ceiling = full gain", velocity_to_gain(VELOCITY_CEILING, 0.8), 1.0)
    check("above ceiling clamps to 1.0", velocity_to_gain(99.0, 0.8), 1.0)
    mid = velocity_to_gain(2.4, 0.8)
    check("hard hit louder than soft", mid > velocity_to_gain(1.0, 0.8), True)
    check("mid gain stays in range", MIN_GAIN < mid < 1.0, True)

    # Synthesised samples
    samples = build_samples()
    check("one sample per voice", sorted(samples) == sorted(_SYNTHS), True)
    check("every drum pad has a voice",
          all(p["voice"] in samples for p in DRUM_PADS), True)
    check("every piano pad has a voice",
          all(p["voice"] in samples for p in PIANO_PADS), True)
    check("kick is longer than hi-hat",
          len(samples["kick"]) > len(samples["hihat"]), True)
    check("samples stay within [-1, 1]",
          all(float(np.max(np.abs(s))) <= 1.0 for s in samples.values()), True)
    check("samples are non-silent",
          all(float(np.max(np.abs(s))) > 0.5 for s in samples.values()), True)
    check("samples are finite",
          all(bool(np.all(np.isfinite(s))) for s in samples.values()), True)

    # A silent engine must still be safe to call
    engine = AudioEngine()
    engine.play(DRUM_PADS[0], 2.0, 0.8)   # no backend started; must not raise
    check("silent engine tolerates play()", engine.backend, "silent")

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "--self-test":
        return self_test()

    parser = argparse.ArgumentParser(
        description="Virtual air drums controlled by hand gestures.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0).")
    parser.add_argument("--display-scale", type=float, default=1.5,
                        help="Window scale factor (default 1.5).")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
