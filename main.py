#!/usr/bin/env python3
"""
main.py — unified launcher for the camera-track apps (KAN-35).

Presents a terminal menu to run any of the project's apps, then hands off to
that app's own `main()`. Because each app imports OpenCV / MediaPipe lazily
(inside its run loop), importing them all here is cheap and needs no camera —
the menu itself is hardware-free.

Usage:

    python main.py                 # show the interactive menu
    python main.py eyes            # jump straight to an app by name
    python main.py 3 --seed 0      # by number, passing flags through to the app
    python main.py --self-test     # verify the menu wiring, no camera

Each app quits back to the menu on 'q'/Esc; from the menu, 'q' quits.
"""

import sys

import hand_counter
import eye_tracker
import rps_game
import verify_camera
import mood_meter
import reaction_game

# Ordered menu: number -> (name, description, app entry point). The name is the
# keyword accepted on the command line; the number is what the menu prompts for.
APPS = [
    ("hands", "Finger counter (count 0-10 fingers)", hand_counter.main),
    ("eyes", "Eye tracker (gaze / blink / drowsiness)", eye_tracker.main),
    ("rps", "Rock-Paper-Scissors game", rps_game.main),
    ("mood", "Mood Meter (0-100 emotion detection)", mood_meter.main),
    ("react", "Reaction Game (touch targets with your hand)", reaction_game.main),
    ("check", "Camera / environment check", verify_camera.main),
]


def select(choice):
    """Map a menu choice to an app entry point, or None if it doesn't match.

    Accepts the 1-based menu number ("1".."4") or the app's keyword name
    (case-insensitive, e.g. "eyes"). Pure and hardware-free, so it can be
    unit-tested without launching anything.
    """
    if choice is None:
        return None
    key = choice.strip().lower()
    if key.isdigit():
        idx = int(key) - 1
        if 0 <= idx < len(APPS):
            return APPS[idx][2]
        return None
    for name, _desc, entry in APPS:
        if key == name:
            return entry
    return None


def render_menu():
    """Return the menu text (kept separate so the self-test can inspect it)."""
    lines = ["", "camera-track - pick an app:", ""]
    for i, (name, desc, _entry) in enumerate(APPS, start=1):
        lines.append(f"  {i}. {desc}  [{name}]")
    lines.append("  q. Quit")
    lines.append("")
    return "\n".join(lines)


def menu_loop():
    """Interactive loop: show the menu, launch the chosen app, repeat."""
    while True:
        print(render_menu())
        try:
            choice = input("Choice: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if choice.strip().lower() in ("q", "quit", "exit"):
            return 0

        entry = select(choice)
        if entry is None:
            print(f"  '{choice.strip()}' is not a valid choice.\n")
            continue

        try:
            entry([])  # launch with the app's own defaults
        except KeyboardInterrupt:
            print("\n  (interrupted)")
        print("\n--- back to menu ---")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] in ("--self-test",):
        return self_test()

    if argv and argv[0] in ("-h", "--help"):
        print(render_menu())
        print("Run 'python main.py <name|number> [flags]' to launch directly.")
        return 0

    # Direct launch: first token picks the app, the rest passes through to it.
    if argv:
        entry = select(argv[0])
        if entry is None:
            print(f"Unknown app '{argv[0]}'.")
            print(render_menu())
            return 1
        return entry(argv[1:])

    return menu_loop()


def self_test():
    """Verify the menu mapping without launching anything or touching a camera."""
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<34} expected {want}, got {got}")

    # Numbers map to the apps in order.
    check("choice '1' -> hands", select("1"), hand_counter.main)
    check("choice '2' -> eyes", select("2"), eye_tracker.main)
    check("choice '3' -> rps", select("3"), rps_game.main)
    check("choice '4' -> mood", select("4"), mood_meter.main)
    check("choice '5' -> react", select("5"), reaction_game.main)
    check("choice '6' -> check", select("6"), verify_camera.main)

    # Names map too, case-insensitively and with surrounding whitespace.
    check("name 'eyes' -> eye_tracker", select("eyes"), eye_tracker.main)
    check("name ' RPS ' -> rps_game", select("  RPS  "), rps_game.main)

    # Invalid choices return None.
    check("out-of-range number", select("9"), None)
    check("unknown name", select("nope"), None)
    check("empty choice", select(""), None)
    check("None choice", select(None), None)

    # Every app is reachable by both its number and its name, and the menu
    # lists each one.
    menu = render_menu()
    for i, (name, _desc, entry) in enumerate(APPS, start=1):
        check(f"'{name}' listed in menu", name in menu, True)
        check(f"'{name}' reachable by number", select(str(i)), entry)
        check(f"'{name}' reachable by name", select(name), entry)

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
