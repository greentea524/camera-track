#!/usr/bin/env python3
"""
cube_solver.py — pure-Python 3x3 Rubik's Cube model + beginner-method solver.

The solving core for the Rubik's Cube tracker's guidance mode (GitHub issue
#26). No third-party dependencies — just a facelet-level cube model, a move
engine for the six face turns, and a layer-by-layer beginner solve:

  1. bottom cross          (daisy -> plant each edge)
  2. bottom corners        (repeat corner trigger until seated)
  3. middle-layer edges    (left/right insert algorithms)
  4. top cross             (orient last-layer edges)
  5. top edges             (permute last-layer edges)
  6. top corners: position (3-cycle until placed)
  7. top corners: orient   (R' D' R D pairs)

Stages 4-6 are found with a tiny breadth-first search over a fixed alg set,
so they are correct by construction rather than by case analysis. The output
is a flat move sequence in standard face-turn notation (U R F D L B, with '
for counter-clockwise and 2 for half turns), simplified by merging adjacent
same-face turns. A beginner solve typically lands around 100-160 moves.

Faces are letters U R F D L B; a cube state is {face: [9 sticker letters]},
row-major as if looking straight at each face with:

  U viewed with B at the top, D viewed with F at the top,
  F/R/B/L each viewed with U at the top.

The solver doubles as scan validation: a state that is miscoloured or
physically impossible (twisted corner, flipped edge, swapped pieces) makes a
stage fail its bounded search, which raises ValueError instead of emitting a
bogus move list.

Run `python cube_solver.py` to self-test (no camera or OpenCV needed).
"""

import random
import sys
from collections import deque

FACES = "URFDLB"

# Clockwise rotation of a face's own 3x3 stickers: new[i] = old[_FACE_CW[i]].
_FACE_CW = [6, 3, 0, 7, 4, 1, 8, 5, 2]

# For each clockwise move, the four adjacent strips whose contents cycle
# A -> B -> C -> D -> A (the strip listed first moves onto the second, etc.).
_STRIPS = {
    "U": (("F", (0, 1, 2)), ("L", (0, 1, 2)), ("B", (0, 1, 2)), ("R", (0, 1, 2))),
    "D": (("F", (6, 7, 8)), ("R", (6, 7, 8)), ("B", (6, 7, 8)), ("L", (6, 7, 8))),
    "R": (("F", (2, 5, 8)), ("U", (2, 5, 8)), ("B", (6, 3, 0)), ("D", (2, 5, 8))),
    "L": (("U", (0, 3, 6)), ("F", (0, 3, 6)), ("D", (0, 3, 6)), ("B", (8, 5, 2))),
    "F": (("U", (6, 7, 8)), ("R", (0, 3, 6)), ("D", (2, 1, 0)), ("L", (8, 5, 2))),
    "B": (("U", (2, 1, 0)), ("L", (0, 3, 6)), ("D", (6, 7, 8)), ("R", (8, 5, 2))),
}

# Every edge/corner slot and the facelets that make it up. U/D-layer slots
# list the U/D facelet first; that ordering is relied on by the solver.
EDGE_SLOTS = {
    "UF": (("U", 7), ("F", 1)), "UR": (("U", 5), ("R", 1)),
    "UB": (("U", 1), ("B", 1)), "UL": (("U", 3), ("L", 1)),
    "DF": (("D", 1), ("F", 7)), "DR": (("D", 5), ("R", 7)),
    "DB": (("D", 7), ("B", 7)), "DL": (("D", 3), ("L", 7)),
    "FR": (("F", 5), ("R", 3)), "FL": (("F", 3), ("L", 5)),
    "BR": (("B", 3), ("R", 5)), "BL": (("B", 5), ("L", 3)),
}
CORNER_SLOTS = {
    "UFR": (("U", 8), ("F", 2), ("R", 0)), "UBR": (("U", 2), ("B", 0), ("R", 2)),
    "UBL": (("U", 0), ("B", 2), ("L", 0)), "UFL": (("U", 6), ("F", 0), ("L", 2)),
    "DFR": (("D", 2), ("F", 8), ("R", 6)), "DBR": (("D", 8), ("B", 6), ("R", 8)),
    "DBL": (("D", 6), ("B", 8), ("L", 6)), "DFL": (("D", 0), ("F", 6), ("L", 8)),
}

# (right neighbour, left neighbour) of each side face, as seen facing it.
_NEIGH = {"F": ("R", "L"), "R": ("B", "F"), "B": ("L", "R"), "L": ("F", "B")}

# A U turn carries side stickers F->L->B->R and top corners UFR->UFL->UBL->UBR.
_U_SIDE_CYCLE = ("F", "L", "B", "R")
_U_CORNER_CYCLE = ("UFR", "UFL", "UBL", "UBR")


class Cube:
    """A 3x3 cube as six 9-sticker faces, with the standard face turns."""

    def __init__(self, state=None):
        if state is None:
            self.f = {face: [face] * 9 for face in FACES}
        else:
            self.f = {face: list(state[face]) for face in FACES}

    def copy(self):
        return Cube(self.f)

    def key(self):
        return "".join("".join(self.f[face]) for face in FACES)

    def is_solved(self):
        return all(all(s == face for s in self.f[face]) for face in FACES)

    def _turn_cw(self, face):
        old = self.f[face]
        self.f[face] = [old[_FACE_CW[i]] for i in range(9)]
        strips = _STRIPS[face]
        saved = [[self.f[f][i] for i in idx] for f, idx in strips]
        for k, (f, idx) in enumerate(strips):
            src = saved[(k - 1) % 4]
            for j, i in enumerate(idx):
                self.f[f][i] = src[j]

    def apply(self, moves):
        """Apply a move sequence: a list of tokens or a space-separated string."""
        if isinstance(moves, str):
            moves = moves.split()
        for token in moves:
            face, turns = parse_move(token)
            for _ in range(turns):
                self._turn_cw(face)
        return self


def parse_move(token):
    """Split a move token into (face, clockwise quarter turns 1-3)."""
    face = token[0]
    if face not in FACES:
        raise ValueError(f"bad move token {token!r}")
    suffix = token[1:]
    turns = {"": 1, "2": 2, "'": 3}.get(suffix)
    if turns is None:
        raise ValueError(f"bad move token {token!r}")
    return face, turns


def simplify(moves):
    """Merge adjacent same-face turns (R R -> R2, R R' -> nothing, ...)."""
    stack = []
    for token in moves:
        face, turns = parse_move(token)
        if stack and stack[-1][0] == face:
            face, prev = stack.pop()
            turns = (prev + turns) % 4
            if turns == 0:
                continue
        stack.append((face, turns))
    return [face + {1: "", 2: "2", 3: "'"}[turns] for face, turns in stack]


def scramble(n=25, seed=None):
    """Random move sequence for testing (avoids consecutive same-face turns)."""
    rng = random.Random(seed)
    seq, last = [], None
    while len(seq) < n:
        face = rng.choice(FACES)
        if face == last:
            continue
        seq.append(face + rng.choice(["", "2", "'"]))
        last = face
    return seq


def validate(state):
    """Cheap sanity checks on a scanned state. Returns a list of problems."""
    issues = []
    counts = {}
    for face in FACES:
        stickers = state.get(face)
        if stickers is None or len(stickers) != 9:
            return [f"face {face} is missing or not 9 stickers"]
        for s in stickers:
            counts[s] = counts.get(s, 0) + 1
    for face in FACES:
        if counts.get(face, 0) != 9:
            issues.append(f"expected 9 '{face}' stickers, found {counts.get(face, 0)}")
    centers = sorted(state[face][4] for face in FACES)
    if centers != sorted(FACES):
        issues.append("the six centers are not six distinct faces")
    return issues


# --- solver -----------------------------------------------------------------
class _Solver:
    def __init__(self, cube):
        self.c = cube
        self.seq = []

    def do(self, moves):
        self.c.apply(moves)
        self.seq.extend(moves)

    def _fail(self, why):
        raise ValueError(f"cube state looks invalid or misdetected ({why})")

    # -- piece lookups
    def _find_edge(self, pair):
        for name, slots in EDGE_SLOTS.items():
            if {self.c.f[f][i] for f, i in slots} == pair:
                return name
        self._fail(f"edge {sorted(pair)} not found")

    def _find_corner(self, trio):
        for name, slots in CORNER_SLOTS.items():
            if {self.c.f[f][i] for f, i in slots} == trio:
                return name
        self._fail(f"corner {sorted(trio)} not found")

    def _slot_solved(self, slots_table, name):
        return all(self.c.f[f][i] == f for f, i in slots_table[name])

    # -- stage 1: bottom cross via the daisy
    def _protect_petal(self, side):
        """Rotate U so the U-slot above `side` holds no daisy petal."""
        slot = {"F": 7, "R": 5, "B": 1, "L": 3}[side]
        for _ in range(4):
            if self.c.f["U"][slot] != "D":
                return
            self.do(["U"])
        self._fail("could not clear a daisy slot")

    def _daisy(self):
        # Single face turn that lifts a middle-layer D sticker onto the U face,
        # keyed by (edge slot, face the D sticker is on).
        lift = {
            ("FR", "F"): "R", ("FR", "R"): "F'", ("FL", "F"): "L'", ("FL", "L"): "F",
            ("BR", "B"): "R'", ("BR", "R"): "B", ("BL", "B"): "L", ("BL", "L"): "B'",
        }
        for guard in range(24):
            target = None
            for other in "FRBL":
                name = self._find_edge({"D", other})
                slots = EDGE_SLOTS[name]
                d_face = next(f for f, i in slots if self.c.f[f][i] == "D")
                if d_face != "U":
                    target = (name, slots, d_face)
                    break
            if target is None:
                return
            name, slots, d_face = target
            if d_face == "D":
                # D sticker facing straight down: half-turn its side face up.
                side = next(f for f, _ in slots if f != "D")
                self._protect_petal(side)
                self.do([side, side])
            elif name in ("UF", "UR", "UB", "UL") or name in ("DF", "DR", "DB", "DL"):
                # D sticker sideways in the top or bottom layer: one turn of
                # that face pushes the edge into the middle layer.
                self._protect_petal(d_face)
                self.do([d_face])
            else:
                move = lift[(name, d_face)]
                self._protect_petal(move[0])
                self.do([move])
        self._fail("daisy did not converge")

    def _plant_cross(self):
        for _ in range(4):
            if not any(self.c.f["U"][i] == "D" for i in (1, 3, 5, 7)):
                break
            for _ in range(4):
                if self.c.f["U"][7] == "D":
                    break
                self.do(["U"])
            side = self.c.f["F"][1]
            self.do(["U"] * _U_SIDE_CYCLE.index(side))
            self.do([side, side])
        for name in ("DF", "DR", "DB", "DL"):
            if not self._slot_solved(EDGE_SLOTS, name):
                self._fail("bottom cross incomplete")

    # -- stage 2: bottom corners
    def _corners_first_layer(self):
        trigger = {
            "DFR": ["R", "U", "R'", "U'"], "DBR": ["B", "U", "B'", "U'"],
            "DBL": ["L", "U", "L'", "U'"], "DFL": ["F", "U", "F'", "U'"],
        }
        staging = {"DFR": "UFR", "DBR": "UBR", "DBL": "UBL", "DFL": "UFL"}
        for dest in ("DFR", "DBR", "DBL", "DFL"):
            trio = set(dest)  # e.g. {"D", "F", "R"}
            for guard in range(14):
                if self._slot_solved(CORNER_SLOTS, dest):
                    break
                pos = self._find_corner(trio)
                if pos in trigger:
                    # In the bottom layer but wrong/misoriented: pop it up.
                    self.do(trigger[pos])
                else:
                    k = (_U_CORNER_CYCLE.index(staging[dest])
                         - _U_CORNER_CYCLE.index(pos)) % 4
                    self.do(["U"] * k)
                    self.do(trigger[dest])
            else:
                self._fail("a bottom corner would not seat")

    # -- stage 3: middle-layer edges
    @staticmethod
    def _right_insert(side):
        rn = _NEIGH[side][0]
        return ["U", rn, "U'", rn + "'", "U'", side + "'", "U", side]

    @staticmethod
    def _left_insert(side):
        ln = _NEIGH[side][1]
        return ["U'", ln + "'", "U", ln, "U", side, "U'", side + "'"]

    def _middle_edges(self):
        eject = {"FR": "F", "BR": "R", "BL": "B", "FL": "L"}
        for pair in ({"F", "R"}, {"F", "L"}, {"B", "R"}, {"B", "L"}):
            name = "".join(sorted(pair, key="FRBL".index))
            slot = name if name in EDGE_SLOTS else name[::-1]
            for guard in range(10):
                if self._slot_solved(EDGE_SLOTS, slot):
                    break
                pos = self._find_edge(pair)
                if pos in eject:
                    self.do(self._right_insert(eject[pos]))
                    continue
                # In the U layer: align its side sticker over the matching
                # centre, then insert right or left.
                (fu, iu), (fs, iside) = EDGE_SLOTS[pos]
                side_val = self.c.f[fs][iside]
                up_val = self.c.f[fu][iu]
                k = (_U_SIDE_CYCLE.index(side_val) - _U_SIDE_CYCLE.index(fs)) % 4
                self.do(["U"] * k)
                if up_val == _NEIGH[side_val][0]:
                    self.do(self._right_insert(side_val))
                else:
                    self.do(self._left_insert(side_val))
            else:
                self._fail("a middle edge would not insert")

    # -- stages 4-6: last layer via tiny breadth-first searches
    def _f2l_intact(self, c):
        return (all(s == "D" for s in c.f["D"])
                and all(c.f[f][i] == f for f in "FRBL" for i in (3, 4, 5, 6, 7, 8)))

    def _edges_oriented(self, c):
        return all(c.f["U"][i] == "U" for i in (1, 3, 5, 7))

    def _edges_solved(self, c):
        return self._edges_oriented(c) and all(c.f[f][1] == f for f in "FRBL")

    def _corners_positioned(self, c):
        return all(
            {c.f[f][i] for f, i in CORNER_SLOTS[name]} == set(name)
            for name in ("UFR", "UBR", "UBL", "UFL")
        )

    def _search(self, ops, goal, max_depth, why):
        """BFS over short alg sequences; applies and records the first hit."""
        if goal(self.c):
            return
        seen = {self.c.key()}
        queue = deque([(self.c.copy(), [])])
        while queue:
            cube, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for alg in ops:
                nxt = cube.copy().apply(alg)
                key = nxt.key()
                if key in seen:
                    continue
                seen.add(key)
                if goal(nxt):
                    for step in path + [alg]:
                        self.do(step)
                    return
                queue.append((nxt, path + [alg]))
        self._fail(why)

    def _last_layer(self):
        u_turns = [["U"], ["U'"], ["U2"]]

        def oll(side):
            rn = _NEIGH[side][0]
            return [side, rn, "U", rn + "'", "U'", side + "'"]

        def sune(side):
            return [side, "U", side + "'", "U", side, "U2", side + "'"]

        def niklas(a, b):
            return ["U", a, "U'", b + "'", "U", a + "'", "U'", b]

        self._search(
            u_turns + [oll(s) for s in "FRBL"],
            lambda c: self._f2l_intact(c) and self._edges_oriented(c),
            5, "top edges cannot be oriented (flipped edge?)",
        )
        self._search(
            u_turns + [sune(s) for s in "FRBL"],
            lambda c: self._f2l_intact(c) and self._edges_solved(c),
            5, "top edges cannot be permuted (swapped pieces?)",
        )
        self._search(
            u_turns + [niklas(a, b) for a, b in (("R", "L"), ("L", "R"), ("B", "F"), ("F", "B"))],
            lambda c: (self._f2l_intact(c) and self._edges_solved(c)
                       and self._corners_positioned(c)),
            5, "top corners cannot be positioned (swapped pieces?)",
        )

        # Orient each top corner in place at UFR with R' D' R D pairs; the
        # lower layers look scrambled mid-way but restore once every corner
        # is done (guaranteed only for a physically valid cube).
        for _ in range(4):
            for guard in range(7):
                if self.c.f["U"][8] == "U":
                    break
                self.do(["R'", "D'", "R", "D"])
            else:
                self._fail("a top corner cannot be oriented (twisted corner?)")
            self.do(["U"])
        for _ in range(4):
            if self.c.is_solved():
                return
            self.do(["U"])
        if not self.c.is_solved():
            self._fail("final alignment failed")

    def run(self):
        self._daisy()
        self._plant_cross()
        self._corners_first_layer()
        self._middle_edges()
        self._last_layer()
        return simplify(self.seq)


def solve(state):
    """Solve a cube state ({face: [9 letters]}) into a move list.

    Returns [] when already solved. Raises ValueError when the state fails
    basic validation or a solving stage cannot complete (which is what a
    misdetected scan looks like).
    """
    issues = validate(state)
    if issues:
        raise ValueError("; ".join(issues))
    cube = Cube(state)
    if cube.is_solved():
        return []
    return _Solver(cube).run()


# --- self-test (pure logic, no camera) ---------------------------------------
def self_test():
    all_ok = True

    def check(desc, got, want):
        nonlocal all_ok
        ok = got == want
        all_ok = all_ok and ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {desc:<40} expected {want}, got {got}")

    # Move-engine algebra. These are properties of the real cube group, so
    # they catch mis-derived permutations, not just internal inconsistency.
    for face in FACES:
        check(f"{face}*4 is identity", Cube().apply([face] * 4).is_solved(), True)

    c = Cube()
    order = 0
    while True:
        c.apply("R U")
        order += 1
        if c.is_solved() or order > 200:
            break
    check("(R U) has order 105", order, 105)

    c = Cube()
    for _ in range(6):
        c.apply("R U R' U'")
    check("sexy move has order 6", c.is_solved(), True)

    check("simplify merges R R", simplify(["R", "R"]), ["R2"])
    check("simplify cancels R R'", simplify(["R", "R'"]), [])
    check("simplify cascades", simplify(["R", "L", "L'", "R'"]), [])

    check("solved cube solves to []", solve(Cube().f), [])

    # The ground truth: random scrambles must come back solved.
    lengths = []
    solved_all = True
    for i in range(30):
        cube = Cube().apply(scramble(25, seed=i))
        moves = solve(cube.f)
        cube.apply(moves)
        solved_all = solved_all and cube.is_solved()
        lengths.append(len(moves))
    check("30 random scrambles all solved", solved_all, True)
    check("solutions reasonably short", max(lengths) < 300, True)
    print(f"       (solution lengths: min {min(lengths)}, "
          f"avg {sum(lengths) // len(lengths)}, max {max(lengths)})")

    # Physically impossible states must be rejected, not "solved".
    twisted = Cube()
    a, b, c2 = CORNER_SLOTS["UFR"]
    va = twisted.f[a[0]][a[1]]
    twisted.f[a[0]][a[1]] = twisted.f[b[0]][b[1]]
    twisted.f[b[0]][b[1]] = twisted.f[c2[0]][c2[1]]
    twisted.f[c2[0]][c2[1]] = va
    twisted.apply(scramble(10, seed=99))
    try:
        solve(twisted.f)
        check("twisted corner rejected", "no error", "ValueError")
    except ValueError:
        check("twisted corner rejected", "ValueError", "ValueError")

    bad = Cube().f
    bad["U"] = ["U"] * 8 + ["R"]
    try:
        solve(bad)
        check("bad sticker counts rejected", "no error", "ValueError")
    except ValueError:
        check("bad sticker counts rejected", "ValueError", "ValueError")

    print("\nSelf-test", "passed." if all_ok else "FAILED.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
