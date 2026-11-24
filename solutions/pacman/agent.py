"""A scripted Pac-Man for the rust `pacman` env that plays from the ASCII view.

The agent is handed one thing per step: the text frame `mapox.ascii` draws for
Pac-Man's own observation (the egocentric crop plus the two UI rows), and it
answers with an action name. It never sees the full map, the env state, the
seed, the reward or the termination flag.

What it does know is the game itself, the way a human player knows it: the
maze, how Pac-Man and the ghosts move, and the env config (timers and reward
sizes). From that and the frames it keeps

  * its own position, by matching the wall pattern in view against the maze,
  * the pellets left, by remembering where it has walked,
  * the scatter/chase clock, by counting steps since the round began,
  * a belief over every ghost: a weighted set of states that the ghost rules
    carry forward each step and each frame prunes. A ghost that has been seen
    once stays pinned down exactly until a power pellet makes it wander.

It then searches its own moves a couple of dozen steps ahead against those
beliefs and takes the first move of the best line.

How it plays follows from what being caught costs. Under the default rewards
that is nothing: a catch pays 0 and swaps the maze for a full one, so the most
reward comes from eating a pellet on every step and never trekking after
leftovers. The agent does exactly that, some 70 pellets and then into a ghost,
for about 1030 reward in 1024 steps. Give a catch a price in the env config
and it stays with each maze longer; from a `death_reward` of about -15 it
plays them out, some 2.5 mazes and 780 reward in 1024 steps. `--play-to-clear`
gives up reward to be safe: 99.5% of mazes cleared, at 0.65 reward per step.
`--play-for-return` plays each episode for its return discounted at 0.996,
the way an RL value function counts it: a catch or a clear ends it and
nothing after counts. Under the training rewards (pellet 0.03, ghost 0.3,
catch -0.3, clear 3.0, random start) its first episode is worth 5.47 against
5.34 for the default style and 5.20 for `--play-to-clear`; the
web-demo-v6 policy makes about 6.03, mostly by eating ghosts sooner.

Run it:

    uv run python solutions/pacman/agent.py                # score 16 budgets
    uv run python solutions/pacman/agent.py --play-to-clear
    uv run python solutions/pacman/agent.py --play-for-return
    uv run python solutions/pacman/agent.py --show         # watch one, as text
    uv run python solutions/pacman/agent.py --config '{"death_reward": -0.5}'
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# The game, as the agent knows it. This mirrors crates/mapox-core/src/envs/
# pacman.rs; it is the rulebook, not a view into a running env.
# --------------------------------------------------------------------------

# `#` wall, `.` pellet, `x` power pellet, `p` Pac-Man's spawn, `1`-`4` the
# ghost spawns. Row 14 is the tunnel: x wraps around.
MAZE = (
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#x####.#####.##.#####.####x#",
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "######.##### ## #####.######",
    "######.##   1234   ##.######",
    "######.## ######## ##.######",
    "######.## ######## ##.######",
    "      .   ########   .      ",
    "######.## ######## ##.######",
    "######.## ######## ##.######",
    "######.##          ##.######",
    "######.## ######## ##.######",
    "######.## ######## ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#.####.#####.##.#####.####.#",
    "#x..##.......p .......##..x#",
    "###.##.##.########.##.##.###",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################",
)

W = len(MAZE[0])
H = len(MAZE)
NCELL = W * H

# Cells are `row * W + x`, rows counted from the top like the frames. Moves
# and headings are in the env's order: up, right, down, left.
DX = (0, 1, 0, -1)
DR = (-1, 0, 1, 0)
UP, RIGHT = 0, 1
ACTIONS = ("move/up", "move/right", "move/down", "move/left")

PINKY, BLINKY, INKY, CLYDE = range(4)
# the frame glyph of each ghost, by ghost index; ghosts are drawn in index
# order, so a later one hides an earlier one on the same cell
GHOST_GLYPHS = "PBIC"
GHOST_MARKERS = "3214"
# a ghost weighs its exits in this order, so ties go to the earlier one
GHOST_EXIT_ORDER = (2, 0, 1, 3)
# scatter targets as (x, row): points off the maze beyond each home corner
CORNERS = ((3, -3), (W - 4, -3), (W - 1, H), (0, H))
PINKY_LEAD = 4
INKY_LEAD = 2
CLYDE_RADIUS_SQ = 8 * 8

NORMAL, FRIGHTENED, EYES = 0, 1, 2

CX = [c % W for c in range(NCELL)]
CR = [c // W for c in range(NCELL)]
WALL = [ch == "#" for row in MAZE for ch in row]
OPEN = [c for c in range(NCELL) if not WALL[c]]


def _neighbor(c: int, d: int) -> int:
    r = CR[c] + DR[d]
    if not 0 <= r < H:
        return -1
    n = r * W + (CX[c] + DX[d]) % W
    return -1 if WALL[n] else n


# NB[c][d]: the cell one step from c in direction d, or -1 for a wall
NB = [
    tuple(_neighbor(c, d) for d in range(4)) if not WALL[c] else (-1,) * 4
    for c in range(NCELL)
]


def _ghost_exits(c: int, d: int) -> tuple:
    """The exits a ghost at c heading d may take: never straight back."""
    return tuple(
        (e, NB[c][e], CX[c] + DX[e], CR[c] + DR[e])
        for e in GHOST_EXIT_ORDER
        if e != d ^ 2 and NB[c][e] >= 0
    )


GEX = [
    tuple(_ghost_exits(c, d) for d in range(4)) if not WALL[c] else ((),) * 4
    for c in range(NCELL)
]


def _touching(c: int) -> frozenset:
    """The cells a ghost catches (or is eaten) from: c and its four sides."""
    x, r = CX[c], CR[c]
    cells = {c, r * W + (x + 1) % W, r * W + (x - 1) % W}
    if r > 0:
        cells.add(c - W)
    if r < H - 1:
        cells.add(c + W)
    return frozenset(cells)


TOUCH = [_touching(c) for c in range(NCELL)]

FAR = 999


def _distances(src: int) -> list:
    dist = [FAR] * NCELL
    dist[src] = 0
    frontier = [src]
    while frontier:
        reached = []
        for c in frontier:
            for n in NB[c]:
                if n >= 0 and dist[n] == FAR:
                    dist[n] = dist[c] + 1
                    reached.append(n)
        frontier = reached
    return dist


# DIST[a][b]: steps from a to b
DIST = [_distances(c) if not WALL[c] else None for c in range(NCELL)]

PELLETS = [
    r * W + x for r, row in enumerate(MAZE) for x, ch in enumerate(row) if ch in ".x"
]
NPELLET = len(PELLETS)
PIDX = [-1] * NCELL
PBIT = [0] * NCELL
for _i, _c in enumerate(PELLETS):
    PIDX[_c] = _i
    PBIT[_c] = 1 << _i
# the pellet cells beside each cell: pellets that touch are one fragment
PELLET_NB = [tuple(n for n in NB[c] if n >= 0 and PBIT[n]) for c in range(NCELL)]
POWER_CELLS = [
    r * W + x for r, row in enumerate(MAZE) for x, ch in enumerate(row) if ch == "x"
]
POWER_MASK = sum(PBIT[c] for c in POWER_CELLS)
ALL_PELLETS = (1 << NPELLET) - 1


def _find(marker: str) -> int:
    for r, row in enumerate(MAZE):
        x = row.find(marker)
        if x >= 0:
            return r * W + x
    raise ValueError(f"the maze has no {marker!r}")


GHOST_SPAWN = tuple(_find(m) for m in GHOST_MARKERS)
GHOST_SPAWN_XR = tuple((CX[c], CR[c]) for c in GHOST_SPAWN)


@dataclass(frozen=True)
class Rules:
    """The env config the agent plays under: public, like the maze."""

    view_width: int = 15
    view_height: int = 15
    min_start_timeout: int = 0
    max_start_timeout: int = 49
    frightened_time: int = 35
    max_mode_changes: int = 6
    scatter_mode_length: int = 70
    chase_mode_length: int = 140
    pellet_reward: float = 1.0
    ghost_reward: float = 1.0
    death_reward: float = 0.0
    clear_reward: float = 0.0

    @classmethod
    def from_config(cls, config) -> Rules:
        """From a `RustPacmanConfig`, keeping the keys the agent uses."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.model_dump().items() if k in names})


def mode_schedule(rules: Rules) -> tuple[list, list]:
    """`scatter[n]` and `flip[n]` for the n-th step of a round, n from 1.

    The last entry holds for every later step: once the mode changes run out
    the ghosts keep their mode and never turn around on the clock again.
    """
    scatter, flip = [False], [False]
    mode, left, changes = False, 0, 0
    while changes < rules.max_mode_changes:
        left -= 1
        flipped = left <= 0
        if flipped:
            mode = not mode
            changes += 1
            left = rules.scatter_mode_length if mode else rules.chase_mode_length
        scatter.append(mode)
        flip.append(flipped)
    scatter.append(mode)
    flip.append(False)
    return scatter, flip


def _pick_exit(exits: tuple, tx: int, tr: int) -> int:
    """The exit that leaves a ghost nearest its target, first one on a tie."""
    best, best_d = exits[0][0], 1 << 30
    for e, _, x, r in exits:
        d = (x - tx) * (x - tx) + (r - tr) * (r - tr)
        if d < best_d:
            best, best_d = e, d
    return best


def make_advance(rules: Rules, split_floor: float = 0.0):
    """The ghosts' half of one env step, over weighted ghost states.

    A particle is `(ghost, cell, heading, timeout, state, half, weight)`:
    `timeout` is the steps left before the ghost first moves (-1 once free),
    `state` one of NORMAL / FRIGHTENED / EYES, and `half` the env's flag that
    makes a frightened ghost rest every other step.

    The returned function carries every particle through step `n1` of the
    round, given where Pac-Man stood and faced before his move (`pb`, `hd`)
    and where he stands after it (`pa`). A frightened ghost at a junction
    splits into one particle per exit. It returns the surviving particles,
    and per ghost the weight that caught Pac-Man and the weight he ate (each
    None when there is none).

    Inky steers by Blinky, so his move depends on where Blinky is. `blinky`
    says where that may be, as `(cell, probability)` pairs.

    With `exact` the result is the true distribution, which is what tracking
    needs: a frightened ghost always splits, and Inky splits over `blinky`.
    Without it the function serves the search, which cannot afford to drown
    in ghost states: a particle lighter than `split_floor` takes one exit
    instead of splitting, and Inky follows the heaviest Blinky particle he
    is handed alongside.
    """
    scatter_tab, flip_tab = mode_schedule(rules)
    last = len(scatter_tab) - 1

    def advance(parts, n1, pb, hd, pa, ate_power, fright, blinky=None, exact=False):
        i = n1 if n1 < last else last
        scatter = scatter_tab[i]
        reverse = flip_tab[i] or ate_power
        touch = TOUCH[pa]
        # the power ends before the step is drawn: every ghost is itself again
        calm = fright == 0
        out = []
        dead = eaten = None

        for gi, c, d, to, st, half, w in parts:
            if ate_power and st == NORMAL:
                st = FRIGHTENED
            if to >= 0:
                to -= 1

            if st == FRIGHTENED and half:
                if c in touch:
                    if eaten is None:
                        eaten = [0.0] * 4
                    eaten[gi] += w
                    out.append((gi, c, d, to, EYES, 0, w))
                else:
                    out.append((gi, c, d, to, NORMAL if calm else FRIGHTENED, 0, w))
                continue

            if reverse and st != EYES:
                heads = ((d ^ 2, w),)
            else:
                exits = GEX[c][d]
                if len(exits) == 1:
                    heads = ((exits[0][0], w),)
                elif not exits:
                    heads = ((d ^ 2, w),)
                elif st == FRIGHTENED:
                    if not exact and w < split_floor:
                        heads = ((exits[(c + n1) % len(exits)][0], w),)
                    else:
                        share = w / len(exits)
                        heads = [(e[0], share) for e in exits]
                else:
                    px, pr = CX[pb], CR[pb]
                    if st == EYES:
                        tx, tr = GHOST_SPAWN_XR[gi]
                    elif scatter:
                        tx, tr = CORNERS[gi]
                    elif gi == BLINKY:
                        tx, tr = px, pr
                    elif gi == PINKY:
                        tx, tr = px + PINKY_LEAD * DX[hd], pr + PINKY_LEAD * DR[hd]
                    elif gi == CLYDE:
                        ox, orow = px - CX[c], pr - CR[c]
                        if ox * ox + orow * orow > CLYDE_RADIUS_SQ:
                            tx, tr = px, pr
                        else:
                            tx, tr = CORNERS[CLYDE]
                    else:
                        # Inky mirrors Blinky through a point ahead of Pac-Man
                        vx = 2 * (px + INKY_LEAD * DX[hd])
                        vr = 2 * (pr + INKY_LEAD * DR[hd])
                        bc = None
                        if not exact:
                            bw = -1.0
                            for q in parts:
                                if q[0] == BLINKY and q[6] > bw:
                                    bc, bw = q[1], q[6]
                            if bc is None:
                                bc = pb
                        elif len(blinky) == 1:
                            bc = blinky[0][0]
                        if bc is not None:
                            tx, tr = vx - CX[bc], vr - CR[bc]
                        else:
                            split = {}
                            for bc, bw in blinky:
                                e = _pick_exit(exits, vx - CX[bc], vr - CR[bc])
                                split[e] = split.get(e, 0.0) + bw
                            heads = [(e, w * p) for e, p in split.items()]
                            tx = None
                    if tx is not None:
                        heads = ((_pick_exit(exits, tx, tr), w),)

            for nd, ww in heads:
                nc = c
                if to < 0:
                    m = NB[c][nd]
                    if m >= 0:
                        nc = m
                s, h = st, 1
                if s == EYES:
                    if nc == GHOST_SPAWN[gi]:
                        s = NORMAL
                elif nc in touch:
                    if s == FRIGHTENED:
                        if eaten is None:
                            eaten = [0.0] * 4
                        eaten[gi] += ww
                        s, h = EYES, 0
                    else:
                        if dead is None:
                            dead = [0.0] * 4
                        dead[gi] += ww
                        continue
                if calm:
                    if s == FRIGHTENED:
                        s = NORMAL
                    h = 0
                out.append((gi, nc, nd, to, s, h, ww))

        return out, dead, eaten

    return advance


# --------------------------------------------------------------------------
# Reading a frame
# --------------------------------------------------------------------------

PACMAN_GLYPHS = {"^": 0, ">": 1, "v": 2, "<": 3}
FRAME_CHARS = frozenset("-#.o@^>v<PBICSE0123456789")
# every glyph but a wall reads as floor when matching the wall pattern
_WALLS_ONLY = {ord(ch): "." for ch in FRAME_CHARS if ch != "#"}


class FrameError(ValueError):
    """The text handed to the agent is not a Pac-Man frame."""


@dataclass
class Frame:
    score: int
    power: int
    rows: list  # the view, top row first


def parse_frame(text, rules: Rules) -> Frame:
    """Pull the view and the two UI numbers out of an ASCII frame.

    Accepts the bare grid or a full `ascii_frame` with its title and action
    lines, as a string or a list of rows.
    """
    lines = text.splitlines() if isinstance(text, str) else list(text)
    rows = [
        ln for ln in lines if len(ln) == rules.view_width and FRAME_CHARS.issuperset(ln)
    ]
    if len(rows) < rules.view_height + 1:
        raise FrameError(
            f"expected a {rules.view_width}-wide Pac-Man frame, got {lines!r}"
        )
    ui, view = rows[: -rules.view_height], rows[-rules.view_height :]

    def number(row: str) -> int:
        digits = "".join(ch for ch in row if ch.isdigit())
        return int(digits) if digits else 0

    return Frame(
        score=number(ui[0]),
        power=number(ui[1]) if len(ui) > 1 else 0,
        rows=view,
    )


def view_origin(rules: Rules) -> tuple[int, int]:
    """Pac-Man's (column, row) inside the view, row counted from the top."""
    return rules.view_width // 2, rules.view_height - 1 - rules.view_height // 2


def wall_patterns(rules: Rules) -> dict:
    """Wall pattern of the view -> the cells Pac-Man could be standing on."""
    col, row = view_origin(rules)
    table: dict = {}
    for c in OPEN:
        chars = []
        for vr in range(rules.view_height):
            r = CR[c] - row + vr
            for vc in range(rules.view_width):
                x = (CX[c] - col + vc) % W
                chars.append("#" if not 0 <= r < H or WALL[r * W + x] else ".")
        table.setdefault("".join(chars), []).append(c)
    return table


# --------------------------------------------------------------------------
# Routes: the order to eat the pellets in
# --------------------------------------------------------------------------


def tour_length(pos: int, tour: list, dist: list) -> int:
    """Steps to walk a tour from pos, eating each pellet in turn."""
    steps, at = 0, pos
    for c in tour:
        steps += dist[at][c]
        at = c
    return steps


def nearest_tour(pos: int, cells, dist: list) -> list:
    """A first tour: always the nearest pellet next, loose ends before
    pellets with more pellets beyond them."""
    left = set(cells)
    tour = []
    at = pos
    while left:
        row = dist[at]
        at = min(left, key=lambda c: (row[c], sum(n in left for n in PELLET_NB[c])))
        tour.append(at)
        left.discard(at)
    return tour


def improve_tour(pos: int, tour: list, dist: list, moves: int) -> bool:
    """Shorten a tour in place, one local move at a time, at most `moves`.

    The tour is only ever cut where it already jumps between pellets that do
    not touch, or at a junction, so runs of pellets stay whole. A move either
    reverses a stretch between two cuts or lifts one run out and drops it in
    at another cut, either way round. Returns True once no move helps.
    """
    n = len(tour)
    for _ in range(moves):
        cuts = [-1] + [
            i
            for i in range(n - 1)
            if dist[tour[i]][tour[i + 1]] > 1 or len(PELLET_NB[tour[i]]) > 2
        ]
        ends = cuts + [n - 1]
        moved = False

        # reverse tour[i + 1 .. j]
        for at, i in enumerate(cuts):
            a = tour[i] if i >= 0 else pos
            b = tour[i + 1]
            from_a = dist[a]
            ab = from_a[b]
            for j in ends[at + 1 :]:
                c = tour[j]
                if j < n - 1:
                    e = tour[j + 1]
                    gain = ab + dist[c][e] - from_a[c] - dist[b][e]
                else:
                    gain = ab - from_a[c]
                if gain > 0:
                    tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
                    moved = True
                    break
            if moved:
                break
        if moved:
            continue

        # move the run tour[a .. b] to just after cut i
        for k in range(len(ends) - 1):
            a, b = ends[k] + 1, ends[k + 1]
            before = tour[a - 1] if a > 0 else pos
            first, last = tour[a], tour[b]
            if b < n - 1:
                after = tour[b + 1]
                gain = dist[before][first] + dist[last][after] - dist[before][after]
            else:
                gain = dist[before][first]
            if gain <= 0:
                continue
            for i in ends:
                if a - 1 <= i <= b:
                    continue
                x = tour[i] if i >= 0 else pos
                if i < n - 1:
                    y = tour[i + 1]
                    forward = dist[x][first] + dist[last][y] - dist[x][y]
                    backward = dist[x][last] + dist[first][y] - dist[x][y]
                else:
                    forward, backward = dist[x][first], dist[x][last]
                if min(forward, backward) < gain:
                    run = tour[a : b + 1]
                    if backward < forward:
                        run.reverse()
                    if i < a:
                        tour[i + 1 : b + 1] = run + tour[i + 1 : a]
                    else:
                        tour[a : i + 1] = tour[b + 1 : i + 1] + run
                    moved = True
                    break
            if moved:
                break
        if not moved:
            return True
    return False


def polish_tour(pos: int, tour: list, dist: list, rounds: int) -> list:
    """A better tour than local moves alone find: shake it and tidy again.

    Each round reverses two random stretches of the best tour so far, tidies
    the result, and keeps it if it is no longer. The shaking is seeded by the
    start, so the same start always gets the same tour.
    """
    rng = random.Random(pos * 1009 + len(tour))
    best = list(tour)
    improve_tour(pos, best, dist, 400)
    best_length = tour_length(pos, best, dist)
    for _ in range(rounds):
        n = len(best)
        cuts = [
            i
            for i in range(n - 1)
            if dist[best[i]][best[i + 1]] > 1 or len(PELLET_NB[best[i]]) > 2
        ]
        if len(cuts) < 2:
            break
        trial = list(best)
        for _ in range(2):
            i, j = sorted(rng.sample(cuts, 2))
            trial[i + 1 : j + 1] = trial[i + 1 : j + 1][::-1]
        improve_tour(pos, trial, dist, 400)
        length = tour_length(pos, trial, dist)
        if length <= best_length:
            best, best_length = trial, length
    return best


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tuning:
    """Search knobs. Values are in pellets: what a pellet pays is worth 1."""

    # plies searched exhaustively, and the node count that trims it
    depth: int = 22
    min_depth: int = 12
    node_limit: int = 6000

    # What being caught costs. None leaves it at the env's own price: under
    # the default rewards that is nothing, since a catch only swaps the maze
    # for a full one, and the agent does not mind dying. A cost per pellet
    # still on the board can be added to it.
    death_cost: float | None = None
    death_cost_per_pellet: float = 0.0
    # play carefully rather than for reward per step (see `PacmanAgent`)
    careful: bool = False

    # a frightened ghost still loose at a leaf counts this much, fading with
    # how far it is against the time left
    chase_value: float = 0.8
    chase_pace: float = 1.6
    # turning back or standing still mid-path to time a power pellet: this
    # many times, only this near one or just after eating one (or forced)
    dither_budget: int = 2
    dither_radius: int = 2
    # turning back right now costs this much, so that two plans that are as
    # good as each other do not have Pac-Man pacing between them
    turn_cost: float = 0.3
    # ghost states the search carries per ghost, and the weight below which
    # a frightened one stops forking
    max_particles: int = 6
    split_floor: float = 0.6

    # -- playing for reward per step --
    # What a step is worth: the reward the agent expects to make per step.
    # Every step is charged this, so a plan is judged by how far ahead of
    # that pace it gets, and a maze is dropped once it falls behind. None
    # works it out from the rewards (see `PacmanAgent._best_pace`), taking a
    # clear to last `clear_slowdown` times its tour with ghosts in the way,
    # and to pick up `clear_ghosts` of them.
    pace: float | None = None
    clear_slowdown: float = 1.25
    clear_ghosts: float = 4.0
    # steps it takes to get caught on purpose when no ghost is known near
    die_time: float = 4.0
    # the tour: shakes when a round starts, local moves per step after, and
    # how far its start may drift before a fresh one is tried
    tour_polish: int = 40
    tour_moves: int = 4
    tour_drift: int = 6

    # -- playing for discounted return --
    # Set, the agent plays each episode for the reward it makes discounted
    # by this much a step, the way an RL return counts it: a catch or a
    # clear ends the episode and nothing after it counts. There is no pace;
    # the discount is what makes a step cost. None plays for reward per
    # step (or carefully, with `careful`).
    discount: float | None = None
    # the tour still to walk is taken to last this many times its length,
    # which puts its pellets and the clear further off
    discount_slowdown: float = 1.25
    # each power pellet left is worth this many ghosts (fading as the last
    # plain pellets go), so one is not spent unless ghosts are eaten for it
    # or it saves Pac-Man
    discount_power_ghosts: float = 1.0
    # an unlikely catch counts as this many times likelier than it is
    discount_risk_gain: float = 1.0
    # leave the power pellets off the tour while they are worth holding, as
    # careful play does: a tour through one leaves Pac-Man pacing beside it
    discount_skip_power: bool = True

    # -- playing carefully --
    # later rewards count a little less, so the search eats sooner
    gamma: float = 0.985
    # an unlikely death counts as this many times likelier than it is: the
    # ghost that kills is nearly always one that was only probably elsewhere
    risk_gain: float = 3.0
    # leaf: every pellet left counts this much, fading with its distance.
    # It has to stay below what eating the pellet pays, or the search would
    # rather stand next to a pellet than eat it.
    field_weight: float = 0.6
    field_decay: float = 0.94
    # every separate fragment of pellets left behind costs a trip back for
    # it: this much per fragment. A fragment counts as its own once a walk
    # of `fragment_scan` cells through it finds no way to the rest.
    fragment_cost: float = 2.0
    fragment_scan: int = 12
    # An uneaten power pellet is worth this at a leaf, or `power_ghosts`
    # ghosts if that is more, so one is only spent on ghosts or to survive,
    # never just because it is on the way: eaten with no ghost near it is
    # wasted, and it sends every ghost off on a random walk nobody can
    # track. The worth fades to nothing as the last `power_fade` plain
    # pellets go.
    power_value: float = 5.0
    power_ghosts: float = 3.0
    power_fade: int = 40
    # the best lines are checked this many steps further for a way to live
    verify_len: int = 32
    verify_budget: int = 400
    verify_checks: int = 24
    verify_eps: float = 0.02
    # ghost states lighter than this are left to the search's own odds
    verify_floor: float = 0.02


# what `--play-to-clear` sets: careful play, where a catch costs the rest of
# the maze
PLAY_TO_CLEAR = {"careful": True, "death_cost": 60.0, "death_cost_per_pellet": 1.0}
# what `--play-for-return` sets: each episode played for its return
# discounted at the training discount, with frightened ghosts chased harder
PLAY_FOR_RETURN = {"discount": 0.996, "chase_value": 1.2}


class PacmanAgent:
    """Plays Pac-Man from ASCII frames: `act(frame) -> action name`.

    It plays one of two ways.

    For reward per step, as a rule. It eats along a tour of the pellets for
    as long as that keeps ahead of its pace, and weighs a catch at what the
    env charges for one. Under the default rewards that is nothing: a catch
    only swaps the maze for a full one, so the agent lets itself be caught
    rather than trek after leftovers. The more a catch costs, the longer it
    stays with a maze, until it plays them out to the last pellet.

    Carefully, with `Tuning.careful`. It plays to clear every maze: it
    weighs each line against the odds of dying on it, checks the best ones
    far ahead for a trap, and holds its power pellets until there is a
    ghost to use them on. That clears more mazes and makes less reward.
    """

    def __init__(self, rules: Rules | None = None, tuning: Tuning | None = None):
        rules = rules or Rules()
        # The agent reckons in pellets: every reward is taken as a multiple
        # of a pellet's, so the knobs mean the same whatever the env pays.
        unit = rules.pellet_reward if rules.pellet_reward > 0 else 1.0
        self.rules = rules = replace(
            rules,
            pellet_reward=rules.pellet_reward / unit,
            ghost_reward=rules.ghost_reward / unit,
            death_reward=rules.death_reward / unit,
            clear_reward=rules.clear_reward / unit,
        )
        self.tuning = tuning = tuning or Tuning()
        self._advance = make_advance(rules, tuning.split_floor)
        self._patterns = wall_patterns(rules)
        self._origin = view_origin(rules)

        self._death_cost = (
            -rules.death_reward if tuning.death_cost is None else tuning.death_cost
        )
        self._careful = tuning.careful
        self._discount = None if tuning.careful else tuning.discount
        # the power pellets within dithering reach of each cell, as bits
        self._near_power = [
            sum(PBIT[q] for q in POWER_CELLS if DIST[c][q] <= tuning.dither_radius)
            if not WALL[c]
            else 0
            for c in range(NCELL)
        ]
        if self._careful:
            self._gamma = [tuning.gamma**k for k in range(tuning.depth + 2)]
            # pellet field: FIELD[c] @ pellets-left = sum of decay ** distance
            field = np.zeros((NCELL, NPELLET))
            for c in OPEN:
                dist = DIST[c]
                field[c] = [tuning.field_decay ** dist[q] for q in PELLETS]
            self._field = field
            self._field_rows = [row.tolist() for row in field]
        elif self._discount is not None:
            self._gamma = [self._discount**k for k in range(tuning.depth + 2)]
            self._tour_cache = {}
            self._pace = 0.0
        else:
            self._gamma = [1.0] * (tuning.depth + 2)
            self._tour_cache: dict = {}
            self._pace = tuning.pace if tuning.pace is not None else self._best_pace()

        timeouts = (
            [rules.min_start_timeout]
            if rules.min_start_timeout == rules.max_start_timeout
            else range(rules.min_start_timeout, rules.max_start_timeout)
        )
        self._fresh_beliefs = [
            {(GHOST_SPAWN[gi], UP, t, NORMAL, 0): 1.0 / len(timeouts) for t in timeouts}
            for gi in range(4)
        ]
        self.stats = dict.fromkeys(
            ("rounds", "reseeds", "nodes", "steps", "checks", "check_nodes"), 0
        )
        self.reset()

    def reset(self) -> None:
        """Forget everything: the next frame is the first of a new game."""
        self.pos = None
        self.heading = RIGHT
        self.pellets = ALL_PELLETS
        self.score = 0
        self.round_step = 0
        self.fright = 0
        self.beliefs = [dict(b) for b in self._fresh_beliefs]
        self._last_action = None
        self._depth = self.tuning.depth
        self._tour = None

    def _best_pace(self) -> float:
        """The reward per step to hold a round to, from the rewards in play.

        It is the best a round can do on the tour from the classic start:
        eat its first so many pellets and be caught, or eat them all, which
        the ghosts make slower than the tour alone would be. Costlier
        catches make for longer rounds and a slower pace.
        """
        T, R = self.tuning, self.rules
        start = _find("p")
        tour = polish_tour(
            start, nearest_tour(start, PELLETS, DIST), DIST, T.tour_polish
        )
        self._tour_cache[start, ALL_PELLETS] = tour
        best, steps, at = 0.0, 0, start
        for j, c in enumerate(tour, 1):
            steps += DIST[at][c]
            at = c
            eaten = j * R.pellet_reward
            if j == len(tour):
                eaten += R.clear_reward + T.clear_ghosts * R.ghost_reward
                rate = eaten / (steps * T.clear_slowdown)
            else:
                rate = (eaten - self._death_cost) / (steps + T.die_time)
            if rate > best:
                best = rate
        return best

    # ---- keeping track --------------------------------------------------

    def act(self, frame) -> str:
        """One step: read the frame, update the beliefs, pick a move."""
        obs = parse_frame(frame, self.rules)
        self._track(obs)
        action = self._plan()
        self._last_action = action
        self.stats["steps"] += 1
        return ACTIONS[action]

    def _locate(self, obs: Frame, guess) -> tuple[int, int, dict]:
        """Where the frame says Pac-Man is, his heading, and what is in view."""
        col, row = self._origin
        glyph = obs.rows[row][col]
        if glyph not in PACMAN_GLYPHS:
            raise FrameError(f"no Pac-Man at the centre of the view: {obs.rows!r}")
        cells = self._patterns.get("".join(obs.rows).translate(_WALLS_ONLY))
        if not cells:
            raise FrameError(
                f"the walls in view match nowhere in the maze: {obs.rows!r}"
            )
        pos = guess if guess in cells else cells[0]

        seen = {}
        x0 = CX[pos] - col
        for vr, line in enumerate(obs.rows):
            r = CR[pos] - row + vr
            if 0 <= r < H:
                base = r * W
                for vc, ch in enumerate(line):
                    seen[base + (x0 + vc) % W] = ch
        return pos, PACMAN_GLYPHS[glyph], seen

    def _move(self, action: int) -> tuple[int, int]:
        """Where a move takes Pac-Man: into a wall, he keeps his heading."""
        d = action if NB[self.pos][action] >= 0 else self.heading
        n = NB[self.pos][d]
        return (n if n >= 0 else self.pos), d

    def _pellets_match(self, pellets: int, seen: dict) -> bool:
        for c, ch in seen.items():
            bit = PBIT[c]
            if bit and ch in ".o@" and (ch != ".") != bool(pellets & bit):
                return False
        return True

    def _track(self, obs: Frame) -> None:
        if self.pos is None:
            pos, heading, seen = self._locate(obs, None)
            self._new_round(pos, heading, obs, seen)
            return

        prev_pos, prev_heading = self.pos, self.heading
        want_pos, want_heading = self._move(self._last_action)
        pellets, score, ate_power = self.pellets, self.score, False
        if want_pos != prev_pos and pellets & PBIT[want_pos]:
            pellets ^= PBIT[want_pos]
            score += 1
            ate_power = bool(PBIT[want_pos] & POWER_MASK)

        pos, heading, seen = self._locate(obs, want_pos)
        if (
            pellets == 0
            or pos != want_pos
            or heading != want_heading
            or obs.score != score
            or not self._pellets_match(pellets, seen)
        ):
            # caught, or the maze was cleared: either way a new round began
            self._new_round(pos, heading, obs, seen)
            return

        self.pos, self.heading = pos, heading
        self.pellets, self.score = pellets, score
        self.round_step += 1
        fright = self.fright - 1 if self.fright > 0 else 0
        if ate_power:
            fright = self.rules.frightened_time
        self.fright = fright

        parts = [(gi, *s, w) for gi in range(4) for s, w in self.beliefs[gi].items()]
        blinky: dict = {}
        for s, w in self.beliefs[BLINKY].items():
            blinky[s[0]] = blinky.get(s[0], 0.0) + w
        out, _, _ = self._advance(
            parts,
            self.round_step,
            prev_pos,
            prev_heading,
            pos,
            ate_power,
            fright,
            list(blinky.items()),
            True,
        )
        beliefs: list = [{}, {}, {}, {}]
        for gi, c, d, to, st, half, w in out:
            b = beliefs[gi]
            key = (c, d, to, st, half)
            b[key] = b.get(key, 0.0) + w
        self.beliefs = beliefs
        # the countdown is on screen; trust it over the count kept here
        self.fright = obs.power
        self._observe(seen)

    def _new_round(self, pos: int, heading: int, obs: Frame, seen: dict) -> None:
        self.stats["rounds"] += 1
        self.pos, self.heading = pos, heading
        self.pellets = ALL_PELLETS
        self.score = obs.score
        self.round_step = 0
        self.fright = obs.power
        self.beliefs = [dict(b) for b in self._fresh_beliefs]
        self._tour = None
        if not self._pellets_match(self.pellets, seen):
            # not a fresh maze (the agent was attached mid-round): take what
            # is in view as it is and assume the rest is still there
            for c, ch in seen.items():
                if PBIT[c] and ch == ".":
                    self.pellets &= ~PBIT[c]
        self._observe(seen)

    def _observe(self, seen: dict) -> None:
        """Drop every ghost state the frame rules out."""
        pac = self.pos
        named = {}
        loose = {}
        for c, ch in seen.items():
            if ch in GHOST_GLYPHS:
                named[GHOST_GLYPHS.index(ch)] = c
            elif ch == "S":
                loose[c] = FRIGHTENED
            elif ch == "E":
                loose[c] = EYES

        prior = self.beliefs

        def hidden_by_later(gi: int, c: int, ch: str) -> bool:
            # a ghost with a higher index is drawn over this one
            if ch in GHOST_GLYPHS:
                return GHOST_GLYPHS.index(ch) > gi
            want = loose.get(c)
            if want is None:
                return False
            return any(
                s[0] == c and s[3] == want for gj in range(gi + 1, 4) for s in prior[gj]
            )

        beliefs = []
        for gi in range(4):
            at = named.get(gi)
            kept = {}
            for s, w in prior[gi].items():
                c, st = s[0], s[3]
                if at is not None:
                    if c == at and st == NORMAL:
                        kept[s] = w
                    continue
                ch = seen.get(c)
                if (
                    ch is None
                    or (st != NORMAL and loose.get(c) == st)
                    # Pac-Man is drawn over a ghost on his cell: a pair of
                    # eyes, or a ghost that came home under him this step
                    or c == pac
                    or hidden_by_later(gi, c, ch)
                ):
                    kept[s] = w
            beliefs.append(kept)

        # a frightened ghost or a pair of eyes only one ghost can explain
        unexplained = []
        for c, st in loose.items():
            who = [
                gi
                for gi in range(4)
                if any(s[0] == c and s[3] == st for s in beliefs[gi])
            ]
            if len(who) == 1:
                gi = who[0]
                beliefs[gi] = {
                    s: w for s, w in beliefs[gi].items() if s[0] == c and s[3] == st
                }
            elif not who:
                unexplained.append((c, st))

        for gi in range(4):
            if not beliefs[gi]:
                beliefs[gi] = self._reseed(gi, named.get(gi), unexplained, seen)
            total = sum(beliefs[gi].values())
            beliefs[gi] = {s: w / total for s, w in beliefs[gi].items()}
        self.beliefs = beliefs

    def _reseed(self, gi: int, at, unexplained: list, seen: dict) -> dict:
        """A belief the frames emptied: start over from what is in view.

        The rules are exact, so this should not happen; it keeps the agent
        playing if it ever does.
        """
        self.stats["reseeds"] += 1
        half = 1 if self.fright > 0 else 0
        if at is not None:
            spots = [(at, NORMAL)]
        elif unexplained:
            spots = [unexplained.pop()]
        else:
            spots = [(c, NORMAL) for c in OPEN if c not in seen]
        return {
            (c, d, -1, st, h): 1.0
            for c, st in spots
            for d in range(4)
            if NB[c][d ^ 2] >= 0
            for h in ((0, 1) if st == FRIGHTENED else (half,))
        }

    # ---- choosing a move ------------------------------------------------

    def _root_particles(self) -> list:
        """The ghost states worth searching against, a few per ghost.

        Usually a ghost is in exactly one state. One that wandered off
        frightened may be in hundreds; the search follows the nearest and
        likeliest of them, each at its own weight.
        """
        T = self.tuning
        dist = DIST[self.pos]
        parts = []
        for gi, belief in enumerate(self.beliefs):
            if len(belief) == 1:
                ((state, w),) = belief.items()
                parts.append((gi, *state, w))
                continue
            # ghosts still waiting at home differ only in how long for:
            # search against the one that leaves soonest
            merged: dict = {}
            for (c, d, to, st, half), w in belief.items():
                key = (c, d, st, half)
                if key in merged:
                    pw, pto = merged[key]
                    merged[key] = (pw + w, min(pto, to))
                else:
                    merged[key] = (w, to)
            ranked = sorted(
                ((w / (4 + dist[k[0]]), k, w, to) for k, (w, to) in merged.items()),
                reverse=True,
            )
            for _, (c, d, st, half), w, to in ranked[: T.max_particles]:
                parts.append((gi, c, d, to, st, half, w))
        return parts

    def _chase(self, pc: int, fr: int, parts: list, tots) -> float:
        """What the frightened ghosts still loose are worth from pc."""
        T = self.tuning
        worth = 0.0
        from_pc = DIST[pc]
        for gi, c, _, _, st, _, w in parts:
            if st == FRIGHTENED:
                reach = 1.0 - T.chase_pace * from_pc[c] / (fr + 1)
                if reach > 0:
                    worth += reach * w / tots[gi]
        return T.chase_value * self.rules.ghost_reward * worth

    def _skips_power(self) -> bool:
        """Whether the tour leaves the power pellets out (see `Tuning`)."""
        T = self.tuning
        return (
            self._discount is not None
            and T.discount_skip_power
            and T.discount_power_ghosts > 0
            and bool(self.pellets & ~POWER_MASK)
        )

    def _route(self) -> list:
        """Bring the tour up to date and return it.

        The tour is the order the agent means to eat the remaining pellets
        in. It is built when a round starts, loses each pellet as it is
        eaten, is tidied a little every step, and is started over when
        Pac-Man has been driven far from where it begins.
        """
        T = self.tuning
        pos, pellets = self.pos, self.pellets
        if self._skips_power():
            pellets &= ~POWER_MASK
        tour = self._tour
        if tour is not None and len(tour) != pellets.bit_count():
            tour = [c for c in tour if PBIT[c] & pellets]
            if not tour and pellets:
                tour = None
        if tour is None:
            key = (pos, pellets)
            made = self._tour_cache.get(key)
            if made is None:
                cells = [c for c in PELLETS if PBIT[c] & pellets]
                made = polish_tour(
                    pos, nearest_tour(pos, cells, DIST), DIST, T.tour_polish
                )
                self._tour_cache[key] = made
            tour = list(made)

        if tour:
            improve_tour(pos, tour, DIST, T.tour_moves)
            if DIST[pos][tour[0]] > T.tour_drift or self.round_step % 16 == 0:
                fresh = nearest_tour(pos, tour, DIST)
                improve_tour(pos, fresh, DIST, 100)
                if tour_length(pos, fresh, DIST) < tour_length(pos, tour, DIST):
                    tour = fresh
        self._tour = tour
        return tour

    def _paced_worth(self, parts0: list, death_value: float):
        """How playing for reward per step values where a path ends.

        The rest of the maze is read off the tour: the pellets still on it,
        less the pace for the steps it takes to walk them. A path that
        follows the tour shortens it step for step; one that leaves pellets
        behind or dawdles does not, and that is what keeps the search
        efficient. The tour is only worth walking while it keeps ahead of
        the pace: past that point the agent would rather be caught and
        start on a full maze, so the rest of it counts for nothing.

        Returns the hooks the search calls: `eat` and `uneat` as a path
        takes a pellet and gives it back, `uturn` for whether the tour
        doubles back here, `leaf` for the worth of a path's end, and the
        two numbers a path carries: the walk between the pellets still on
        the tour, and how many there are.
        """
        T, R = self.tuning, self.rules
        pace, die_time = self._pace, T.die_time
        pellet_reward = R.pellet_reward
        pos = self.pos
        tour = self._route()
        dist = DIST
        chase = self._chase

        def bail(pc, fr, parts):
            """What giving the maze up is worth from pc: a catch, and the
            pace for the steps until a ghost can be found to do it."""
            wait = die_time
            from_pc = dist[pc]
            for _, c, _, _, st, _, w in parts:
                if st == NORMAL and w >= 0.3:
                    t = from_pc[c] / 2 + 1
                    if t < wait:
                        wait = t
            return death_value - pace * (wait + fr)

        # how far along the tour is worth walking
        n = len(tour)
        after = death_value - pace * die_time
        best, worth = bail(pos, self.fright, parts0), 0
        steps, at = 0, pos
        for j, c in enumerate(tour, 1):
            steps += dist[at][c]
            at = c
            v = j * pellet_reward - pace * steps + (R.clear_reward if j == n else after)
            if v > best:
                best, worth = v, j
        end_value = R.clear_reward if worth == n else after

        # That stretch as a linked list, so a path can drop the pellets it
        # eats and put them back: index 0 is the head, the tail is past the
        # last pellet.
        cell_of = [0, *tour[:worth], 0]
        tail = worth + 1
        nxt = list(range(1, worth + 3))
        prv = list(range(-1, worth + 1))
        where = [0] * NCELL
        for i in range(1, tail):
            where[cell_of[i]] = i
        inner0 = sum(dist[cell_of[i]][cell_of[i + 1]] for i in range(1, worth))

        # Playing for discounted return, the rest of the tour is worth what
        # it pays discounted by when it pays it: its pellets taken as evenly
        # spread along the walk, the power pellets at some ghosts each on
        # top, and the clear at the end of it.
        discount = self._discount
        if discount is not None:
            step = discount**T.discount_slowdown
            # fading out as the last plain pellets go, as in careful play,
            # or the last power pellets would never be eaten
            plain_left = (self.pellets & ~POWER_MASK).bit_count()
            power_worth = T.discount_power_ghosts * R.ghost_reward
            power_worth *= min(1.0, plain_left / T.power_fade)
            # a power pellet off the tour pays its pellet on top
            power_extra = power_worth + (pellet_reward if self._skips_power() else 0)

        def eat(nc, pellets, inner, count):
            i = where[nc]
            if not i:
                return inner, count, 0, False
            # take the pellet off the tour, bridging the gap it leaves
            left, right = prv[i], nxt[i]
            nxt[left], prv[right] = right, left
            if left:
                row = dist[cell_of[left]]
                inner -= row[nc]
                if right != tail:
                    inner += row[cell_of[right]] - dist[nc][cell_of[right]]
            elif right != tail:
                inner -= dist[nc][cell_of[right]]
            return inner, count - 1, i, left == 0

        def uneat(i):
            nxt[prv[i]] = i
            prv[nxt[i]] = i

        def uturn(pc, behind):
            first = nxt[0]
            return (
                first != tail
                and dist[behind][cell_of[first]] < dist[pc][cell_of[first]]
            )

        def leaf(pc, pellets, fr, parts, tots, inner, count):
            first = nxt[0]
            if discount is not None and first == tail:
                # only power pellets left off the tour, if anything
                power = (pellets & POWER_MASK).bit_count()
                v = power * power_extra + step ** (8 * power) * R.clear_reward
            elif first == tail:
                v = end_value
            elif discount is not None:
                lead = step ** dist[pc][cell_of[first]]
                if count > 1:
                    q = step ** (inner / (count - 1))
                    spread = (1.0 - q**count) / (1.0 - q) if q < 1.0 else count
                else:
                    spread = 1.0
                v = lead * (
                    spread * pellet_reward
                    + (pellets & POWER_MASK).bit_count() * power_extra * spread / count
                    + step**inner * R.clear_reward
                )
            else:
                walk = dist[pc][cell_of[first]] + inner
                v = count * pellet_reward - pace * walk + end_value
            b = bail(pc, fr, parts)
            if b > v:
                v = b
            if fr > 0:
                v += chase(pc, fr, parts, tots)
            return v

        return eat, uneat, uturn, leaf, inner0, worth

    def _careful_worth(self):
        """How playing to clear values where a path ends.

        Every pellet left counts, fading with its distance, and every
        fragment of pellets a path strands costs the trip back for it. An
        uneaten power pellet is worth more than the pellet, so it is kept
        for when there are ghosts to eat or no other way out.

        Returns the same hooks as `_paced_worth`; here a path carries the
        number of fragments it has made.
        """
        T = self.tuning
        pellets0 = self.pellets
        plain_left = (pellets0 & ~POWER_MASK).bit_count()
        power_value = max(T.power_value, T.power_ghosts * self.rules.ghost_reward)
        power_value *= min(1.0, plain_left / T.power_fade)
        field_weight = T.field_weight
        fragment_cost, fragment_scan = T.fragment_cost, T.fragment_scan
        chase = self._chase

        has = np.zeros(NPELLET)
        for i in range(NPELLET):
            if pellets0 >> i & 1:
                has[i] = 1.0
        field = (self._field @ has).tolist()
        field_rows = self._field_rows
        eaten = []

        def fragments_made(c, pellets):
            """How many more fragments there are once the pellet at c is gone.

            Only plain pellets count: a power pellet is left for when the
            ghosts are close, so it must not look like a loose end to tidy.
            """
            pellets &= ~POWER_MASK
            sides = [q for q in PELLET_NB[c] if PBIT[q] & pellets]
            if len(sides) < 2:
                return len(sides) - 1
            small, large = 0, 0
            reached: set = set()
            for q in sides:
                if q in reached:
                    continue
                piece = {q}
                stack = [q]
                while stack and len(piece) <= fragment_scan:
                    for m in PELLET_NB[stack.pop()]:
                        if m not in piece and PBIT[m] & pellets:
                            piece.add(m)
                            stack.append(m)
                if stack:
                    # too big to walk: part of the bulk of the maze
                    large = 1
                else:
                    small += 1
                reached |= piece
            return small + large - 1

        def eat(nc, pellets, fragments, _):
            eaten.append(PIDX[nc])
            if not PBIT[nc] & POWER_MASK:
                fragments += fragments_made(nc, pellets)
            return fragments, 0, 1, False

        def uneat(_):
            eaten.pop()

        def leaf(pc, pellets, fr, parts, tots, fragments, _):
            f = field[pc]
            row = field_rows[pc]
            for i in eaten:
                f -= row[i]
            v = field_weight * f - fragment_cost * fragments
            if power_value:
                v += power_value * (pellets & POWER_MASK).bit_count()
            if fr > 0:
                v += chase(pc, fr, parts, tots)
            return v

        return eat, uneat, None, leaf, 0, 0

    def _plan(self) -> int:
        """Search Pac-Man's moves `depth` steps ahead and pick the first one.

        Every path is walked against the ghost particles, which react to it
        move by move as the real ghosts would. It scores what it eats, less
        what dying costs times the odds of it, plus what the place it ends
        in is worth; playing for reward per step, every step is charged the
        pace as well.
        """
        T, R = self.tuning, self.rules
        advance = self._advance
        depth = self._depth
        careful = self._careful
        # careful and discounted play hold power pellets for the ghosts, so
        # may wait beside one for them to come
        holds_power = careful or (
            self._discount is not None and T.discount_power_ghosts > 0
        )
        gam = self._gamma
        pellet_reward, ghost_reward = R.pellet_reward, R.ghost_reward
        clear_reward = R.clear_reward
        frightened_time = R.frightened_time
        near_power = self._near_power
        turn_cost = T.turn_cost
        pellets0 = self.pellets
        death_value = -(
            self._death_cost + T.death_cost_per_pellet * pellets0.bit_count()
        )
        parts0 = self._root_particles()
        if careful:
            pace, risk_gain = 0.0, T.risk_gain
            eat, uneat, uturn, leaf, a0, b0 = self._careful_worth()
        else:
            pace = self._pace
            risk_gain = 1.0 if self._discount is None else T.discount_risk_gain
            eat, uneat, uturn, leaf, a0, b0 = self._paced_worth(parts0, death_value)

        leaves = []
        nodes = 0

        def search(
            k,
            pc,
            hd,
            pellets,
            n,
            fr,
            parts,
            tots,
            alive,
            acc,
            budget,
            first,
            powered,
            a,
            b,
            on_tour,
        ):
            nonlocal nodes
            if k == depth:
                v = acc + alive * gam[k] * leaf(pc, pellets, fr, parts, tots, a, b)
                leaves.append((v, first, (pc, hd, pellets, n, fr, parts, tots), alive))
                return v

            nb = NB[pc]
            back = hd ^ 2
            options = [(m, nb[m], 0) for m in range(4) if m != back and nb[m] >= 0]
            # Deeper in, Pac-Man only turns back where the tour does, or to
            # time a power pellet: beside one he is holding, or just after
            # eating one. Standing still against a wall is for timing too.
            # The very next move may always turn back. Whatever is left out
            # is still tried when every other way ends badly.
            later = []
            timing = budget > 0 and (
                powered or (holds_power and near_power[pc] & pellets)
            )
            if nb[back] >= 0:
                if on_tour and uturn(pc, nb[back]):
                    options.append((back, nb[back], 0))
                else:
                    (options if timing or k == 0 else later).append((back, nb[back], 1))
            if nb[hd] < 0:
                (options if timing else later).append((hd, pc, 1))

            best = -1e18
            g = gam[k + 1]
            n1 = n + 1
            while True:
                for m, nc, special in options:
                    fr1 = fr - 1 if fr > 0 else 0
                    pellets1, reward, ate_power = pellets, 0.0, False
                    a1, b1, token, on_tour1 = a, b, 0, False
                    if nc != pc:
                        bit = PBIT[nc]
                        if bit & pellets:
                            pellets1 = pellets ^ bit
                            reward = pellet_reward
                            if bit & POWER_MASK:
                                ate_power = True
                                fr1 = frightened_time
                            a1, b1, token, on_tour1 = eat(nc, pellets1, a, b)
                    nodes += 1
                    out, dead, eaten = advance(parts, n1, pc, hd, nc, ate_power, fr1)
                    alive1, tots1 = alive, tots
                    if eaten:
                        for gi in range(4):
                            if eaten[gi]:
                                reward += ghost_reward * eaten[gi] / tots[gi]
                    acc1 = acc + alive * (g * reward - pace)
                    if k == 0 and special:
                        acc1 -= turn_cost
                    if dead:
                        survive = 1.0
                        tots1 = list(tots)
                        for gi in range(4):
                            if dead[gi]:
                                survive *= max(0.0, 1.0 - dead[gi] / tots[gi])
                                tots1[gi] = tots[gi] - dead[gi]
                        acc1 += (
                            alive
                            * min(1.0, risk_gain * (1.0 - survive))
                            * g
                            * death_value
                        )
                        alive1 = alive * survive

                    act = m if k == 0 else first
                    if alive1 < 1e-4:
                        v = acc1
                        leaves.append((v, act, None, 0.0))
                    elif pellets1 == 0:
                        v = acc1 + alive1 * g * clear_reward
                        leaves.append((v, act, None, 0.0))
                    else:
                        v = search(
                            k + 1,
                            nc,
                            hd if nc == pc else m,
                            pellets1,
                            n1,
                            fr1,
                            out,
                            tots1,
                            alive1,
                            acc1,
                            budget - special if k else budget,
                            act,
                            ate_power,
                            a1,
                            b1,
                            on_tour1,
                        )
                    if token:
                        uneat(token)
                    if v > best:
                        best = v
                if (
                    not later
                    or not careful
                    or best > acc + 0.5 * alive * g * death_value
                ):
                    return best
                options, later = later, None

        search(
            0,
            self.pos,
            self.heading,
            pellets0,
            self.round_step,
            self.fright,
            parts0,
            (1.0, 1.0, 1.0, 1.0),
            1.0,
            0.0,
            T.dither_budget,
            self.heading,
            False,
            a0,
            b0,
            False,
        )
        self.stats["nodes"] += nodes
        if nodes > T.node_limit and self._depth > T.min_depth:
            self._depth -= 1
        elif nodes < T.node_limit // 2 and self._depth < T.depth:
            self._depth += 1

        leaves.sort(key=lambda leaf: leaf[0], reverse=True)
        best_value, best_action = -1e18, leaves[0][1]
        checks = 0
        checked: dict = {}
        for v, action, state, alive in leaves:
            if v < best_value:
                break
            if careful and state is not None:
                # a leaf that looks good may be a trap: check it for a way
                # to stay alive, and charge it for the odds of there being none
                key = (
                    state[0],
                    state[1],
                    state[2] & POWER_MASK,
                    state[3],
                    state[4],
                    tuple(state[5]),
                )
                q = checked.get(key)
                if q is None:
                    if checks >= T.verify_checks:
                        break
                    checks += 1
                    q = checked[key] = self._survives(state)
                v += alive * gam[depth] * min(1.0, risk_gain * (1.0 - q)) * death_value
            if v > best_value or (v == best_value and action == self.heading):
                best_value, best_action = v, action
        self.stats["checks"] += checks
        return best_action

    def _survives(self, state) -> float:
        """The best odds of living `verify_len` more steps from a leaf.

        Looks for any path that stays clear of the ghosts, safest moves
        first, so it returns at once when there is an easy way out and only
        works hard when there is not.
        """
        T = self.tuning
        advance = self._advance
        length, budget = T.verify_len, T.verify_budget
        target = 1.0 - T.verify_eps
        frightened_time = self.rules.frightened_time
        pc0, hd0, pellets0, n0, fr0, parts0, tots0 = state
        parts0 = [p for p in parts0 if p[6] >= T.verify_floor * tots0[p[0]]]
        best = 0.0
        nodes = 0

        def walk(k, pc, hd, pellets, n, fr, parts, tots, alive):
            nonlocal best, nodes
            if k == length or pellets == 0:
                if alive > best:
                    best = alive
                return
            nodes += 1
            if nodes > budget:
                return
            nb = NB[pc]
            back = hd ^ 2
            threats = [p[1] for p in parts if p[4] == NORMAL and p[6] > 0.05]
            options = []
            for a in range(4):
                nc = nb[a]
                if nc < 0:
                    continue
                dist = DIST[nc]
                room = min([dist[c] for c in threats], default=FAR)
                options.append((room - (2 if a == back else 0), a, nc))
            if nb[hd] < 0:
                dist = DIST[pc]
                room = min([dist[c] for c in threats], default=FAR)
                options.append((room - 3, hd, pc))
            options.sort(reverse=True)

            n1 = n + 1
            for _, a, nc in options:
                fr1 = fr - 1 if fr > 0 else 0
                pellets1, ate_power = pellets, False
                if nc != pc:
                    bit = PBIT[nc]
                    if bit & pellets:
                        pellets1 = pellets ^ bit
                        if bit & POWER_MASK:
                            ate_power = True
                            fr1 = frightened_time
                out, dead, _ = advance(parts, n1, pc, hd, nc, ate_power, fr1)
                alive1, tots1 = alive, tots
                if dead:
                    tots1 = list(tots)
                    for gi in range(4):
                        if dead[gi]:
                            alive1 *= max(0.0, 1.0 - dead[gi] / tots[gi])
                            tots1[gi] = tots[gi] - dead[gi]
                if alive1 > best:
                    walk(
                        k + 1,
                        nc,
                        hd if nc == pc else a,
                        pellets1,
                        n1,
                        fr1,
                        out,
                        tots1,
                        alive1,
                    )
                    if best >= target or nodes > budget:
                        return

        walk(0, pc0, hd0, pellets0, n0, fr0, parts0, tots0, 1.0)
        self.stats["check_nodes"] += nodes
        if nodes > budget and best < target:
            # ran out of looking: not proven dead, not shown safe
            return max(best, 0.3)
        return best


# --------------------------------------------------------------------------
# Running it against the env
# --------------------------------------------------------------------------


def play_budget(
    seed: int,
    steps: int,
    config,
    tuning: Tuning,
    show: int = 0,
    report_discount: float = 0.996,
) -> dict:
    """Play one budget of turns through the ASCII view and score it.

    Episodes run back to back inside the budget: each is one maze, played
    until it is cleared, Pac-Man is caught, or the turns run out. The agent
    gets the frame text and nothing else; the rewards and the env's own
    per-episode metrics are read here only to report how it did.
    """
    from mapox.ascii import AsciiRenderer
    from mapox.envs.rust_env_numpy import RustEnvNumpy
    from mapox.rust_play import ascii_frame

    env = RustEnvNumpy(config, steps)
    renderer = AsciiRenderer(env.obs_vocab, env.action_vocab)
    agent = PacmanAgent(Rules.from_config(config), tuning)
    action = np.zeros(1, np.uint16)

    timestep = env.reset(seed)
    env.consume_metrics()
    total = 0.0
    lives = []
    life_reward, life_start, life_return = 0.0, 0, 0.0
    started = time.perf_counter()
    for step in range(steps):
        frame = renderer.render(timestep.obs[0])
        name = agent.act(frame)
        if show and step % show == 0:
            title = f"seed {seed} step {step}  reward {total:g}  ->  {name}"
            print(
                ascii_frame(
                    renderer, timestep.obs[0], timestep.action_mask[0], title=title
                )
            )
        action[0] = env.action_vocab.id(name)
        timestep = env.step(action)

        reward = float(timestep.reward[0])
        total += reward
        life_reward += reward
        life_return += report_discount ** (step - life_start) * reward
        if timestep.terminated[0]:
            metrics = env.consume_metrics()
            outcome = (
                "clear"
                if metrics["clears"]
                else "death"
                if metrics["deaths"]
                else "cut"
            )
            lives.append(
                {
                    "outcome": outcome,
                    "reward": life_reward,
                    "return": life_return,
                    "steps": step + 1 - life_start,
                    "ghosts": metrics["ghosts_eaten"],
                }
            )
            life_reward, life_start, life_return = 0.0, step + 1, 0.0
    return {
        "seed": seed,
        "reward": total,
        "lives": lives,
        "seconds": time.perf_counter() - started,
        "stats": dict(agent.stats),
    }


def _play(args) -> dict:
    return play_budget(*args)


def summarize(results: list, steps: int, report_discount: float = 0.996) -> str:
    """The scores of some budgets, by budget and by episode."""
    rewards = np.array([r["reward"] for r in results])
    lives = [life for r in results for life in r["lives"]]
    clears = [life for life in lives if life["outcome"] == "clear"]
    caught = [life for life in lives if life["outcome"] == "death"]
    ended = clears + caught
    seconds = sum(r["seconds"] for r in results)
    moves = sum(r["stats"]["steps"] for r in results)
    nodes = sum(r["stats"]["nodes"] + r["stats"]["check_nodes"] for r in results)
    reseeds = sum(r["stats"]["reseeds"] for r in results)
    # Reward per episode the way the env's own `reward` metric counts it: a
    # budget's reward over all the episodes played in it, the one the budget
    # cut short included; then the mean of that over the budgets.
    per_episode = np.array([r["reward"] / len(r["lives"]) for r in results])
    # The first episode of each budget, discounted from its first step: the
    # return an RL value function estimates from a fresh maze. No first
    # episode is long enough for the budget to cut it.
    first_return = np.array([r["lives"][0]["return"] for r in results])

    def spread(values) -> float:
        return values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0

    def mean(episodes: list, key: str) -> float:
        return float(np.mean([life[key] for life in episodes]))

    lines = [
        f"budgets           {len(results)} x {steps} turns",
        (
            f"reward / budget   {rewards.mean():.1f} +- {spread(rewards):.1f}"
            f"  (min {rewards.min():g}, max {rewards.max():g})"
        ),
        f"reward / turn     {rewards.mean() / steps:.3f}",
        (
            f"episodes          {len(clears)} cleared, {len(caught)} caught,"
            f" {len(lives) - len(ended)} cut off by the budget"
        ),
        (
            f"reward / episode  {per_episode.mean():.2f} +- {spread(per_episode):.2f}"
            f"  (every episode, cut-off ones included)"
        ),
        (
            f"first episode     {first_return.mean():.3f} +- {spread(first_return):.3f}"
            f"  (return discounted at {report_discount:g}, first episode of each budget)"
        ),
    ]
    if ended:
        lines.append(
            f"finished episode  {mean(ended, 'steps'):.0f} turns, "
            f"{mean(ended, 'reward'):.1f} reward"
        )
    if clears:
        lines.append(f"clear rate        {100 * len(clears) / len(ended):.1f}%")
        lines.append(
            f"a clear           {mean(clears, 'steps'):.0f} turns, "
            f"{mean(clears, 'ghosts'):.2f} ghosts eaten"
        )
    lines.append(
        f"agent             {1000 * seconds / moves:.1f} ms/turn, {nodes / moves:.0f} nodes/turn"
    )
    if reseeds:
        lines.append(
            f"warning           the ghost tracking lost a ghost {reseeds} times"
        )
    return "\n".join(lines)


def main() -> None:
    from mapox.envs.rust_env_numpy import RustPacmanConfig

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--budgets", type=int, default=16, help="how many budgets to play"
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=1024,
        help="turns in a budget; episodes run back to back until they are spent",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the first budget")
    parser.add_argument("--procs", type=int, default=mp.cpu_count())
    parser.add_argument(
        "--config",
        metavar="JSON",
        help="the env config: `RustPacmanConfig` fields as JSON, or a file of them",
    )
    parser.add_argument(
        "--randomize-start",
        action="store_true",
        help="start each round on a random pellet",
    )
    parser.add_argument(
        "--play-to-clear",
        action="store_true",
        help="value staying alive: play every maze to the last pellet instead "
        "of for reward per step",
    )
    parser.add_argument(
        "--play-for-return",
        action="store_true",
        help="play each episode for its discounted return, as an RL value "
        "function counts it (the discount is --discount, 0.996 by default)",
    )
    parser.add_argument(
        "--report-discount",
        type=float,
        default=0.996,
        help="the discount the first-episode return in the report uses",
    )
    parser.add_argument(
        "--show",
        type=int,
        nargs="?",
        const=1,
        default=0,
        metavar="N",
        help="print every N-th frame of one budget with the move chosen",
    )
    # every search knob is a flag too: --depth 24, --pace 0.7, ...
    defaults = Tuning()
    for f in fields(Tuning):
        kind = type(getattr(defaults, f.name))
        if kind is bool:
            continue
        parser.add_argument(
            f"--{f.name.replace('_', '-')}",
            type=float if kind is type(None) else kind,
            default=None,
            help=argparse.SUPPRESS,
        )
    args = parser.parse_args()

    settings = {}
    if args.config:
        text = args.config
        if Path(text).is_file():
            text = Path(text).read_text()
        text = text.strip().rstrip(",")
        # a few keys lifted out of a larger config are fine as they are
        settings = json.loads(text if text.startswith("{") else "{" + text + "}")
    if args.randomize_start:
        settings["randomize_starting_position"] = True
    config = RustPacmanConfig(**settings)
    overrides = {
        f.name: getattr(args, f.name)
        for f in fields(Tuning)
        if getattr(args, f.name, None) is not None
    }
    if args.play_to_clear:
        overrides = {**PLAY_TO_CLEAR, **overrides}
    elif args.play_for_return:
        overrides = {**PLAY_FOR_RETURN, **overrides}
    tuning = replace(defaults, **overrides)

    if args.show:
        result = play_budget(
            args.seed, args.turns, config, tuning, args.show, args.report_discount
        )
        print(summarize([result], args.turns, args.report_discount))
        return

    jobs = [
        (args.seed + i, args.turns, config, tuning, 0, args.report_discount)
        for i in range(args.budgets)
    ]
    if args.procs > 1 and args.budgets > 1:
        with mp.Pool(min(args.procs, args.budgets)) as pool:
            results = list(pool.imap_unordered(_play, jobs))
    else:
        results = [_play(job) for job in jobs]
    results.sort(key=lambda r: r["seed"])
    print(summarize(results, args.turns, args.report_discount))


if __name__ == "__main__":
    main()
