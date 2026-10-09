"""Generates the mapox tileset: one 12px sprite per observation symbol.

    python scripts/make_tileset.py            # writes both asset copies
    python scripts/make_tileset.py --preview  # also writes an 8x zoomed sheet

The sheet is drawn from code rather than kept as a hand-edited image so every
sprite shares one palette and one floor, and so adding a symbol is a diff you
can review. Stdlib only: the PNG is encoded by hand, no Pillow needed.

Layout matches the renderers' addressing: 12px tiles on a 13px stride with a
1px border, so tile (col, row) starts at (col * 13 + 1, row * 13 + 1). The
slot each sprite lands in is fixed by `SHEET` below, and the tables in
`crates/mapox-core/src/render/mod.rs` and `python/mapox/renderer.py` point
into it; moving a sprite means updating both.

Every tile is fully opaque. A cell shows exactly one tile, so agents, chests
and pickups carry their own patch of floor instead of relying on a layer
underneath.
"""

import argparse
import random
import struct
import zlib
from pathlib import Path

TILE = 12
PAD = 1
COLS = 12

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = [
    ROOT / "crates/mapox-core/assets/mapox_tileset.png",
    ROOT / "python/mapox/assets/mapox_tileset.png",
]

Color = tuple[int, int, int]
Sprite = list[list[Color]]


def hex_color(value: str) -> Color:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def scale(color: Color, factor: float) -> Color:
    return tuple(max(0, min(255, round(c * factor))) for c in color)  # type: ignore[return-value]


def mix(a: Color, b: Color, t: float) -> Color:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


# --- palette ---------------------------------------------------------------

OUTLINE = hex_color("10121a")

FLOOR = hex_color("1c1f27")
FLOOR_DOT = hex_color("2a2e39")

MASK = hex_color("0b0c10")
MASK_HATCH = hex_color("171920")

UI = hex_color("2c3349")
UI_DOT = hex_color("3d4661")

STONE = hex_color("5c6374")
STONE_LIGHT = hex_color("8089a0")
STONE_DARK = hex_color("444a58")
MORTAR = hex_color("2a2d38")

DIRT = hex_color("6e4a2c")
DIRT_DARK = hex_color("54371f")
DIRT_LIGHT = hex_color("87603a")
PEBBLE = hex_color("a8875e")

WATER = hex_color("1f3f86")
WATER_MID = hex_color("2f5cb8")
WATER_CREST = hex_color("86b4ff")

COPPER_EDGE = hex_color("4f2610")
COPPER_SHADE = hex_color("8c4a20")
COPPER = hex_color("c26f35")
COPPER_LIGHT = hex_color("f0a769")
COPPER_FLANGE = hex_color("d98a4c")

WOOD = hex_color("7a4b26")
WOOD_DARK = hex_color("4a2c14")
WOOD_LIGHT = hex_color("a06a38")
IRON = hex_color("8a93a4")
IRON_LIGHT = hex_color("c8cfdb")
GOLD = hex_color("ffc933")
GOLD_LIGHT = hex_color("fff09a")
GOLD_DARK = hex_color("c7891a")
SPARKLE = hex_color("ffffff")

SKIN = hex_color("eab38a")
SKIN_SHADE = hex_color("b97c56")
EYE = OUTLINE
BOOT = hex_color("3b2b20")

GRASS_DIM = hex_color("2d4a33")
GRASS_DIM_LIGHT = hex_color("3b6343")
GRASS = hex_color("3f9a4a")
GRASS_LIGHT = hex_color("6fd26a")

# Snake colours, in the order of the AGENT_SNAKE_* symbols they draw.
SNAKE_COLORS = {
    "red": "e5383f",
    "orange": "f77f22",
    "yellow": "fae255",
    "gold": "c99a1e",
    "green": "3ec54b",
    "blue": "3a78ff",
    "purple": "9d4edd",
    "pink": "ff6fb5",
    "gray": "8b93a1",
    "white": "f2f2ee",
}

UI_DIGIT = hex_color("e8ecf4")
UI_DIGIT_SHADOW = hex_color("161a26")

PAC = hex_color("ffd43b")
PAC_EDGE = hex_color("c99a1e")
PELLET = hex_color("ffd9b3")
POWER = hex_color("ffb08a")
POWER_LIGHT = hex_color("fff1e6")

# Ghost bodies, in the order of the AGENT_GHOST_* symbols they draw.
GHOST_COLORS = {
    "pinky": "ffa3d8",
    "blinky": "e5383f",
    "inky": "3fd6e8",
    "clyde": "ffb347",
}
GHOST_FRIGHTENED = hex_color("2f4bd8")
GHOST_FRIGHTENED_FACE = hex_color("ffd9c7")
GHOST_EYE = hex_color("f2f2ee")
GHOST_PUPIL = hex_color("2121c4")

HEART = hex_color("e5383f")
HEART_DARK = hex_color("8f1d22")
HEART_LIGHT = hex_color("ff8a8f")

FLAME_DEEP = hex_color("e85d04")
FLAME = hex_color("ff9f1c")
FLAME_CORE = hex_color("ffd166")
EMBER = hex_color("c2410c")

MEAT = hex_color("c8763c")
MEAT_DARK = hex_color("8c4a20")
MEAT_LIGHT = hex_color("f0a769")
BONE = hex_color("f2efe6")

TEAM_RED = hex_color("e5383f")
TEAM_BLUE = hex_color("3a78ff")
TEAM_NEUTRAL = hex_color("d6dbe4")


# --- drawing helpers -------------------------------------------------------


def blank(color: Color) -> Sprite:
    return [[color] * TILE for _ in range(TILE)]


def floor() -> Sprite:
    """The walkable ground every open-cell sprite stands on.

    One brighter texel in the corner of each cell puts a faint dot lattice
    across open ground, which makes distances countable at a glance.
    """
    sprite = blank(FLOOR)
    sprite[0][0] = FLOOR_DOT
    return sprite


def stamp(sprite: Sprite, art: list[str], palette: dict[str, Color], dx=0, dy=0):
    """Paints ASCII `art` onto `sprite`; '.' leaves the pixel alone."""
    for y, line in enumerate(art):
        for x, char in enumerate(line):
            if char != ".":
                sprite[y + dy][x + dx] = palette[char]
    return sprite


def check(art: list[str], name: str) -> list[str]:
    assert len(art) == TILE and all(len(row) == TILE for row in art), name
    return art


# --- terrain ---------------------------------------------------------------


def ui() -> Sprite:
    sprite = blank(UI)
    for y in range(TILE):
        for x in range(TILE):
            if x % 4 == 1 and y % 4 == 1:
                sprite[y][x] = UI_DOT
    return sprite


def mask() -> Sprite:
    """Out of sight: a dim diagonal hatch, darker than any floor."""
    sprite = blank(MASK)
    for y in range(TILE):
        for x in range(TILE):
            if (x + y) % 4 == 0:
                sprite[y][x] = MASK_HATCH
    return sprite


def wall() -> Sprite:
    """Dressed stone in running bond. Seamless, so walls read as one mass.

    Two courses of 6px per tile keep the bond alternating across tile seams.
    """
    sprite = blank(STONE)
    rng = random.Random(3)
    for y in range(TILE):
        course, yy = divmod(y, 6)
        for x in range(TILE):
            xx = (x + course * 6) % TILE
            if yy == 5 or xx == 0:
                color = MORTAR
            elif yy == 0 or xx == 1:
                color = STONE_LIGHT
            elif yy == 4 or xx == TILE - 1:
                color = STONE_DARK
            else:
                color = STONE_DARK if rng.random() < 0.08 else STONE
            sprite[y][x] = color
    return sprite


DIRT_CLODS = check(
    [
        "..l......k..",
        ".kk...ll....",
        "......kk..l.",
        "..pl........",
        "..kk...k....",
        "l.......pl..",
        "k...ll..kk..",
        "....kkk....l",
        ".l.........k",
        ".k...pl..l..",
        ".....kk..kk.",
        "..k.........",
    ],
    "dirt",
)


def dirt() -> Sprite:
    """Diggable earth: warm, crumbly and structureless, unlike cut stone.

    Clods are lit from above, each a light crumb over its own shadow, and the
    ones on the edges wrap so a bank of dirt shows no seams.
    """
    return stamp(
        blank(DIRT), DIRT_CLODS, {"k": DIRT_DARK, "l": DIRT_LIGHT, "p": PEBBLE}
    )


def water() -> Sprite:
    """Deep water with wave crests that wrap across tile edges."""
    sprite = blank(WATER)
    for y in range(TILE):
        for x in range(TILE):
            if (x + 2 * y) % 12 in (0, 1) and y % 3 == 0:
                sprite[y][x] = WATER_MID
    for ox, oy in [(1, 3), (7, 9)]:
        for x, y, color in [
            (0, 1, WATER_MID),
            (1, 0, WATER_CREST),
            (2, 0, WATER_CREST),
            (3, 1, WATER_MID),
        ]:
            sprite[(oy + y) % TILE][(ox + x) % TILE] = color
    return sprite


def pipe_horizontal() -> Sprite:
    """A copper slideway. Flanges on both ends double up at every joint, so a
    run shows how many tiles it spans."""
    sprite = floor()
    bands = [COPPER_EDGE, COPPER_LIGHT, COPPER, COPPER, COPPER_SHADE, COPPER_EDGE]
    for i, color in enumerate(bands):
        for x in range(TILE):
            sprite[3 + i][x] = color
    for x in (0, TILE - 1):
        for y in range(2, 10):
            sprite[y][x] = COPPER_EDGE if y in (2, 9) else COPPER_FLANGE
    # rivet highlights on the flanges
    sprite[4][0] = sprite[4][TILE - 1] = COPPER_LIGHT
    return sprite


def pipe_vertical() -> Sprite:
    horizontal = pipe_horizontal()
    return [[horizontal[x][y] for x in range(TILE)] for y in range(TILE)]


def decor(art: list[str], palette: dict[str, Color]) -> Sprite:
    return stamp(floor(), art, palette)


DECOR_GRASS = check(
    [
        "............",
        "............",
        "............",
        "............",
        "......l.....",
        "...g..l.g...",
        "...g.lg.g...",
        "....gg.gg...",
        ".....ggg....",
        "............",
        "............",
        "............",
    ],
    "decor grass",
)

DECOR_PEBBLES = check(
    [
        "............",
        "............",
        "............",
        "........l...",
        ".......ls...",
        "............",
        "...l........",
        "..lss.......",
        "...s....l...",
        "........s...",
        "............",
        "............",
    ],
    "decor pebbles",
)

DECOR_CRACKS = check(
    [
        "............",
        "............",
        "..c.........",
        "...c........",
        "...cc.......",
        ".....c......",
        ".....c.cc...",
        "......c.....",
        ".........c..",
        "............",
        "............",
        "............",
    ],
    "decor cracks",
)

DECOR_MOSS = check(
    [
        "............",
        "............",
        "............",
        ".......g....",
        "......gmg...",
        "..g....g....",
        ".gmg........",
        "..g.....g...",
        "........m...",
        "............",
        "............",
        "............",
    ],
    "decor moss",
)

GRASS_TILE = check(
    [
        "............",
        "..l......l..",
        "..g..l...g..",
        ".g.g.g..g.g.",
        ".g.g.g..g.g.",
        "....g.......",
        "......l.....",
        ".l....g..l..",
        ".g...g.g.g..",
        "g.g..g.g.g.g",
        "g.g.........",
        "............",
    ],
    "grass",
)


# --- items -----------------------------------------------------------------

CHEST_LOCKED = check(
    [
        "............",
        "............",
        "...OOOOOO...",
        "..OlllWWwO..",
        ".OIlWWWWwIO.",
        ".OIWWWWWwIO.",
        ".OOOOLLOOOO.",
        ".OIwwLkwwIO.",
        ".OIWWLLWwIO.",
        ".OIWWWWWwIO.",
        ".OOOOOOOOOO.",
        "............",
    ],
    "chest locked",
)

CHEST_OPEN = check(
    [
        "............",
        "..OOOOOOOO..",
        ".OwwwwwwwwO.",
        ".OwgG*GGgwO.",
        ".OgGhGGGGgO.",
        ".OGGGGh*GGO.",
        ".OOOOLLOOOO.",
        ".OIwwLLwwIO.",
        ".OIWWWWWwIO.",
        ".OIWWWWWwIO.",
        ".OOOOOOOOOO.",
        "............",
    ],
    "chest open",
)

CHEST_PALETTE = {
    "O": OUTLINE,
    "W": WOOD,
    "w": WOOD_DARK,
    "l": WOOD_LIGHT,
    "I": IRON,
    "L": IRON_LIGHT,
    "k": OUTLINE,
    "G": GOLD,
    "g": GOLD_DARK,
    "h": GOLD_LIGHT,
    "*": SPARKLE,
}

APPLE = check(
    [
        "............",
        ".......lL...",
        "......sll...",
        "...OOOsOO...",
        "..ORRRsRRO..",
        ".ORwRRRRRrO.",
        ".ORwRRRRRrO.",
        ".ORRRRRRrrO.",
        ".OrRRRRrrrO.",
        "..OrrRrrrO..",
        "...OOOOOO...",
        "............",
    ],
    "apple",
)

APPLE_PALETTE = {
    "O": hex_color("3a0d12"),
    "R": hex_color("ff4d4d"),
    "r": hex_color("b3202b"),
    "w": hex_color("ffd6d6"),
    "l": GRASS,
    "L": GRASS_LIGHT,
    "s": hex_color("6b4423"),
}

BOLT = check(
    [
        "............",
        "............",
        "............",
        ".....O......",
        "....OYO.....",
        "...OYWYO....",
        "....OYO.....",
        ".....O......",
        "............",
        "............",
        "............",
        "............",
    ],
    "bolt",
)

BANNER = check(
    [
        "..O.........",
        ".OPOOOOOOO..",
        ".OPCCCCCCCO.",
        ".OPCCCCCCO..",
        ".OPcccccO...",
        ".OPCCCCCCO..",
        ".OPOOOOOOO..",
        ".OPO........",
        ".OPO........",
        ".OPO........",
        "OSSSO.......",
        "OOOOO.......",
    ],
    "banner",
)


def banner(team: Color) -> Sprite:
    return stamp(
        floor(),
        BANNER,
        {"O": OUTLINE, "P": IRON_LIGHT, "S": IRON, "C": team, "c": scale(team, 0.7)},
    )


# --- agents ----------------------------------------------------------------

# A blank token rather than a character: no hair, skin or clothes, nothing
# that suggests a role, just enough face and limbs to read as someone.
AGENT = check(
    [
        "............",
        "....OOOO....",
        "...OhhWWO...",
        "..OhWWWWwO..",
        "..OWEWWEwO..",
        "..OWEWWEwO..",
        ".OOWWWWWwOO.",
        ".OWOWWWWOwO.",
        ".OWOWWWWOwO.",
        ".OOOWWWwOOO.",
        "...OwOOwO...",
        "...OOOOOO...",
    ],
    "agent",
)

AGENT_PALETTE = {
    "O": OUTLINE,
    "W": hex_color("d3d7e0"),
    "w": hex_color("9ea4b2"),
    "h": hex_color("f4f6fa"),
    "E": EYE,
}

# Scouts are rabbits: quick, and all ears.
RABBIT = check(
    [
        "..OO....OO..",
        ".OWPO..OPWO.",
        ".OWPO..OPWO.",
        "..OWPOOPWO..",
        "..OWWWWWWO..",
        ".OWEWWWWEWO.",
        ".OWWWPPWWWO.",
        "..OwWWWWwO..",
        "..OOwWWwOO..",
        ".OWWOCCOWWO.",
        ".OWwWCCWwWO.",
        "..OOOOOOOO..",
    ],
    "rabbit",
)

RABBIT_PALETTE = {
    "O": OUTLINE,
    "W": hex_color("ece7df"),
    "w": hex_color("b3aca3"),
    "P": hex_color("f29ab0"),
    "C": hex_color("fffaf2"),
    "E": EYE,
}

# Harvesters are turtles: slow and armoured, mid-plod with the head craned
# up on an S of neck. Drawn as a bare silhouette, with gaps rather than an
# outline separating neck, shell and legs.
TURTLE = check(
    [
        "............",
        ".SSS........",
        "SESES.......",
        "SSSSS.......",
        "...SS.......",
        ".SSS..h.....",
        ".....hhGg...",
        "..SSGGGgGg..",
        ".SRRRRRRRRS.",
        ".S......SSS.",
        "..SS.SS.....",
        ".SS...SS.S..",
    ],
    "turtle",
)

TURTLE_PALETTE = {
    "G": hex_color("4f9a3a"),
    "g": hex_color("2f6424"),
    "h": hex_color("8ccf5e"),
    "R": hex_color("d8b860"),
    "S": hex_color("a6c77a"),
    "E": EYE,
}


def snake_segment(base: Color, eyes: bool = False) -> Sprite:
    """A rounded body scale. Heads and tails share this tile, so the colour
    alone carries the owner; the 1px floor gap shows each segment."""
    sprite = floor()
    light = mix(base, (255, 255, 255), 0.35)
    shade = scale(base, 0.62)
    edge = scale(base, 0.35)
    lo, hi = 1, TILE - 2
    for y in range(lo, hi + 1):
        for x in range(lo, hi + 1):
            corner = (x in (lo, hi)) and (y in (lo, hi))
            if corner:
                continue
            if x in (lo, hi) or y in (lo, hi):
                color = edge
            elif y == lo + 1 or x == lo + 1:
                color = light
            elif y == hi - 1 or x == hi - 1:
                color = shade
            else:
                color = base
            sprite[y][x] = color
    # a diamond of scale texture in the middle
    for x, y in [(5, 4), (4, 5), (6, 5), (5, 6)]:
        sprite[y][x] = shade
    if eyes:
        for x in (3, 7):
            sprite[3][x] = sprite[3][x + 1] = (255, 255, 255)
            sprite[4][x] = (255, 255, 255)
            sprite[4][x + 1] = OUTLINE
    return sprite


def facing_pip(sprite: Sprite, direction: int, color: Color) -> Sprite:
    """Marks the side an agent faces: 0 up, 1 right, 2 down, 3 left, in screen
    space (the renderers draw env +y upward)."""
    wide = [(4, 0), (5, 0), (6, 0), (7, 0), (5, 1), (6, 1)]
    for along, depth in wide:
        x, y = {
            0: (along, depth),
            1: (TILE - 1 - depth, along),
            2: (along, TILE - 1 - depth),
            3: (depth, along),
        }[direction]
        sprite[y][x] = color
    return sprite


KNIGHT = check(
    [
        "............",
        "............",
        "....OOOO....",
        "...OLMMmO...",
        "...OMkkmO...",
        "..OOMMMmOO..",
        ".OMTTTTTTmO.",
        ".OMTTWWTtmO.",
        "..OTTWWTtO..",
        "..OttttttO..",
        "..OKKOOKKO..",
        "............",
    ],
    "knight",
)

ARCHER = check(
    [
        "............",
        "............",
        "....OO......",
        "...OTTO.....",
        "..OTSSTO.b..",
        "..OSESEO..b.",
        ".OTTTTTTOsb.",
        ".OTTTttTOsb.",
        ".OTTTttTOsb.",
        "..OttttO.b..",
        ".OKKOOKKO...",
        "............",
    ],
    "archer",
)


def soldier(art: list[str], team: Color, direction: int, wounded: bool) -> Sprite:
    palette = {
        "O": OUTLINE,
        "M": IRON_LIGHT,
        "m": IRON,
        "L": SPARKLE,
        "k": OUTLINE,
        "T": team,
        "t": scale(team, 0.65),
        "W": hex_color("f2f2ee"),
        "S": SKIN,
        "E": EYE,
        "K": BOOT,
        "b": hex_color("a0703c"),
        "s": hex_color("d8d2c4"),
    }
    if wounded:
        palette = {k: scale(v, 0.6) for k, v in palette.items()}
    sprite = stamp(floor(), art, palette)
    return facing_pip(sprite, direction, GOLD_LIGHT)


# --- ui --------------------------------------------------------------------

# 5x7 glyphs for the UI_DIGITS symbols, digit d at index d.
DIGIT_GLYPHS = [
    [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
]


def digit(value: int) -> Sprite:
    """A digit on the plain UI ground, drop-shadowed so a run of them reads
    as one number across the band."""
    sprite = blank(UI)
    glyph = DIGIT_GLYPHS[value]
    stamp(sprite, glyph, {"#": UI_DIGIT_SHADOW}, dx=4, dy=3)
    stamp(sprite, glyph, {"#": UI_DIGIT}, dx=3, dy=2)
    return sprite


# --- pac-man ---------------------------------------------------------------


def pacman(direction: int) -> Sprite:
    """Pac-Man mouth-first in `direction`: 0 up, 1 right, 2 down, 3 left, in
    screen space like `facing_pip`."""
    sprite = floor()
    cx = cy = (TILE - 1) / 2
    fx, fy = [(0, -1), (1, 0), (0, 1), (-1, 0)][direction]
    for y in range(TILE):
        for x in range(TILE):
            dx, dy = x - cx, y - cy
            distance = (dx * dx + dy * dy) ** 0.5
            if distance > 5.3:
                continue
            along = dx * fx + dy * fy
            across = abs(dx * fy - dy * fx)
            if along > 0 and across < along * 0.85:
                continue  # the mouth: a ~80 degree wedge toward the heading
            sprite[y][x] = PAC if distance < 4.3 else PAC_EDGE
    # the eye sits above the mouth, or beside it when facing up or down
    px, py = (0, -1) if fx else (-1, 0)
    sprite[round(cy + fy * 0.8 + py * 2.8)][round(cx + fx * 0.8 + px * 2.8)] = OUTLINE
    return sprite


def pellet() -> Sprite:
    sprite = floor()
    for x, y in [(5, 5), (6, 5), (5, 6), (6, 6)]:
        sprite[y][x] = PELLET
    return sprite


def power_pellet() -> Sprite:
    sprite = floor()
    c = (TILE - 1) / 2
    for y in range(TILE):
        for x in range(TILE):
            if (x - c) ** 2 + (y - c) ** 2 <= 3.2**2:
                sprite[y][x] = POWER
    sprite[4][4] = sprite[4][5] = sprite[5][4] = POWER_LIGHT
    return sprite


GHOST = check(
    [
        "....OOOO....",
        "..OGGGGGGO..",
        ".OGGGGGGGGO.",
        ".OGWWGGWWGO.",
        "OGGWPGGWPGGO",
        "OGGWWGGWWGGO",
        "OGGGGGGGGGGO",
        "OGGGGGGGGGGO",
        "OGGGGGGGGGGO",
        "OGGGGGGGGGGO",
        "OGOGGOOGGOGO",
        "OO.OO..OO.OO",
    ],
    "ghost",
)

GHOST_SCARED = check(
    [
        "....OOOO....",
        "..OGGGGGGO..",
        ".OGGGGGGGGO.",
        ".OGGGGGGGGO.",
        "OGGGFFGFFGGO",
        "OGGGFFGFFGGO",
        "OGGGGGGGGGGO",
        "OGFGFGGFGFGO",
        "OGGFGFFGFGGO",
        "OGGGGGGGGGGO",
        "OGOGGOOGGOGO",
        "OO.OO..OO.OO",
    ],
    "frightened ghost",
)

GHOST_EYES = check(
    [
        "............",
        "............",
        "............",
        "..WW....WW..",
        ".WWWW..WWWW.",
        ".WWPP..WWPP.",
        ".WWPP..WWPP.",
        "..WW....WW..",
        "............",
        "............",
        "............",
        "............",
    ],
    "ghost eyes",
)


def ghost(body: Color) -> Sprite:
    return stamp(
        floor(),
        GHOST,
        {"O": OUTLINE, "G": body, "W": GHOST_EYE, "P": GHOST_PUPIL},
    )


# --- survival --------------------------------------------------------------

HEART_ICON = check(
    [
        "............",
        "............",
        "..OOO..OOO..",
        ".OhRROORRrO.",
        ".OhRRRRRRrO.",
        ".ORRRRRRRrO.",
        "..ORRRRRrO..",
        "...ORRRrO...",
        "....ORrO....",
        ".....OO.....",
        "............",
        "............",
    ],
    "heart",
)

DRUMSTICK = check(
    [
        "............",
        "............",
        "....OOOO....",
        "...OMMmmO...",
        "..OMhMMmmO..",
        "..OMMMMmmO..",
        "..OmMMmmmO..",
        "...OmmmmO...",
        "....OOBO....",
        ".....OBO....",
        "....OBBBO...",
        "....OOOOO...",
    ],
    "drumstick",
)

FIRE = check(
    [
        "............",
        "......y.....",
        ".....yY.....",
        "....yYY..y..",
        "...yYWY.yY..",
        "...YWWWYYY..",
        "..yYWWWWWYy.",
        "..YYWWWWWYY.",
        "...rYYYYYr..",
        ".TTtrrrrrtTT",
        "..tTTttTTt..",
        "............",
    ],
    "fire",
)


# --- sheet -----------------------------------------------------------------


def build_sheet() -> list[list[Sprite]]:
    terrain = [
        ui(),
        mask(),
        floor(),
        wall(),
        dirt(),
        water(),
        pipe_horizontal(),
        pipe_vertical(),
        decor(DECOR_GRASS, {"g": GRASS_DIM, "l": GRASS_DIM_LIGHT}),
        decor(DECOR_PEBBLES, {"l": hex_color("4a5060"), "s": hex_color("323641")}),
        decor(DECOR_CRACKS, {"c": hex_color("0f1116")}),
        decor(DECOR_MOSS, {"g": GRASS_DIM, "m": GRASS_DIM_LIGHT}),
    ]
    items = [
        stamp(floor(), CHEST_LOCKED, CHEST_PALETTE),
        stamp(floor(), CHEST_OPEN, CHEST_PALETTE),
        stamp(floor(), APPLE, APPLE_PALETTE),
        stamp(
            floor(),
            BOLT,
            {"O": OUTLINE, "Y": GOLD, "W": SPARKLE},
        ),
        banner(TEAM_NEUTRAL),
        banner(TEAM_RED),
        banner(TEAM_BLUE),
        stamp(floor(), AGENT, AGENT_PALETTE),
        stamp(floor(), RABBIT, RABBIT_PALETTE),
        stamp(floor(), TURTLE, TURTLE_PALETTE),
        snake_segment(hex_color(SNAKE_COLORS["green"]), eyes=True),
        # Lush grass is terrain, but row 0 is full.
        decor(GRASS_TILE, {"g": GRASS, "l": GRASS_LIGHT}),
    ]
    snakes = [snake_segment(hex_color(c)) for c in SNAKE_COLORS.values()]

    def team_row(team: Color) -> list[Sprite]:
        knights = [
            soldier(KNIGHT, team, direction, wounded)
            for direction in range(4)
            for wounded in (True, False)
        ]
        archers = [soldier(ARCHER, team, direction, False) for direction in range(4)]
        return knights + archers

    # The digits leave two slots free, which the survival stat labels fill.
    digits = [
        *(digit(value) for value in range(10)),
        stamp(
            blank(UI),
            HEART_ICON,
            {"O": OUTLINE, "R": HEART, "r": HEART_DARK, "h": HEART_LIGHT},
        ),
        stamp(
            blank(UI),
            DRUMSTICK,
            {"O": OUTLINE, "M": MEAT, "m": MEAT_DARK, "h": MEAT_LIGHT, "B": BONE},
        ),
    ]
    pacman_row = [
        pellet(),
        power_pellet(),
        *(pacman(direction) for direction in range(4)),
        *(ghost(hex_color(c)) for c in GHOST_COLORS.values()),
        stamp(
            floor(),
            GHOST_SCARED,
            {"O": OUTLINE, "G": GHOST_FRIGHTENED, "F": GHOST_FRIGHTENED_FACE},
        ),
        stamp(floor(), GHOST_EYES, {"W": GHOST_EYE, "P": GHOST_PUPIL}),
    ]

    survival_row = [
        stamp(
            floor(),
            FIRE,
            {
                "y": FLAME_DEEP,
                "Y": FLAME,
                "W": FLAME_CORE,
                "r": EMBER,
                "T": WOOD,
                "t": WOOD_DARK,
            },
        ),
    ]

    return [
        terrain,
        items,
        snakes,
        team_row(TEAM_RED),
        team_row(TEAM_BLUE),
        digits,
        pacman_row,
        survival_row,
    ]


def rasterize(rows: list[list[Sprite]], zoom: int = 1) -> tuple[int, int, bytes]:
    stride = TILE + PAD
    width = (COLS * stride + PAD) * zoom
    height = (len(rows) * stride + PAD) * zoom
    # Transparent border and separators; the renderers never sample them.
    pixels = bytearray(width * height * 4)
    for row_index, row in enumerate(rows):
        assert len(row) <= COLS, f"row {row_index} overflows {COLS} columns"
        for col_index, sprite in enumerate(row):
            ox = col_index * stride + PAD
            oy = row_index * stride + PAD
            for y in range(TILE):
                for x in range(TILE):
                    r, g, b = sprite[y][x]
                    for zy in range(zoom):
                        for zx in range(zoom):
                            px = (ox + x) * zoom + zx
                            py = (oy + y) * zoom + zy
                            i = (py * width + px) * 4
                            pixels[i : i + 4] = bytes((r, g, b, 255))
    return width, height, bytes(pixels)


def encode_png(width: int, height: int, rgba: bytes) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(
        b"\x00" + rgba[y * width * 4 : (y + 1) * width * 4] for y in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--preview", type=Path, nargs="?", const=Path("tileset_preview.png")
    )
    args = parser.parse_args()

    sheet = build_sheet()
    png = encode_png(*rasterize(sheet))
    for path in OUTPUTS:
        path.write_bytes(png)
        print(f"wrote {path.relative_to(ROOT)} ({len(png)} bytes)")
    if args.preview:
        args.preview.write_bytes(encode_png(*rasterize(sheet, zoom=8)))
        print(f"wrote {args.preview}")


if __name__ == "__main__":
    main()
