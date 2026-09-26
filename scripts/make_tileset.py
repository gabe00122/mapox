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
COLS = 16

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


def dirt() -> Sprite:
    """Diggable earth: warm, crumbly and structureless, unlike cut stone."""
    rng = random.Random(11)
    sprite = blank(DIRT)
    for y in range(TILE):
        for x in range(TILE):
            roll = rng.random()
            if roll < 0.18:
                sprite[y][x] = DIRT_DARK
            elif roll < 0.30:
                sprite[y][x] = DIRT_LIGHT
    for px, py in [(2, 2), (8, 4), (4, 8), (10, 10)]:
        sprite[py][px] = PEBBLE
        sprite[py][px + 1] = DIRT_LIGHT
        sprite[py + 1][px] = DIRT_DARK
    return sprite


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
        "............",
        "..OOOOOOOO..",
        ".OWWWIIWWWO.",
        ".OwwwIIwwwO.",
        ".OOOOLLOOOO.",
        ".OWWWLkWWWO.",
        ".OWIWLLWIWO.",
        ".OwIwwwwIwO.",
        ".OOOOOOOOOO.",
        "............",
    ],
    "chest locked",
)

CHEST_OPEN = check(
    [
        "..........*.",
        "..OOOOOOOO..",
        ".OwwwwwwwwO.",
        ".OWggGGggWO.",
        ".OgGGGGGGgO*",
        ".OGGhGGhGGO.",
        ".OOOOOOOOOO.",
        ".OWWWIIWWWO.",
        ".OWIWIIWIWO.",
        ".OwIwwwwIwO.",
        ".OOOOOOOOOO.",
        "............",
    ],
    "chest open",
)

CHEST_PALETTE = {
    "O": OUTLINE,
    "W": WOOD,
    "w": WOOD_DARK,
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
        "............",
        "......l.....",
        ".....sl.....",
        "...OOsOOO...",
        "..ORwRRRRO..",
        "..ORRRRRRO..",
        "..ORRRRRRO..",
        "..OrRRRRrO..",
        "...OrrrrO...",
        "....OOOO....",
        "............",
    ],
    "apple",
)

APPLE_PALETTE = {
    "O": hex_color("3a0d12"),
    "R": hex_color("ff4d4d"),
    "r": hex_color("b3202b"),
    "w": hex_color("ffd6d6"),
    "l": GRASS_LIGHT,
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

MINER = check(
    [
        "....OOOO....",
        "...OYYYYO...",
        "..OYYLLYYO..",
        "..OyyyyyyO..",
        "..OSESSESO..",
        "..OsSSSSsO..",
        "..OBBbbBBO..",
        ".OSBBBBBBSO.",
        ".OSBBBBBBSO.",
        "..ObbbbbbO..",
        "..OKKO.OKKO.",
        "..OOOO.OOOO.",
    ],
    "miner",
)

MINER_PALETTE = {
    "O": OUTLINE,
    "Y": hex_color("f4c430"),
    "y": hex_color("b8891c"),
    "L": hex_color("fffbd0"),
    "S": SKIN,
    "s": SKIN_SHADE,
    "E": EYE,
    "B": hex_color("4a86e8"),
    "b": hex_color("2e5cb3"),
    "K": BOOT,
}

SCOUT = check(
    [
        ".....OO.....",
        "....OHHO....",
        "...OHHHHO...",
        "...OSESEO...",
        "...OSSSSO...",
        "..OHHHHHHO..",
        ".OSHhHHhHSO.",
        "..OHhHHhHO..",
        "..OHHHHHHO..",
        "...OLOOLO...",
        "..OLO..OLO..",
        "..OO....OO..",
    ],
    "scout",
)

SCOUT_PALETTE = {
    "O": OUTLINE,
    "H": hex_color("3fd6c2"),
    "h": hex_color("218f82"),
    "S": SKIN,
    "E": EYE,
    "L": hex_color("2b3040"),
}

HARVESTER = check(
    [
        "...OOOOOO...",
        "..OMMMMMMO..",
        ".OMMMMMMMmO.",
        ".OSSESSESSO.",
        ".OSSSSSSSSO.",
        "OORRRRRRRROO",
        "ORRRRRRRRRRO",
        "OSRRRrrRRRSO",
        "OSRRRRRRRRSO",
        ".OrrrrrrrrO.",
        ".OKKKO.OKKKO",
        ".OOOOO.OOOOO",
    ],
    "harvester",
)

HARVESTER_PALETTE = {
    "O": OUTLINE,
    "M": hex_color("a3acbb"),
    "m": hex_color("6a7282"),
    "S": SKIN,
    "E": EYE,
    "R": hex_color("e8642e"),
    "r": hex_color("9c3a17"),
    "K": BOOT,
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
        "...OMMMMO...",
        "...OMkkMO...",
        "..OOMMMMOO..",
        ".OMTTTTTTMO.",
        ".OMTTWWTTMO.",
        "..OTTWWTTO..",
        "..OttttttO..",
        "...OKO.OKO..",
        "............",
    ],
    "knight",
)

ARCHER = check(
    [
        "............",
        "............",
        ".....OO.....",
        "....OTTO....",
        "...OTSSTO...",
        "...OSESEO...",
        "..OTTTTTTOb.",
        "..OTTttTTOb.",
        "..OTTttTTOb.",
        "...OTTTTO...",
        "...OKO.OKO..",
        "............",
    ],
    "archer",
)


def soldier(art: list[str], team: Color, direction: int, wounded: bool) -> Sprite:
    palette = {
        "O": OUTLINE,
        "M": IRON_LIGHT,
        "k": OUTLINE,
        "T": team,
        "t": scale(team, 0.65),
        "W": hex_color("f2f2ee"),
        "S": SKIN,
        "E": EYE,
        "K": BOOT,
        "b": hex_color("a0703c"),
    }
    if wounded:
        palette = {k: scale(v, 0.6) for k, v in palette.items()}
    sprite = stamp(floor(), art, palette)
    return facing_pip(sprite, direction, GOLD_LIGHT)


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
        decor(GRASS_TILE, {"g": GRASS, "l": GRASS_LIGHT}),
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
        stamp(floor(), MINER, MINER_PALETTE),
        stamp(floor(), SCOUT, SCOUT_PALETTE),
        stamp(floor(), HARVESTER, HARVESTER_PALETTE),
        snake_segment(hex_color(SNAKE_COLORS["green"]), eyes=True),
        snake_segment(hex_color(SNAKE_COLORS["green"])),
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

    return [terrain, items, snakes, team_row(TEAM_RED), team_row(TEAM_BLUE)]


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
