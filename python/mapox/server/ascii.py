"""Symbol to character legend for ASCII observation rendering.

Symbols not in the table render as `?`; ids beyond the vocabulary render as a
space. The legend is a plain dict so the server can hand it to clients
verbatim: keys are symbol names, values the single characters the ASCII grids
use.
"""

import numpy as np

from mapox.symbols import (
    AGENT_ARCHER,
    AGENT_GENERIC,
    AGENT_HARVESTER,
    AGENT_KNIGHT,
    AGENT_PREY,
    AGENT_PREDATOR,
    AGENT_SCOUT,
    AGENT_SNAKE_BLUE,
    AGENT_SNAKE_GOLD,
    AGENT_SNAKE_GRAY,
    AGENT_SNAKE_GREEN,
    AGENT_SNAKE_ORANGE,
    AGENT_SNAKE_PINK,
    AGENT_SNAKE_PURPLE,
    AGENT_SNAKE_RED,
    AGENT_SNAKE_WHITE,
    AGENT_SNAKE_YELLOW,
    TILE_ARROW,
    TILE_DECOR_1,
    TILE_DECOR_2,
    TILE_DECOR_3,
    TILE_DECOR_4,
    TILE_DESTRUCTIBLE_WALL,
    TILE_EMPTY,
    TILE_FLAG,
    TILE_FLAG_UNLOCKED,
    TILE_FOOD,
    TILE_GRASS,
    TILE_MASK,
    TILE_WALL,
)

UNKNOWN_CHAR = "?"

LEGEND: dict[str, str] = {
    # unseen tiles and the synthetic UI band render blank, like unexplored
    # areas in classic ASCII roguelikes; only genuinely unknown symbols get ?
    "ui": " ",
    TILE_MASK: " ",
    TILE_EMPTY: ".",
    TILE_WALL: "#",
    TILE_DESTRUCTIBLE_WALL: "+",
    TILE_FLAG: "f",
    TILE_FLAG_UNLOCKED: "F",
    TILE_ARROW: "*",
    TILE_GRASS: ",",
    TILE_FOOD: "%",
    # rust-only symbols, absent from mapox.symbols
    "tile/water": "~",
    "tile/pipe_horizontal": "=",
    "tile/pipe_virtical": "|",
    TILE_DECOR_1: "'",
    TILE_DECOR_2: '"',
    TILE_DECOR_3: "`",
    TILE_DECOR_4: ":",
    AGENT_GENERIC: "a",
    AGENT_SCOUT: "s",
    AGENT_HARVESTER: "h",
    AGENT_PREY: "p",
    AGENT_PREDATOR: "P",
    AGENT_KNIGHT: "k",
    AGENT_ARCHER: "r",
    AGENT_SNAKE_RED: "r",
    AGENT_SNAKE_ORANGE: "o",
    AGENT_SNAKE_YELLOW: "y",
    AGENT_SNAKE_GOLD: "g",
    AGENT_SNAKE_GREEN: "n",
    AGENT_SNAKE_BLUE: "b",
    AGENT_SNAKE_PURPLE: "u",
    AGENT_SNAKE_PINK: "p",
    AGENT_SNAKE_GRAY: "e",
    AGENT_SNAKE_WHITE: "w",
}


def build_char_table(vocab_symbols: list[str] | tuple[str, ...]) -> list[str]:
    """Per-id characters for a vocabulary: index by obs id to get the char."""

    return [LEGEND.get(symbol, UNKNOWN_CHAR) for symbol in vocab_symbols]


def render_grid(obs: np.ndarray, char_table: list[str]) -> list[str]:
    """Rows of characters for one agent's observation crop, top row first.

    `obs` is the (view_width, view_height) id grid for a single agent, stored
    column-major (map x, then map y) with y index 0 the lowest map row —
    transposing and reversing yields screen rows, high map y first, matching
    the renderer. The UI band (obs y above fov_height), where the env has
    one, lands at the top. Ids past the vocabulary render as a space.
    """

    return [
        "".join(
            char_table[tile] if tile < len(char_table) else " "
            for tile in col
        )
        for col in obs.T[::-1]
    ]
