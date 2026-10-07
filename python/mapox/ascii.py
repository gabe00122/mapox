"""ASCII rendering of agent observations for terminals and text-only clients.

The graphical renderer draws an agent's crop with the tileset; this is its
text counterpart. Rows read top down with map ``x`` running left to right,
exactly as the renderer paints them, so the UI band the Rust envs append to
the observation lands at the top of the grid as a rule of ``-`` characters.

Symbols outside [`LEGEND`] render as ``?`` and ids past the vocabulary render
as a space, so an old legend can still draw an observation from a growing
vocabulary.
"""

import numpy as np
from numpy.typing import ArrayLike

import mapox.symbols as SB
from mapox.vocab import Vocabulary

UNKNOWN_CHAR = "?"

LEGEND: dict[str, str] = {
    # unseen tiles render blank, like unexplored areas in classic ASCII
    # roguelikes; only genuinely unknown symbols get ?. The synthetic UI band
    # is a separate rule so it cannot be mistaken for fog of war.
    "ui": "-",
    SB.TILE_MASK: " ",
    SB.TILE_EMPTY: ".",
    SB.TILE_WALL: "#",
    SB.TILE_DESTRUCTIBLE_WALL: "+",
    SB.TILE_FLAG: "f",
    SB.TILE_FLAG_UNLOCKED: "F",
    SB.TILE_ARROW: "*",
    SB.TILE_GRASS: ",",
    SB.TILE_FOOD: "%",
    # rust-only symbols, absent from mapox.symbols
    "tile/water": "~",
    "tile/pipe_horizontal": "=",
    "tile/pipe_virtical": "|",
    SB.TILE_DECOR_1: "'",
    SB.TILE_DECOR_2: '"',
    SB.TILE_DECOR_3: "`",
    SB.TILE_DECOR_4: ":",
    SB.AGENT_GENERIC: "a",
    SB.AGENT_SCOUT: "s",
    SB.AGENT_HARVESTER: "h",
    SB.AGENT_PREY: "p",
    SB.AGENT_PREDATOR: "P",
    SB.AGENT_KNIGHT: "k",
    SB.AGENT_ARCHER: "r",
    SB.AGENT_SNAKE_HEAD: "O",
    SB.AGENT_SNAKE_BODY: "o",
    SB.AGENT_SNAKE_RED: "r",
    SB.AGENT_SNAKE_ORANGE: "o",
    SB.AGENT_SNAKE_YELLOW: "y",
    SB.AGENT_SNAKE_GOLD: "g",
    SB.AGENT_SNAKE_GREEN: "n",
    SB.AGENT_SNAKE_BLUE: "b",
    SB.AGENT_SNAKE_PURPLE: "u",
    SB.AGENT_SNAKE_PINK: "p",
    SB.AGENT_SNAKE_GRAY: "e",
    SB.AGENT_SNAKE_WHITE: "w",
    SB.TILE_PELLET: "o",
    SB.TILE_POWER_PELLET: "@",
    SB.AGENT_PACMAN_UP: "^",
    SB.AGENT_PACMAN_RIGHT: ">",
    SB.AGENT_PACMAN_DOWN: "v",
    SB.AGENT_PACMAN_LEFT: "<",
    SB.AGENT_GHOST_BLINKY: "B",
    SB.AGENT_GHOST_PINKY: "P",
    SB.AGENT_GHOST_INKY: "I",
    SB.AGENT_GHOST_CLYDE: "C",
    SB.AGENT_GHOST_FRIGHTENED: "S",
    SB.AGENT_GHOST_EYES: "E",
    # the rust survival env, absent from mapox.symbols
    "agent/generic_up": "^",
    "agent/generic_right": ">",
    "agent/generic_down": "v",
    "agent/generic_left": "<",
    "tile/tree": "T",
    "tile/berry_bush": "B",
    "tile/bush": "b",
    "tile/tall_grass": ";",
    "tile/buried_carrot": "q",
    "tile/ice": "_",
    "tile/dead_bush": "d",
    "tile/fire": "&",
    "tile/fire_low": "!",
    "tile/spider_eggs": "e",
    "agent/spider": "S",
    "item/stick": "/",
    "item/stone": "o",
    "item/wood": "=",
    "item/berry": "*",
    "item/cooked_berry": "@",
    "item/carrot": "c",
    "item/cooked_carrot": "C",
    "item/grass": "l",
    "item/axe": "x",
    "item/campfire": "m",
    "item/torch": "i",
    "ui/health": "H",
    "ui/hunger": "F",
    "ui/hands": "h",
    "ui/backpack": "k",
    "ui/day": "D",
    "ui/night": "N",
    "ui/temperature": "t",
    "ui/winter": "W",
    # UI band numbers draw as themselves
    **{symbol: str(d) for d, symbol in enumerate(SB.UI_DIGITS)},
}


def build_char_table(vocab_symbols: list[str] | tuple[str, ...]) -> list[str]:
    """Per-id characters for a vocabulary: index by obs id to get the char."""

    return [LEGEND.get(symbol, UNKNOWN_CHAR) for symbol in vocab_symbols]


def render_grid(obs: ArrayLike, char_table: list[str]) -> list[str]:
    """Rows of characters for one agent's observation crop, top row first.

    `obs` is the ``(view_width, view_height)`` id grid for a single agent,
    stored column-major (map x, then map y) with y index 0 the lowest map row
    — transposing and reversing yields screen rows, high map y first,
    matching the renderer. The UI band the Rust envs append (obs y at or
    above the FOV height), where the env has one, lands at the top. Ids past
    the vocabulary render as a space.
    """

    return [
        "".join(
            char_table[tile] if 0 <= tile < len(char_table) else " " for tile in column
        )
        for column in np.asarray(obs).T[::-1]
    ]


def _tile_channel(obs: ArrayLike) -> np.ndarray:
    array = np.asarray(obs)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(
            "observation must be (view_width, view_height) or "
            f"(view_width, view_height, channels), got shape {array.shape}"
        )

    return array


class AsciiRenderer:
    """Draws one agent's observation crop and resolves its legal actions.

    Built once per vocabulary pair: the observation vocab names the tile ids
    in ``timestep.obs`` and the action vocab names the columns of
    ``timestep.action_mask``. Both the Python envs' 4-channel observations
    (only the tile channel is drawn) and the Rust envs' single-channel ones
    are accepted. A wrapper that remaps ids must be built with its own vocab,
    the one it reports as ``obs_vocab``.

        renderer = AsciiRenderer(env.obs_vocab, env.action_vocab)
        print(renderer.render(timestep.obs[agent_id]))
        print(renderer.available_actions(timestep.action_mask[agent_id]))
    """

    def __init__(self, obs_vocab: Vocabulary, action_vocab: Vocabulary):
        self._obs_vocab = obs_vocab
        self._action_vocab = action_vocab
        self._char_table = build_char_table(obs_vocab.symbols)

    @property
    def obs_vocab(self) -> Vocabulary:
        return self._obs_vocab

    @property
    def action_vocab(self) -> Vocabulary:
        return self._action_vocab

    @property
    def char_table(self) -> tuple[str, ...]:
        """One character per observation id, in vocab order."""

        return tuple(self._char_table)

    def render_grid(self, obs: ArrayLike) -> list[str]:
        """Rows of characters for one agent's observation, top row first.

        `obs` is either the ``(view_width, view_height)`` tile-id grid or the
        full ``(view_width, view_height, channels)`` observation. The UI band
        is included: where the env encodes one it is the ``-`` rows at the
        top of the grid.
        """

        return render_grid(_tile_channel(obs), self._char_table)

    def render(self, obs: ArrayLike) -> str:
        """The ASCII grid as one string, rows separated by newlines."""

        return "\n".join(self.render_grid(obs))

    def available_actions(self, action_mask: ArrayLike) -> list[str]:
        """Action symbol names whose mask entry is set, in vocab order."""

        mask = np.asarray(action_mask, dtype=bool)
        shape = (len(self._action_vocab),)
        if mask.shape != shape:
            raise ValueError(
                "action mask must have shape "
                f"{shape} for this action vocab, got {mask.shape}"
            )

        return [self._action_vocab.symbols[i] for i in np.flatnonzero(mask)]

    def print_legend(self) -> list[str]:
        out = []
        for symbol in self._obs_vocab.symbols:
            char = LEGEND[symbol]
            out.append(f"{char} = {symbol}")
        return out
