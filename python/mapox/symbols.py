"""Symbol names for observation and action vocabularies.

These strings are the stable identity behind every integer id: they outlive
checkpoints, map files, and env reorderings. Add freely; rename never once a
symbol has been persisted anywhere.

Names use a `/` namespace (`tile/wall`, `agent/scout`, `move/up`) and `_`
within words. A symbol shared between envs means shared semantics: two envs
that both register `tile/wall` are declaring that walls transfer.

This module is open-ended — an env may mint symbols not listed here — but
anything meant to be shared across envs should live here so a typo can't
silently split one concept into two ids.
"""

# --- observation symbols: tiles ---

TILE_MASK = "mask"  # a tile the agent cannot see, e.g. hidden behind a wall
TILE_EMPTY = "tile/empty"
TILE_WALL = "tile/wall"
TILE_DESTRUCTIBLE_WALL = "tile/destructible_wall"
TILE_FLAG = "tile/flag"
TILE_FLAG_UNLOCKED = "tile/flag_unlocked"  # scouts: flag made available for taking
TILE_ARROW = "tile/arrow"  # king_hill: projectile in flight
TILE_GRASS = "tile/grass"
TILE_FOOD = "tile/food"
TILE_POWER = "tile/power"  # pacman: the pellet that frightens the ghosts

TILE_DECOR_1 = "tile/decor_1"
TILE_DECOR_2 = "tile/decor_2"
TILE_DECOR_3 = "tile/decor_3"
TILE_DECOR_4 = "tile/decor_4"

# Cosmetic empty-tile variants; register together (order is only cosmetic
# but add_block keeps them contiguous for map_generator's offset math).
TILE_DECOR = (TILE_DECOR_1, TILE_DECOR_2, TILE_DECOR_3, TILE_DECOR_4)

# --- observation symbols: agents (observed like tiles) ---

AGENT_GENERIC = "agent/generic"
AGENT_SCOUT = "agent/scout"
AGENT_HARVESTER = "agent/harvester"
AGENT_PREY = "agent/prey"  # prey env: sneakers
AGENT_PREDATOR = "agent/predator"  # prey env: chasers
AGENT_KNIGHT = "agent/knight"
AGENT_ARCHER = "agent/archer"
AGENT_SNAKE_HEAD = "agent/snake_head"
AGENT_SNAKE_BODY = "agent/snake_body"

# the rust snake paints bodies in ten colours, agent index modulo ten
AGENT_SNAKE_RED = "agent/snake_red"
AGENT_SNAKE_ORANGE = "agent/snake_orange"
AGENT_SNAKE_YELLOW = "agent/snake_yellow"
AGENT_SNAKE_GOLD = "agent/snake_gold"
AGENT_SNAKE_GREEN = "agent/snake_green"
AGENT_SNAKE_BLUE = "agent/snake_blue"
AGENT_SNAKE_PURPLE = "agent/snake_purple"
AGENT_SNAKE_PINK = "agent/snake_pink"
AGENT_SNAKE_GRAY = "agent/snake_gray"
AGENT_SNAKE_WHITE = "agent/snake_white"

AGENT_PACMAN = "agent/pacman"
AGENT_GHOST_PINKY = "agent/ghost_pinky"
AGENT_GHOST_BLINKY = "agent/ghost_blinky"
AGENT_GHOST_INKY = "agent/ghost_inky"
AGENT_GHOST_CLYDE = "agent/ghost_clyde"
AGENT_GHOST_FRIGHTENED = "agent/ghost_frightened"
AGENT_GHOST_EYES = "agent/ghost_eyes"

# --- action symbols ---

MOVE_UP = "move/up"
MOVE_RIGHT = "move/right"
MOVE_DOWN = "move/down"
MOVE_LEFT = "move/left"

# Order matches common.DIRECTIONS rows. Envs that index DIRECTIONS by action
# id must register this via add_block so the moves land at local ids 0..3.
MOVES = (MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT)

STAY = "stay"
PRIMARY_ACTION = "primary"  # context-dependent "use" button (attack in king_hill)
DIG_ACTION = "dig"  # break a destructible wall
