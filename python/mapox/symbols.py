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
TILE_PELLET = "tile/pellet"  # pacman
TILE_POWER_PELLET = "tile/power_pellet"  # pacman: frightens the ghosts

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
# the rust pacman: Pac-Man by heading (MOVES order), then the ghosts
AGENT_PACMAN_UP = "agent/pacman_up"
AGENT_PACMAN_RIGHT = "agent/pacman_right"
AGENT_PACMAN_DOWN = "agent/pacman_down"
AGENT_PACMAN_LEFT = "agent/pacman_left"
AGENT_GHOST_BLINKY = "agent/ghost_blinky"
AGENT_GHOST_PINKY = "agent/ghost_pinky"
AGENT_GHOST_INKY = "agent/ghost_inky"
AGENT_GHOST_CLYDE = "agent/ghost_clyde"
AGENT_GHOST_FRIGHTENED = "agent/ghost_frightened"
AGENT_GHOST_EYES = "agent/ghost_eyes"

# the rust mutiny: the crew by color, which is all that tells them apart
CREW_COLORS = (
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "pink",
    "white",
    "brown",
)
AGENT_CREW = tuple(f"agent/crew_{color}" for color in CREW_COLORS)
AGENT_CREW_GHOST = "agent/crew_ghost"  # a dead crew member, seen only by the dead
TILE_CORPSES = tuple(f"tile/corpse_{color}" for color in CREW_COLORS)
TILE_VOTE_BUTTONS = tuple(f"tile/vote_button_{color}" for color in CREW_COLORS)
TILE_CHORE_BUTTON = "tile/chore_button"
TILE_CHORE_BUTTON_LIT = "tile/chore_button_lit"  # the viewer's own next button
TILE_CHORE_BUTTON_PRESSED = "tile/chore_button_pressed"
TILE_MEETING_BUTTON = "tile/meeting_button"  # calls a vote
TILE_MEETING_BUTTON_OFF = "tile/meeting_button_off"  # the viewer cannot call one
TILE_HIDING_SPOT = "tile/hiding_spot"
TILE_HIDING_SPOT_OCCUPIED = "tile/hiding_spot_occupied"
TILE_SKIP_BUTTON = "tile/skip_button"  # votes to kick no one

# --- observation symbols: UI band ---

# Shared digit glyphs for numbers in the rust envs' UI band, indexed by the
# digit they draw; a number is a run of them, most significant digit first.
UI_DIGITS = tuple(f"ui/digit_{d}" for d in range(10))

# mutiny's band labels: the viewer's role, then what each number counts
UI_ROLE_CREW = "ui/role_crew"
UI_ROLE_TRAITOR = "ui/role_traitor"
UI_DEAD_CREW = "ui/dead_crew"
UI_DEAD_TRAITORS = "ui/dead_traitors"
UI_CHORES = "ui/chores"
UI_ATTACK = "ui/attack"
UI_VOTE = "ui/vote"

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
