// the empty area of a ux tile grid
pub const TILE_UI: &str = "ui";

/// Digit glyphs for numbers in the UI band, shared by every env so a count
/// reads the same everywhere. Indexed by the digit they draw; a number is a
/// run of these, most significant digit leftmost.
pub const UI_DIGITS: [&str; 10] = [
    "ui/digit_0",
    "ui/digit_1",
    "ui/digit_2",
    "ui/digit_3",
    "ui/digit_4",
    "ui/digit_5",
    "ui/digit_6",
    "ui/digit_7",
    "ui/digit_8",
    "ui/digit_9",
];

/// Survival stat labels, each drawn in front of the number it labels.
pub const UI_HEALTH: &str = "ui/health";
pub const UI_HUNGER: &str = "ui/hunger";

pub const TILE_MASK: &str = "mask";
pub const TILE_EMPTY: &str = "tile/empty";
pub const TILE_WALL: &str = "tile/wall";
pub const TILE_DESTRUCTIBLE_WALL: &str = "tile/destructible_wall";
/// A wall seamed with iron, dug out like a [`TILE_DESTRUCTIBLE_WALL`] but
/// sometimes leaving an [`ITEM_IRON_ORE`].
pub const TILE_IRON_ORE: &str = "tile/iron_ore";
/// A flag no one can score off yet. [`TILE_FLAG_UNLOCKED`] is the same flag
/// once it is claimable.
pub const TILE_FLAG: &str = "tile/flag";
pub const TILE_FLAG_UNLOCKED: &str = "tile/flag_unlocked";
pub const TILE_ARROW: &str = "tile/arrow";
pub const TILE_GRASS: &str = "tile/grass";
pub const TILE_WATER: &str = "tile/water";
pub const TILE_PELLET: &str = "tile/pellet";
pub const TILE_POWER_PELLET: &str = "tile/power_pellet";
pub const TILE_TREE: &str = "tile/tree";
/// A tree with apples on it. Picked bare it is a plain [`TILE_TREE`].
pub const TILE_APPLE_TREE: &str = "tile/apple_tree";
/// A bush with ripe berries on it. [`TILE_BUSH`] is the same bush picked bare.
pub const TILE_BERRY_BUSH: &str = "tile/berry_bush";
pub const TILE_BUSH: &str = "tile/bush";
/// Brambles: walked through, but they hurt whoever stands in them.
pub const TILE_THORN_BUSH: &str = "tile/thorn_bush";
/// Tall grass: walked through, and taken from for an [`ITEM_CUT_GRASS`],
/// which leaves it [`TILE_CUT_GRASS`] until it grows back.
pub const TILE_TALL_GRASS: &str = "tile/tall_grass";
pub const TILE_CUT_GRASS: &str = "tile/cut_grass";
pub const TILE_FIRE: &str = "tile/fire";
/// A [`TILE_FIRE`] that has burnt through its fuel.
pub const TILE_FIRE_OUT: &str = "tile/fire_out";
/// Where survivors come into the world, and come back to after dying.
pub const TILE_PORTAL: &str = "tile/portal";

pub const TILE_PIPE_HORIZONTAL: &str = "tile/pipe_horizontal";
pub const TILE_PIPE_VIRTICAL: &str = "tile/pipe_virtical";

pub const TILE_DECOR_1: &str = "tile/decor_1";
pub const TILE_DECOR_2: &str = "tile/decor_2";
pub const TILE_DECOR_3: &str = "tile/decor_3";
pub const TILE_DECOR_4: &str = "tile/decor_4";

// Items: things that lie on the ground or sit in an inventory slot, drawn the
// same in both places.
pub const ITEM_ROCK: &str = "item/rock";
pub const ITEM_STICK: &str = "item/stick";
pub const ITEM_WOOD: &str = "item/wood";
/// A bundle of cut grass.
pub const ITEM_CUT_GRASS: &str = "item/cut_grass";
pub const ITEM_IRON_ORE: &str = "item/iron_ore";
pub const ITEM_STONE_AXE: &str = "item/stone_axe";
/// Digs out a [`TILE_DESTRUCTIBLE_WALL`] or a [`TILE_IRON_ORE`].
pub const ITEM_STONE_PICKAXE: &str = "item/stone_pickaxe";
/// An unlit campfire, carried; using it sets down a [`TILE_FIRE`].
pub const ITEM_CAMPFIRE_KIT: &str = "item/campfire_kit";
pub const ITEM_BERRY: &str = "item/berry";
pub const ITEM_COOKED_BERRY: &str = "item/cooked_berry";
pub const ITEM_APPLE: &str = "item/apple";

pub const AGENT_GENERIC: &str = "agent/generic";
/// The generic agent by facing, in `MOVES` order, for envs where what an
/// agent acts on is the tile in front of it.
pub const AGENT_GENERIC_UP: &str = "agent/generic_up";
pub const AGENT_GENERIC_RIGHT: &str = "agent/generic_right";
pub const AGENT_GENERIC_DOWN: &str = "agent/generic_down";
pub const AGENT_GENERIC_LEFT: &str = "agent/generic_left";
pub const AGENT_SCOUT: &str = "agent/scout";
pub const AGENT_HARVESTER: &str = "agent/harvester";
pub const AGENT_PREY: &str = "agent/prey";
pub const AGENT_PREDATOR: &str = "agent/predator";
pub const AGENT_KNIGHT: &str = "agent/knight";
pub const AGENT_ARCHER: &str = "agent/archer";
/// Pac-Man by heading, in `MOVES` order: the heading is state (a move into a
/// wall keeps it going that way), so the tile carries it.
pub const AGENT_PACMAN_UP: &str = "agent/pacman_up";
pub const AGENT_PACMAN_RIGHT: &str = "agent/pacman_right";
pub const AGENT_PACMAN_DOWN: &str = "agent/pacman_down";
pub const AGENT_PACMAN_LEFT: &str = "agent/pacman_left";
pub const AGENT_GHOST_BLINKY: &str = "agent/ghost_blinky";
pub const AGENT_GHOST_PINKY: &str = "agent/ghost_pinky";
pub const AGENT_GHOST_INKY: &str = "agent/ghost_inky";
pub const AGENT_GHOST_CLYDE: &str = "agent/ghost_clyde";
/// Any ghost while a power pellet lasts: edible, and every one looks alike.
pub const AGENT_GHOST_FRIGHTENED: &str = "agent/ghost_frightened";
/// An eaten ghost heading home: harmless, and not edible again until it's back.
pub const AGENT_GHOST_EYES: &str = "agent/ghost_eyes";

// --- action symbols ---

pub const MOVE_UP: &str = "move/up";
pub const MOVE_RIGHT: &str = "move/right";
pub const MOVE_DOWN: &str = "move/down";
pub const MOVE_LEFT: &str = "move/left";

pub const MOVES: [&str; 4] = [MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT];

pub const NOOP: &str = "noop";
pub const PRIMARY_ACTION: &str = "primary";
pub const DIG_ACTION: &str = "dig";

pub const PLACE_PIPE: &str = "place_pipe";

pub const TAKE: &str = "item/take";
pub const PUT: &str = "item/put";
pub const SWAP1: &str = "item/swap/1";
pub const SWAP2: &str = "item/swap/2";
pub const SWAP3: &str = "item/swap/3";
pub const SWAP4: &str = "item/swap/4";

pub const COMBINE: &str = "item/combine";
pub const USE: &str = "item/use";
pub const REFINE: &str = "item/refine";
