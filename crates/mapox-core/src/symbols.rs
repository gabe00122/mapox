// the empty area of a ux tile grid
pub const TILE_UI: &str = "ui";

pub const TILE_MASK: &str = "mask";
pub const TILE_EMPTY: &str = "tile/empty";
pub const TILE_WALL: &str = "tile/wall";
pub const TILE_DESTRUCTIBLE_WALL: &str = "tile/destructible_wall";
/// A flag no one can score off yet. [`TILE_FLAG_UNLOCKED`] is the same flag
/// once it is claimable.
pub const TILE_FLAG: &str = "tile/flag";
pub const TILE_FLAG_UNLOCKED: &str = "tile/flag_unlocked";
pub const TILE_ARROW: &str = "tile/arrow";
pub const TILE_GRASS: &str = "tile/grass";
pub const TILE_WATER: &str = "tile/water";
pub const TILE_FOOD: &str = "tile/food";

pub const TILE_PIPE_HORIZONTAL: &str = "tile/pipe_horizontal";
pub const TILE_PIPE_VIRTICAL: &str = "tile/pipe_virtical";

pub const TILE_DECOR_1: &str = "tile/decor_1";
pub const TILE_DECOR_2: &str = "tile/decor_2";
pub const TILE_DECOR_3: &str = "tile/decor_3";
pub const TILE_DECOR_4: &str = "tile/decor_4";

pub const AGENT_GENERIC: &str = "agent/generic";
pub const AGENT_SCOUT: &str = "agent/scout";
pub const AGENT_HARVESTER: &str = "agent/harvester";
pub const AGENT_PREY: &str = "agent/prey";
pub const AGENT_PREDATOR: &str = "agent/predator";
pub const AGENT_KNIGHT: &str = "agent/knight";
pub const AGENT_ARCHER: &str = "agent/archer";
pub const AGENT_SNAKE_RED: &str = "agent/snake_red";
pub const AGENT_SNAKE_ORANGE: &str = "agent/snake_orange";
pub const AGENT_SNAKE_YELLOW: &str = "agent/snake_yellow";
pub const AGENT_SNAKE_GOLD: &str = "agent/snake_gold";
pub const AGENT_SNAKE_GREEN: &str = "agent/snake_green";
pub const AGENT_SNAKE_BLUE: &str = "agent/snake_blue";
pub const AGENT_SNAKE_PURPLE: &str = "agent/snake_purple";
pub const AGENT_SNAKE_PINK: &str = "agent/snake_pink";
pub const AGENT_SNAKE_GRAY: &str = "agent/snake_gray";
pub const AGENT_SNAKE_WHITE: &str = "agent/snake_white";

// --- action symbols ---

pub const MOVE_UP: &str = "move/up";
pub const MOVE_RIGHT: &str = "move/right";
pub const MOVE_DOWN: &str = "move/down";
pub const MOVE_LEFT: &str = "move/left";

pub const MOVES: [&str; 4] = [MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT];

pub const STAY: &str = "stay";
pub const NOOP: &str = "noop";
pub const PRIMARY_ACTION: &str = "primary";
pub const DIG_ACTION: &str = "dig";

pub const PLACE_PIPE: &str = "place_pipe";
