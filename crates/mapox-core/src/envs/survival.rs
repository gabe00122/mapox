use std::collections::VecDeque;

use ndarray::{Array2, s};
use rand::{
    RngExt, SeedableRng,
    rngs::SmallRng,
    seq::{IndexedRandom, SliceRandom},
};
use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::common::{
        Position, UI_HEIGHT, fov,
        map_gen::{connect_regions, noise_field, quantile, roll, spread_out},
        ui::write_number,
        vocab_enum::VocabEnum,
    },
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_GENERIC_DOWN, AGENT_GENERIC_LEFT, AGENT_GENERIC_RIGHT, AGENT_GENERIC_UP,
        AGENT_SPIDER, COMBINE_ACTION, DROP_ACTION, EAT_ACTION, GRAB_ACTION, ITEM_AXE, ITEM_BERRY,
        ITEM_CAMPFIRE, ITEM_COOKED_BERRY, ITEM_STICK, ITEM_STONE, ITEM_WOOD, MOVE_DOWN, MOVE_LEFT,
        MOVE_RIGHT, MOVE_UP, NOOP, SWAP_ACTION, TILE_BERRY_BUSH, TILE_BUSH, TILE_DECOR_1,
        TILE_DECOR_2, TILE_DECOR_3, TILE_DECOR_4, TILE_DESTRUCTIBLE_WALL, TILE_EMPTY, TILE_FIRE,
        TILE_FIRE_LOW, TILE_MASK, TILE_SPIDER_EGGS, TILE_TREE, TILE_UI, TILE_WALL, TILE_WATER,
        UI_BACKPACK, UI_DAY, UI_DIGITS, UI_HANDS, UI_HEALTH, UI_HUNGER, UI_NIGHT, USE_ACTION,
    },
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
    vocab_enum,
};

/// Every stat runs from zero to this.
pub const MAX_STAT: u16 = 150;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SurvivalConfig {
    pub num_agents: usize,

    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,

    /// Shares of the map under water and rock: the lowest and the highest
    /// ground of an elevation field.
    pub water_fraction: f64,
    pub rock_fraction: f64,
    /// Shares of the land between that are forest and scrub: its wettest and
    /// its driest ground by a moisture field. The rest is meadow. See
    /// [`Biome::growth`] for what each grows.
    pub forest_fraction: f64,
    pub scrub_fraction: f64,

    pub start_hunger: u16,
    pub start_health: u16,
    /// Hunger drops by one every this many steps.
    pub hunger_interval: u32,
    /// Health lost on every step spent at zero hunger.
    pub starve_damage: u16,
    /// Health grows back by one every `regen_interval` steps while hunger is
    /// at least `regen_threshold`.
    pub regen_threshold: u16,
    pub regen_interval: u32,

    /// Hunger a berry restores, raw and cooked.
    pub berry_food: u16,
    pub cooked_berry_food: u16,
    /// Steps a picked bush takes to fruit again.
    pub bush_regrow_steps: u32,
    /// Steps felling a tree takes, the use that starts it included. See
    /// [`Job`].
    pub chop_steps: u32,
    /// Steps a fire burns before it goes out.
    pub fire_burn_steps: u32,
    /// The last of those steps it burns low, when wood stokes it back up to
    /// a full `fire_burn_steps`.
    pub fire_low_steps: u32,
    /// How far a fire's light reaches, in cells.
    pub fire_light_radius: i32,

    /// Steps of day and then of night in each cycle; the episode starts at
    /// dawn.
    pub day_length: u32,
    pub night_length: u32,
    /// How far an agent sees around itself at night. Lit ground it sees as
    /// far as by day.
    pub night_vision_radius: i32,
    /// Steps at the end of each day over which sight closes in from the
    /// whole view to `night_vision_radius`, warning that night is coming.
    pub dusk_length: u32,

    /// Spider nests reset sets in the forest; each hatches a spider at every
    /// nightfall.
    pub num_spider_eggs: usize,
    /// Health a spider bite takes.
    pub spider_damage: u16,
    /// How far, walking, a spider tracks an agent.
    pub spider_hunt_radius: u32,
}

impl Default for SurvivalConfig {
    fn default() -> Self {
        Self {
            num_agents: 8,
            width: 80,
            height: 70,
            view_width: 15,
            view_height: 15,
            water_fraction: 0.10,
            rock_fraction: 0.15,
            forest_fraction: 0.35,
            scrub_fraction: 0.25,
            start_hunger: 100,
            start_health: MAX_STAT,
            hunger_interval: 2,
            starve_damage: 1,
            regen_threshold: 100,
            regen_interval: 4,
            berry_food: 20,
            cooked_berry_food: 50,
            bush_regrow_steps: 200,
            chop_steps: 6,
            fire_burn_steps: 200,
            fire_low_steps: 50,
            fire_light_radius: 4,
            day_length: 200,
            night_length: 100,
            night_vision_radius: 2,
            dusk_length: 50,
            num_spider_eggs: 4,
            spider_damage: 15,
            spider_hunt_radius: 12,
        }
    }
}

vocab_enum!(SurvivalObs {
    UI => TILE_UI,
    Mask => TILE_MASK,
    TileEmpty => TILE_EMPTY,
    TileWall => TILE_WALL,
    TileDestructibleWall => TILE_DESTRUCTIBLE_WALL,
    TileWater => TILE_WATER,
    TileDecor1 => TILE_DECOR_1,
    TileDecor2 => TILE_DECOR_2,
    TileDecor3 => TILE_DECOR_3,
    TileDecor4 => TILE_DECOR_4,
    TileTree => TILE_TREE,
    TileBerryBush => TILE_BERRY_BUSH,
    TileBush => TILE_BUSH,
    TileFire => TILE_FIRE,
    TileFireLow => TILE_FIRE_LOW,
    TileSpiderEggs => TILE_SPIDER_EGGS,
    ItemStick => ITEM_STICK,
    ItemStone => ITEM_STONE,
    ItemWood => ITEM_WOOD,
    ItemBerry => ITEM_BERRY,
    ItemCookedBerry => ITEM_COOKED_BERRY,
    ItemAxe => ITEM_AXE,
    ItemCampfire => ITEM_CAMPFIRE,
    AgentUp => AGENT_GENERIC_UP,
    AgentRight => AGENT_GENERIC_RIGHT,
    AgentDown => AGENT_GENERIC_DOWN,
    AgentLeft => AGENT_GENERIC_LEFT,
    Spider => AGENT_SPIDER,
    UiHealth => UI_HEALTH,
    UiHunger => UI_HUNGER,
    UiHands => UI_HANDS,
    UiBackpack => UI_BACKPACK,
    UiDay => UI_DAY,
    UiNight => UI_NIGHT,
    Digit0 => UI_DIGITS[0],
    Digit1 => UI_DIGITS[1],
    Digit2 => UI_DIGITS[2],
    Digit3 => UI_DIGITS[3],
    Digit4 => UI_DIGITS[4],
    Digit5 => UI_DIGITS[5],
    Digit6 => UI_DIGITS[6],
    Digit7 => UI_DIGITS[7],
    Digit8 => UI_DIGITS[8],
    Digit9 => UI_DIGITS[9],
});

impl SurvivalObs {
    /// Bare open ground: where an item can be set down or an agent spawn.
    fn is_floor(self) -> bool {
        use SurvivalObs::*;
        matches!(
            self,
            TileEmpty | TileDecor1 | TileDecor2 | TileDecor3 | TileDecor4
        )
    }

    /// Open ground, bare or with an item lying on it. An agent standing on an
    /// item hides it until it steps off; bushes, trees and fires stand in the
    /// way.
    fn walkable(self) -> bool {
        self.is_floor() || self.item().is_some()
    }

    fn is_fire(self) -> bool {
        matches!(self, SurvivalObs::TileFire | SurvivalObs::TileFireLow)
    }

    /// Agents and spiders: drawn over the ground they stand on.
    fn is_creature(self) -> bool {
        AGENT_TILES.contains(&self) || self == SurvivalObs::Spider
    }

    fn opaque(self) -> bool {
        matches!(
            self,
            SurvivalObs::TileWall | SurvivalObs::TileDestructibleWall
        )
    }

    /// What map generation pays to dig a path through this tile when it
    /// joins up open ground; 0 for ground already open. Paths would rather
    /// cut through forest than tunnel rock or bridge water, and fell a bush
    /// last of all.
    fn dig_cost(self) -> u32 {
        use SurvivalObs::*;
        match self {
            _ if self.walkable() => 0,
            TileTree => 1,
            TileWall | TileDestructibleWall => 2,
            TileWater => 3,
            _ => 4,
        }
    }

    fn item(self) -> Option<Item> {
        use SurvivalObs::*;
        Some(match self {
            ItemStick => Item::Stick,
            ItemStone => Item::Stone,
            ItemWood => Item::Wood,
            ItemBerry => Item::Berry,
            ItemCookedBerry => Item::CookedBerry,
            ItemAxe => Item::Axe,
            ItemCampfire => Item::Campfire,
            _ => return None,
        })
    }
}

/// The agent's tile by facing, in `DIRECTIONS` order.
const AGENT_TILES: [SurvivalObs; 4] = [
    SurvivalObs::AgentUp,
    SurvivalObs::AgentRight,
    SurvivalObs::AgentDown,
    SurvivalObs::AgentLeft,
];

const DIGIT_TILES: [SurvivalObs; 10] = [
    SurvivalObs::Digit0,
    SurvivalObs::Digit1,
    SurvivalObs::Digit2,
    SurvivalObs::Digit3,
    SurvivalObs::Digit4,
    SurvivalObs::Digit5,
    SurvivalObs::Digit6,
    SurvivalObs::Digit7,
    SurvivalObs::Digit8,
    SurvivalObs::Digit9,
];

/// The eight cells around one.
const AROUND: [Position; 8] = [
    Position { x: -1, y: -1 },
    Position { x: 0, y: -1 },
    Position { x: 1, y: -1 },
    Position { x: -1, y: 0 },
    Position { x: 1, y: 0 },
    Position { x: -1, y: 1 },
    Position { x: 0, y: 1 },
    Position { x: 1, y: 1 },
];

/// Whether two cells share a side.
fn beside(a: Position, b: Position) -> bool {
    (a.x - b.x).abs() + (a.y - b.y).abs() == 1
}

/// Whether `offset` lies inside a disc of `radius` cells; the `+ radius`
/// rounds the disc out so small ones aren't diamonds.
fn within(offset: Position, radius: i32) -> bool {
    offset.x * offset.x + offset.y * offset.y <= radius * radius + radius
}

/// Facing offsets in `MOVES` order; env +y is up.
const DIRECTIONS: [Position; 4] = [
    Position { x: 0, y: 1 },
    Position { x: 1, y: 0 },
    Position { x: 0, y: -1 },
    Position { x: -1, y: 0 },
];

/// How far map generation bends its noise fields, in cells.
const TERRAIN_WARP: f32 = 8.0;
/// Pockets of open ground smaller than this are filled in rather than joined
/// up to the rest.
const MIN_REGION: usize = 24;

/// The kinds of land between the water and the rock.
#[derive(Debug, Clone, Copy)]
enum Biome {
    Forest,
    Meadow,
    Scrub,
}

impl Biome {
    /// What the biome's ground grows, one roll per cell: dense trees with
    /// sticks under them in forest, berry bushes in meadow, stones in scrub.
    /// Each also has its own decor, so bare ground shows which biome it is.
    fn growth(self) -> &'static [(SurvivalObs, f64)] {
        use SurvivalObs::*;
        match self {
            Biome::Forest => &[
                (TileTree, 0.30),
                (ItemStick, 0.04),
                (TileBerryBush, 0.004),
                (TileDecor4, 0.08),
            ],
            Biome::Meadow => &[
                (TileBerryBush, 0.02),
                (TileTree, 0.015),
                (ItemStick, 0.008),
                (ItemStone, 0.004),
                (TileDecor1, 0.12),
            ],
            Biome::Scrub => &[
                (ItemStone, 0.05),
                (TileTree, 0.008),
                (ItemStick, 0.008),
                (TileDecor2, 0.08),
                (TileDecor3, 0.03),
            ],
        }
    }
}

/// Where the UI band puts things. Its top row is the stats, each a label
/// then a three digit number: health from column 0, hunger from column 5,
/// and the columns after that kept free for temperature. The row under it is
/// the inventory, each slot a label then the item, hands from column 0 and
/// backpack from column 3; then the sun or the moon at column 6.
const HEALTH_COL: usize = 0;
const HUNGER_COL: usize = 5;
const STAT_DIGITS: usize = 3;
const HANDS_COL: usize = 0;
const BACKPACK_COL: usize = 3;
const CLOCK_COL: usize = 6;
/// The narrowest view the band fits in.
const UI_WIDTH: usize = HUNGER_COL + 1 + STAT_DIGITS;
const _: () = assert!(CLOCK_COL < UI_WIDTH);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Item {
    Stick,
    Stone,
    Wood,
    Berry,
    CookedBerry,
    Axe,
    Campfire,
}

impl Item {
    fn tile(self) -> SurvivalObs {
        match self {
            Item::Stick => SurvivalObs::ItemStick,
            Item::Stone => SurvivalObs::ItemStone,
            Item::Wood => SurvivalObs::ItemWood,
            Item::Berry => SurvivalObs::ItemBerry,
            Item::CookedBerry => SurvivalObs::ItemCookedBerry,
            Item::Axe => SurvivalObs::ItemAxe,
            Item::Campfire => SurvivalObs::ItemCampfire,
        }
    }

    /// The achievement for picking this up, for the raw materials.
    fn collected(self) -> Option<Achievement> {
        match self {
            Item::Stick => Some(Achievement::CollectStick),
            Item::Stone => Some(Achievement::CollectStone),
            Item::Wood => Some(Achievement::CollectWood),
            Item::Berry => Some(Achievement::CollectBerry),
            _ => None,
        }
    }
}

/// Combining the item in hand with the backpack's makes the third, either
/// way round, into the hand.
const RECIPES: &[(Item, Item, Item, Achievement)] = &[
    (Item::Stick, Item::Stone, Item::Axe, Achievement::MakeAxe),
    (
        Item::Wood,
        Item::Stone,
        Item::Campfire,
        Achievement::MakeCampfire,
    ),
];

fn recipe(a: Item, b: Item) -> Option<(Item, Achievement)> {
    RECIPES
        .iter()
        .find(|&&(x, y, _, _)| (x, y) == (a, b) || (y, x) == (a, b))
        .map(|&(_, _, made, achievement)| (made, achievement))
}

/// Slow work. Using a job's tool on its tile in front starts it, and locks the
/// agent in place, able only to wait, until its steps are done; then the
/// tile becomes what the job leaves of it. Felling a tree with an axe is the
/// first; digging and mining are meant to join it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Job {
    Chop,
}

impl Job {
    const ALL: [Job; 1] = [Job::Chop];

    /// The job `tool` does on `tile`, if any.
    fn of(tool: Item, tile: SurvivalObs) -> Option<Job> {
        Self::ALL
            .into_iter()
            .find(|job| job.tool() == tool && job.works() == tile)
    }

    fn tool(self) -> Item {
        match self {
            Job::Chop => Item::Axe,
        }
    }

    /// The tile the job works on.
    fn works(self) -> SurvivalObs {
        match self {
            Job::Chop => SurvivalObs::TileTree,
        }
    }

    /// What the job leaves of its tile when it's done.
    fn leaves(self) -> SurvivalObs {
        match self {
            Job::Chop => SurvivalObs::ItemWood,
        }
    }

    /// Steps the job takes, the use that starts it included.
    fn steps(self, config: &SurvivalConfig) -> u32 {
        match self {
            Job::Chop => config.chop_steps,
        }
    }

    fn achievement(self) -> Achievement {
        match self {
            Job::Chop => Achievement::ChopTree,
        }
    }
}

/// A job under way.
#[derive(Debug, Clone, Copy)]
struct Work {
    job: Job,
    /// The tile being worked, in front of the agent.
    target: Position,
    steps_left: u32,
}

/// Craftax-style milestones: each counts at most once per life, so the
/// metric reads as how far agents get rather than how often they repeat a
/// step.
#[derive(Debug, Clone, Copy)]
enum Achievement {
    CollectStick,
    CollectStone,
    CollectWood,
    CollectBerry,
    ChopTree,
    MakeAxe,
    MakeCampfire,
    PlaceFire,
    RefuelFire,
    CookBerry,
    EatBerry,
    EatCookedBerry,
}

/// Metric names, in `Achievement` order.
const ACHIEVEMENTS: [&str; 12] = [
    "collect_stick",
    "collect_stone",
    "collect_wood",
    "collect_berry",
    "chop_tree",
    "make_axe",
    "make_campfire",
    "place_fire",
    "refuel_fire",
    "cook_berry",
    "eat_berry",
    "eat_cooked_berry",
];

vocab_enum!(SurvivalAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
    Grab => GRAB_ACTION,
    Drop => DROP_ACTION,
    Swap => SWAP_ACTION,
    Use => USE_ACTION,
    Eat => EAT_ACTION,
    Combine => COMBINE_ACTION,
    Noop => NOOP,
});

impl SurvivalAction {
    /// The facing a move turns to, as a `DIRECTIONS` index.
    fn heading(self) -> Option<u8> {
        use SurvivalAction::*;
        match self {
            MoveUp => Some(0),
            MoveRight => Some(1),
            MoveDown => Some(2),
            MoveLeft => Some(3),
            _ => None,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Survivor {
    position: Position,
    /// Facing as a `DIRECTIONS` index.
    dir: u8,
    hunger: u16,
    health: u16,
    hands: Option<Item>,
    backpack: Option<Item>,
    /// Achievements unlocked this life, one bit per `Achievement`.
    unlocked: u16,
    /// The job it is busy with, if any.
    work: Option<Work>,
}

impl Survivor {
    fn spawn(position: Position, dir: u8, config: &SurvivalConfig) -> Self {
        Self {
            position,
            dir,
            hunger: config.start_hunger,
            health: config.start_health,
            ..Default::default()
        }
    }

    fn ahead(&self) -> Position {
        self.position + DIRECTIONS[self.dir as usize]
    }

    fn tile(&self) -> SurvivalObs {
        AGENT_TILES[self.dir as usize]
    }

    fn eat(&mut self, food: u16) {
        self.hunger = self.hunger.saturating_add(food).min(MAX_STAT);
    }

    fn unlock(&mut self, metrics: &mut SurvivalMetrics, achievement: Achievement) {
        let bit = 1 << achievement as u16;
        if self.unlocked & bit == 0 {
            self.unlocked |= bit;
            metrics.achievements[achievement as usize] += 1.0;
        }
    }
}

#[derive(Debug, Clone)]
struct SurvivalState {
    rng: SmallRng,
    agents: Vec<Survivor>,
    agent_order: Vec<usize>, // agent turn order
    time: usize,

    base_map: Array2<SurvivalObs>, // terrain and ground items, without agents
    map: Array2<SurvivalObs>,      // the base map plus the agents
    fires: Vec<Fire>,
    /// Picked bushes and the step each fruits again, in that order.
    regrowing: VecDeque<(Position, usize)>,
    /// Ground a fire lights: spiders keep off it, and agents see it at night
    /// as far as by day. Brought up to date every world turn.
    lit: Array2<bool>,
    /// Where the spider nests are, and the spiders out of them.
    eggs: Vec<Position>,
    spiders: Vec<Spider>,
    free_positions: Vec<Position>, // these are used to calculate spawn positions
}

#[derive(Debug, Clone, Copy)]
struct Spider {
    position: Position,
    /// The nest it hatched from, and goes back to at dawn.
    nest: Position,
}

#[derive(Debug, Clone, Copy)]
struct Fire {
    position: Position,
    /// Steps left before it goes out.
    burn_left: u32,
}

#[derive(Debug, Default, Clone)]
struct SurvivalMetrics {
    deaths: f64,
    spider_bites: f64,
    achievements: [f64; ACHIEVEMENTS.len()],
}

/// A first pass at a survival game: agents keep themselves fed off berry
/// bushes, and craft from what lies around.
///
/// Each agent has health and hunger, 0 to [`MAX_STAT`]. Hunger drains with
/// time; while it is high health grows back, and at zero health drains
/// instead. An agent whose health runs out drops what it carries where it
/// stood, is flagged terminated, and respawns with fresh stats elsewhere.
///
/// Everything an agent does to the world it does to the tile in front of it,
/// and its tile shows which way that is. Moving turns the agent to face the
/// move even when the way is blocked, so turning in place is a blocked move.
/// Items on the ground don't block: an agent walks over them, hiding the one
/// it stands on, and has to step off and face it to pick it up.
/// The inventory is two slots, hands and backpack. Grab takes the item in
/// front (or the berries off a bush) into empty hands, drop sets the hands'
/// item down in front, swap trades hands and backpack. Use invokes the
/// hand's item on what is in front: the axe sets to felling a tree, a
/// campfire is set down lit, and a raw berry held to a fire cooks at once.
/// Felling is a [`Job`], slow work: the agent can only wait until it is done,
/// `chop_steps` in all, and then a log lies where the tree stood.
/// Eat eats the food in hand, cooked berries feeding more than raw. Combine
/// turns the hand and backpack items into a new one per [`RECIPES`]: stick
/// and stone make an axe, wood and stone a campfire.
///
/// Days and nights alternate, starting at dawn. By night an agent sees only
/// `night_vision_radius` around itself, except for ground a fire lights,
/// which it sees as far as by day. Night closes in gradually: over the last
/// `dusk_length` steps of the day, sight shrinks step by step from the
/// whole view down to the night's, so the dark is a warning before it is a
/// danger. A fire burns `fire_burn_steps`, the last
/// `fire_low_steps` of them low; using wood on a low fire stokes it back up.
/// Spider nests hatch a giant spider each at nightfall, and the spiders hunt
/// until dawn: each bites an agent next to it, or walks toward the nearest it
/// can track, or wanders. At dawn they walk back to their nests and burrow in;
/// a nest hatches again only once its spider is home. Spiders fear fire. They
/// never set foot on lit ground, can't bite or track an agent standing on it,
/// and one caught in a fire's light walks straight out of it, so a fire is
/// the safe place to spend the night.
///
/// The map is biomes. An elevation field floods the low ground and raises
/// dirt banks on the heights, diggable walls once a tool can dig them; the
/// map's edge is solid wall. A moisture field splits the land between into
/// forest (trees and sticks), meadow (berry bushes) and scrub (stones), so
/// where to look for a thing is something to learn. Pockets of open ground
/// too small to matter are filled in and the rest joined up by paths, so
/// every agent can reach every other, and agents start out of each other's
/// sight where the map allows. Spider nests go in the forest.
///
/// There is no reward yet; the metrics count deaths and per-life
/// achievements.
///
/// The UI band shows health and hunger as numbers, what is in hands and
/// backpack, and whether it is day or night (see [`HEALTH_COL`] and friends
/// for the layout).
#[derive(Debug, Clone)]
pub struct Survival {
    pub config: SurvivalConfig,
    state: SurvivalState,
    metrics: SurvivalMetrics,

    // max steps for a single episode
    length: usize,

    pad_width: i32,
    pad_height: i32,

    // full map size including wall padding on both sides
    width: i32,
    height: i32,

    obs_spec: ObservationSpec,
    action_spec: ActionSpec,

    obs_vocab: Vocabulary,
    action_vocab: Vocabulary,
}

impl Survival {
    pub fn new(config: &SurvivalConfig, length: usize) -> Self {
        assert!(
            config.view_width as usize >= UI_WIDTH,
            "the UI band needs a view at least {UI_WIDTH} wide"
        );
        assert!(
            config.view_height >= 3,
            "the map padding has to reach the tile in front of an agent"
        );
        assert!(
            config.start_hunger <= MAX_STAT && config.start_health <= MAX_STAT,
            "starting stats run past {MAX_STAT}"
        );
        assert!(
            config.start_health > 0,
            "agents would spawn dead at zero health"
        );
        assert!(
            config.hunger_interval > 0 && config.regen_interval > 0,
            "stat intervals must be at least one step"
        );
        let share = |fraction: f64| (0.0..=1.0).contains(&fraction);
        assert!(
            share(config.water_fraction)
                && share(config.rock_fraction)
                && share(config.water_fraction + config.rock_fraction)
                && share(config.forest_fraction)
                && share(config.scrub_fraction)
                && share(config.forest_fraction + config.scrub_fraction),
            "terrain fractions are shares, and add up to at most the whole"
        );
        assert!(config.day_length > 0, "every cycle starts with a day");
        assert!(
            config.dusk_length <= config.day_length,
            "dusk is the end of the day, so no longer than it"
        );

        let action_vocab = SurvivalAction::vocab();
        let obs_vocab = SurvivalObs::vocab();

        let pad_width = config.view_width / 2;
        let pad_height = config.view_height / 2;

        let view_height = config.view_height + UI_HEIGHT as i32;

        let width = config.width + 2 * pad_width;
        let height = config.height + 2 * pad_height;
        let dim = (width as usize, height as usize);

        let obs_spec = ObservationSpec::new(config.view_width, view_height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        Self {
            config: config.clone(),
            state: SurvivalState {
                rng: SmallRng::seed_from_u64(0),
                agents: Vec::with_capacity(config.num_agents),
                agent_order: (0..config.num_agents).collect(),
                time: 0,
                base_map: Array2::from_elem(dim, SurvivalObs::TileEmpty),
                map: Array2::from_elem(dim, SurvivalObs::TileEmpty),
                fires: Vec::new(),
                regrowing: VecDeque::new(),
                lit: Array2::from_elem(dim, false),
                eggs: Vec::new(),
                spiders: Vec::new(),
                free_positions: Vec::new(),
            },
            metrics: SurvivalMetrics::default(),
            length,

            pad_width,
            pad_height,
            width,
            height,

            obs_spec,
            action_spec,

            obs_vocab,
            action_vocab,
        }
    }

    fn calculate_free_positions(&mut self) {
        self.state.free_positions.clear();
        for x in self.pad_width..self.width - self.pad_width {
            for y in self.pad_height..self.height - self.pad_height {
                let position = Position::new(x, y);
                if self.state.map[position.idx()].is_floor() {
                    self.state.free_positions.push(position);
                }
            }
        }

        self.state.free_positions.shuffle(&mut self.state.rng);
    }

    /// Changes what lies on a cell. A creature standing there stays drawn
    /// over it, as over any item it stands on.
    fn set_ground(&mut self, position: Position, tile: SurvivalObs) {
        self.state.base_map[position.idx()] = tile;
        if !self.state.map[position.idx()].is_creature() {
            self.state.map[position.idx()] = tile;
        }
    }

    fn can(&self, agent: &Survivor, action: SurvivalAction) -> bool {
        use SurvivalAction::*;
        if agent.work.is_some() {
            return action == Noop;
        }
        let ahead = self.state.map[agent.ahead().idx()];
        match action {
            MoveUp | MoveRight | MoveDown | MoveLeft | Noop => true,
            Grab => {
                agent.hands.is_none()
                    && (ahead.item().is_some() || ahead == SurvivalObs::TileBerryBush)
            }
            Drop => agent.hands.is_some() && ahead.is_floor(),
            Swap => agent.hands.is_some() || agent.backpack.is_some(),
            Use => match agent.hands {
                Some(Item::Berry) => ahead.is_fire(),
                Some(Item::Wood) => ahead == SurvivalObs::TileFireLow,
                Some(Item::Campfire) => ahead.is_floor(),
                Some(tool) => Job::of(tool, ahead).is_some(),
                None => false,
            },
            Eat => matches!(agent.hands, Some(Item::Berry | Item::CookedBerry)),
            Combine => match (agent.hands, agent.backpack) {
                (Some(a), Some(b)) => recipe(a, b).is_some(),
                _ => false,
            },
        }
    }

    /// Carries out one agent's action, or nothing if the action is not legal
    /// for it right now: an agent earlier in the turn order may have taken
    /// what it reached for.
    fn act(&mut self, agent_id: usize, action: SurvivalAction) {
        let mut agent = self.state.agents[agent_id];
        if agent.work.is_some() {
            // a busy agent's turn goes into its job, whatever it asked for
            self.work(&mut agent);
            self.state.agents[agent_id] = agent;
            return;
        }
        if !self.can(&agent, action) {
            return;
        }

        let ahead = agent.ahead();
        match action {
            SurvivalAction::MoveUp
            | SurvivalAction::MoveRight
            | SurvivalAction::MoveDown
            | SurvivalAction::MoveLeft => {
                agent.dir = action.heading().expect("moves have a heading");
                let target = agent.ahead();
                if self.state.map[target.idx()].walkable() {
                    let from = agent.position.idx();
                    self.state.map[from] = self.state.base_map[from];
                    agent.position = target;
                }
                self.state.map[agent.position.idx()] = agent.tile();
            }
            SurvivalAction::Grab => {
                let tile = self.state.map[ahead.idx()];
                let item = if tile == SurvivalObs::TileBerryBush {
                    self.set_ground(ahead, SurvivalObs::TileBush);
                    let ripe = self.state.time + self.config.bush_regrow_steps as usize;
                    self.state.regrowing.push_back((ahead, ripe));
                    Item::Berry
                } else {
                    self.set_ground(ahead, SurvivalObs::TileEmpty);
                    tile.item()
                        .expect("grab is legal only facing an item or a bush")
                };
                agent.hands = Some(item);
                if let Some(achievement) = item.collected() {
                    agent.unlock(&mut self.metrics, achievement);
                }
            }
            SurvivalAction::Drop => {
                let item = agent
                    .hands
                    .take()
                    .expect("drop is legal only holding something");
                self.set_ground(ahead, item.tile());
            }
            SurvivalAction::Swap => std::mem::swap(&mut agent.hands, &mut agent.backpack),
            SurvivalAction::Use => match agent.hands {
                Some(Item::Berry) => {
                    agent.hands = Some(Item::CookedBerry);
                    agent.unlock(&mut self.metrics, Achievement::CookBerry);
                }
                Some(Item::Wood) => {
                    let fire = self
                        .state
                        .fires
                        .iter_mut()
                        .find(|fire| fire.position == ahead)
                        .expect("a low fire is a burning one");
                    fire.burn_left = self.config.fire_burn_steps;
                    self.set_ground(ahead, self.fire_tile(self.config.fire_burn_steps));
                    agent.hands = None;
                    agent.unlock(&mut self.metrics, Achievement::RefuelFire);
                }
                Some(Item::Campfire) => {
                    let burn_left = self.config.fire_burn_steps;
                    self.set_ground(ahead, self.fire_tile(burn_left));
                    self.state.fires.push(Fire {
                        position: ahead,
                        burn_left,
                    });
                    agent.hands = None;
                    agent.unlock(&mut self.metrics, Achievement::PlaceFire);
                }
                Some(tool) => {
                    let job = Job::of(tool, self.state.map[ahead.idx()])
                        .expect("use is legal only for items that have one");
                    agent.work = Some(Work {
                        job,
                        target: ahead,
                        steps_left: job.steps(&self.config),
                    });
                    // the use is the job's first step
                    self.work(&mut agent);
                }
                None => unreachable!("use is legal only holding something"),
            },
            SurvivalAction::Eat => {
                let (food, achievement) = match agent.hands {
                    Some(Item::Berry) => (self.config.berry_food, Achievement::EatBerry),
                    Some(Item::CookedBerry) => {
                        (self.config.cooked_berry_food, Achievement::EatCookedBerry)
                    }
                    _ => unreachable!("eat is legal only holding food"),
                };
                agent.eat(food);
                agent.hands = None;
                agent.unlock(&mut self.metrics, achievement);
            }
            SurvivalAction::Combine => {
                let (hands, backpack) = (agent.hands, agent.backpack);
                let (made, achievement) = hands
                    .zip(backpack)
                    .and_then(|(a, b)| recipe(a, b))
                    .expect("combine is legal only for a recipe");
                agent.hands = Some(made);
                agent.backpack = None;
                agent.unlock(&mut self.metrics, achievement);
            }
            SurvivalAction::Noop => {}
        }

        self.state.agents[agent_id] = agent;
    }

    /// Puts a step into the agent's job, finishing it once its steps run out.
    /// If its tile has changed meanwhile (another agent got there first), the
    /// job comes to nothing.
    fn work(&mut self, agent: &mut Survivor) {
        let Some(mut work) = agent.work else {
            return;
        };
        work.steps_left = work.steps_left.saturating_sub(1);
        if work.steps_left > 0 {
            agent.work = Some(work);
            return;
        }

        agent.work = None;
        if self.state.base_map[work.target.idx()] == work.job.works() {
            self.set_ground(work.target, work.job.leaves());
            agent.unlock(&mut self.metrics, work.job.achievement());
        }
    }

    /// The world's own turn, after the agents': fires burn down, picked
    /// bushes fruit again, night falls or lifts, and the spiders hunt by night
    /// and go home by day.
    fn tick_world(&mut self) {
        let time = self.state.time;

        self.burn_fires();
        while let Some(&(bush, ripe)) = self.state.regrowing.front() {
            if ripe > time {
                break;
            }
            self.state.regrowing.pop_front();
            self.set_ground(bush, SurvivalObs::TileBerryBush);
        }
        self.light_up();

        // the clock the agents will wake to, at the end of this step
        let now = time + 1;
        if now % self.cycle_length() == self.config.day_length as usize {
            self.hatch_spiders();
        }
        if self.is_night(now) {
            self.hunt();
        } else {
            self.go_home();
        }
    }

    fn cycle_length(&self) -> usize {
        (self.config.day_length + self.config.night_length) as usize
    }

    fn is_night(&self, time: usize) -> bool {
        time % self.cycle_length() >= self.config.day_length as usize
    }

    /// How far agents see around themselves at `time`, firelight aside: no
    /// limit by day, `night_vision_radius` by night, and in between, through
    /// dusk, a radius shrinking a little every step from one that takes in the
    /// whole view.
    fn vision_radius(&self, time: usize) -> Option<i32> {
        let night = self.config.night_vision_radius;
        let phase = time % self.cycle_length();
        let day = self.config.day_length as usize;
        let dusk = self.config.dusk_length as usize;
        if phase >= day {
            return Some(night);
        }
        // steps of daylight left, this one included
        let left = day - phase;
        if left > dusk {
            return None;
        }

        // the smallest disc around the agent that covers its whole view
        let corner = Position::new(self.config.view_width / 2, self.config.view_height / 2);
        let full = (0..)
            .find(|&r| within(corner, r))
            .expect("some disc covers the view");
        Some(night + (full - night) * left as i32 / (dusk as i32 + 1))
    }

    /// A fire shows on the map for `burn_left` steps counting this one, so it
    /// is in its last `fire_low_steps` below that many.
    fn fire_tile(&self, burn_left: u32) -> SurvivalObs {
        if burn_left < self.config.fire_low_steps {
            SurvivalObs::TileFireLow
        } else {
            SurvivalObs::TileFire
        }
    }

    /// Every fire burns a step down, going low near the end and out after
    /// it.
    fn burn_fires(&mut self) {
        let mut i = 0;
        while i < self.state.fires.len() {
            let Fire {
                position,
                burn_left,
            } = self.state.fires[i];
            if burn_left == 0 {
                self.state.fires.swap_remove(i);
                self.set_ground(position, SurvivalObs::TileEmpty);
                continue;
            }
            self.state.fires[i].burn_left = burn_left - 1;
            let tile = self.fire_tile(burn_left - 1);
            if self.state.base_map[position.idx()] != tile {
                self.set_ground(position, tile);
            }
            i += 1;
        }
    }

    /// Marks the ground each fire lights.
    fn light_up(&mut self) {
        let SurvivalState { fires, lit, .. } = &mut self.state;
        let radius = self.config.fire_light_radius;
        lit.fill(false);
        for fire in fires.iter() {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let offset = Position::new(dx, dy);
                    if !within(offset, radius) {
                        continue;
                    }
                    if let Some(cell) = lit.get_mut((fire.position + offset).idx()) {
                        *cell = true;
                    }
                }
            }
        }
    }

    /// Open ground, dark, with no one on it.
    fn spider_can_enter(&self, cell: Position) -> bool {
        self.state.map[cell.idx()].walkable() && !self.state.lit[cell.idx()]
    }

    /// Each nest whose spider is home, with dark open ground beside it,
    /// hatches a spider there.
    fn hatch_spiders(&mut self) {
        for i in 0..self.state.eggs.len() {
            let nest = self.state.eggs[i];
            if self.state.spiders.iter().any(|spider| spider.nest == nest) {
                continue;
            }
            let spot = DIRECTIONS
                .iter()
                .map(|&d| nest + d)
                .find(|&cell| self.spider_can_enter(cell));
            if let Some(position) = spot {
                self.state.spiders.push(Spider { position, nest });
                self.state.map[position.idx()] = SurvivalObs::Spider;
            }
        }
    }

    /// The spiders' turn by night. Each in turn walks out of any light it
    /// stands in; failing that bites an agent beside it in the dark; failing
    /// that closes on the nearest agent it can track; failing that wanders.
    fn hunt(&mut self) {
        if self.state.spiders.is_empty() {
            return;
        }
        let trail = self.scent();

        for i in 0..self.state.spiders.len() {
            let spider = self.state.spiders[i].position;
            if self.state.lit[spider.idx()] {
                self.flee_light(i);
                continue;
            }
            let around = DIRECTIONS.map(|d| spider + d);

            let prey = around.iter().find_map(|&cell| {
                let lit = self.state.lit[cell.idx()];
                let agents = self.state.agents.iter();
                (!lit).then(|| agents.into_iter().position(|a| a.position == cell))?
            });
            if let Some(agent_id) = prey {
                let agent = &mut self.state.agents[agent_id];
                agent.health = agent.health.saturating_sub(self.config.spider_damage);
                self.metrics.spider_bites += 1.0;
                continue;
            }

            let open: Vec<Position> = around
                .into_iter()
                .filter(|&cell| self.spider_can_enter(cell))
                .collect();
            let next = if trail[spider.idx()] != u32::MAX {
                open.iter()
                    .copied()
                    .filter(|cell| trail[cell.idx()] < trail[spider.idx()])
                    .min_by_key(|cell| trail[cell.idx()])
            } else {
                open.choose(&mut self.state.rng).copied()
            };
            if let Some(next) = next {
                self.move_spider(i, next);
            }
        }
    }

    /// The spiders' turn by day. Each walks out of any light it stands in, or
    /// else heads home by the shortest dark way and burrows back into its
    /// nest once beside it. They bite no one on the way.
    fn go_home(&mut self) {
        let mut i = 0;
        while i < self.state.spiders.len() {
            let Spider { position, nest } = self.state.spiders[i];
            if self.state.lit[position.idx()] {
                self.flee_light(i);
            } else if beside(position, nest) {
                self.state.map[position.idx()] = self.state.base_map[position.idx()];
                self.state.spiders.swap_remove(i);
                continue;
            } else {
                let dark = |cell: Position| {
                    self.state.base_map[cell.idx()].walkable() && !self.state.lit[cell.idx()]
                };
                if let Some(next) = self.first_step(position, dark, |cell| beside(cell, nest)) {
                    self.move_spider(i, next);
                }
            }
            i += 1;
        }
    }

    /// Takes a spider standing in light a step along the shortest way out
    /// of it.
    fn flee_light(&mut self, i: usize) {
        let from = self.state.spiders[i].position;
        let open = |cell: Position| self.state.base_map[cell.idx()].walkable();
        if let Some(next) = self.first_step(from, open, |cell| !self.state.lit[cell.idx()]) {
            self.move_spider(i, next);
        }
    }

    /// Moves a spider onto `next`, if no one is standing there.
    fn move_spider(&mut self, i: usize, next: Position) {
        if !self.state.map[next.idx()].walkable() {
            return;
        }
        let from = self.state.spiders[i].position;
        self.state.map[from.idx()] = self.state.base_map[from.idx()];
        self.state.map[next.idx()] = SurvivalObs::Spider;
        self.state.spiders[i].position = next;
    }

    /// The first step of a shortest walk from `from` through `passable` cells
    /// to the nearest cell `goal` accepts; `None` if no such cell is
    /// reachable. Who is standing where is left to the caller.
    fn first_step(
        &self,
        from: Position,
        passable: impl Fn(Position) -> bool,
        goal: impl Fn(Position) -> bool,
    ) -> Option<Position> {
        let mut seen = Array2::from_elem(self.state.map.dim(), false);
        seen[from.idx()] = true;
        // each cell reached, with the first step of the way it was reached by
        let mut frontier = VecDeque::new();
        for step in DIRECTIONS {
            let next = from + step;
            if passable(next) {
                seen[next.idx()] = true;
                frontier.push_back((next, next));
            }
        }
        while let Some((cell, first)) = frontier.pop_front() {
            if goal(cell) {
                return Some(first);
            }
            for step in DIRECTIONS {
                let next = cell + step;
                if !seen[next.idx()] && passable(next) {
                    seen[next.idx()] = true;
                    frontier.push_back((next, first));
                }
            }
        }
        None
    }

    /// How far each cell is, walking dark open ground, from the nearest agent
    /// standing in the dark: what a spider follows. `u32::MAX` past
    /// `spider_hunt_radius`, and through light, which hides the trail.
    fn scent(&self) -> Array2<u32> {
        let walkable_dark = |cell: Position| {
            self.state.base_map[cell.idx()].walkable() && !self.state.lit[cell.idx()]
        };
        let mut trail = Array2::from_elem(self.state.map.dim(), u32::MAX);
        let mut frontier = VecDeque::new();
        for agent in &self.state.agents {
            if walkable_dark(agent.position) {
                trail[agent.position.idx()] = 0;
                frontier.push_back(agent.position);
            }
        }
        while let Some(cell) = frontier.pop_front() {
            let distance = trail[cell.idx()];
            if distance >= self.config.spider_hunt_radius {
                continue;
            }
            for step in DIRECTIONS {
                let next = cell + step;
                if trail[next.idx()] == u32::MAX && walkable_dark(next) {
                    trail[next.idx()] = distance + 1;
                    frontier.push_back(next);
                }
            }
        }
        trail
    }

    /// Hunger drains, health follows it; returns whether the agent died,
    /// starved or bitten.
    fn tick_stats(&mut self, agent_id: usize) -> bool {
        let config = &self.config;
        // counting this step, so the first drain lands after a full interval
        let steps = self.state.time as u32 + 1;
        let agent = &mut self.state.agents[agent_id];

        if steps.is_multiple_of(config.hunger_interval) {
            agent.hunger = agent.hunger.saturating_sub(1);
        }
        if agent.hunger == 0 {
            agent.health = agent.health.saturating_sub(config.starve_damage);
        } else if agent.health > 0
            && agent.hunger >= config.regen_threshold
            && steps.is_multiple_of(config.regen_interval)
        {
            agent.health = (agent.health + 1).min(MAX_STAT);
        }

        agent.health == 0
    }

    /// Takes a dead agent off the map, leaving what it carried behind: the
    /// hands' item where it stood, the backpack's beside it. An item with no
    /// open ground left to land on is lost.
    fn kill(&mut self, agent_id: usize) {
        let agent = self.state.agents[agent_id];
        let position = agent.position;
        self.state.map[position.idx()] = self.state.base_map[position.idx()];

        let mut spots = std::iter::once(position).chain(DIRECTIONS.map(|d| position + d));
        for item in [agent.hands, agent.backpack].into_iter().flatten() {
            if let Some(spot) = spots.find(|spot| self.state.map[spot.idx()].is_floor()) {
                self.set_ground(spot, item.tile());
            }
        }

        self.metrics.deaths += 1.0;
    }

    /// Lays out the interior: an elevation field puts water in the low ground
    /// and rock on the heights, a moisture field splits the land between into
    /// biomes, each biome grows its own things, and the open ground is joined
    /// up.
    fn generate_map(&mut self) {
        let (width, height) = (self.config.width as usize, self.config.height as usize);
        let rng = &mut self.state.rng;
        let elevation = noise_field(width, height, TERRAIN_WARP, rng);
        let moisture = noise_field(width, height, TERRAIN_WARP, rng);

        let water = quantile(elevation.iter().copied(), self.config.water_fraction);
        let rock = quantile(elevation.iter().copied(), 1.0 - self.config.rock_fraction);
        let land: Vec<f32> = elevation
            .iter()
            .zip(&moisture)
            .filter(|&(&e, _)| water <= e && e <= rock)
            .map(|(_, &m)| m)
            .collect();
        let dry = quantile(land.iter().copied(), self.config.scrub_fraction);
        let wet = quantile(land.iter().copied(), 1.0 - self.config.forest_fraction);

        let mut forest = Array2::from_elem((width, height), false);
        let mut interior = self.state.base_map.slice_mut(s![
            self.pad_width as usize..(self.width - self.pad_width) as usize,
            self.pad_height as usize..(self.height - self.pad_height) as usize,
        ]);
        for (cell, tile) in interior.indexed_iter_mut() {
            let (e, m) = (elevation[cell], moisture[cell]);
            *tile = if e < water {
                SurvivalObs::TileWater
            } else if e > rock {
                SurvivalObs::TileDestructibleWall
            } else {
                forest[cell] = m > wet;
                let biome = if m > wet {
                    Biome::Forest
                } else if m < dry {
                    Biome::Scrub
                } else {
                    Biome::Meadow
                };
                roll(biome.growth(), rng).unwrap_or(SurvivalObs::TileEmpty)
            };
        }

        connect_regions(
            interior,
            SurvivalObs::dig_cost,
            MIN_REGION,
            SurvivalObs::TileEmpty,
        );
        self.place_nests(&forest);
    }

    /// Sets the spider nests on open ground, in the forest while it has room.
    /// A nest only goes where all eight cells around it are open, so it
    /// never cuts a path in two: there is always a way round it.
    fn place_nests(&mut self, forest: &Array2<bool>) {
        let corner = Position::new(self.pad_width, self.pad_height);
        let mut sites: Vec<Position> = forest
            .indexed_iter()
            .map(|((x, y), _)| corner + Position::new(x as i32, y as i32))
            .filter(|site| self.state.base_map[site.idx()].is_floor())
            .collect();
        sites.shuffle(&mut self.state.rng);
        // a stable sort, so each group stays shuffled
        sites.sort_by_key(|&site| !forest[(site - corner).idx()]);

        self.state.eggs.clear();
        for site in sites {
            if self.state.eggs.len() == self.config.num_spider_eggs {
                break;
            }
            let base_map = &self.state.base_map;
            if AROUND
                .iter()
                .all(|&d| base_map[(site + d).idx()].walkable())
            {
                self.state.base_map[site.idx()] = SurvivalObs::TileSpiderEggs;
                self.state.eggs.push(site);
            }
        }
    }

    fn spawn(&mut self, agent_id: usize, position: Position) {
        let dir = self.state.rng.random_range(0..4);
        let agent = Survivor::spawn(position, dir, &self.config);
        self.state.map[position.idx()] = agent.tile();

        if agent_id < self.state.agents.len() {
            self.state.agents[agent_id] = agent;
        } else {
            self.state.agents.push(agent);
        }
    }

    fn encode_observations(&self, timestep: &mut TimeStepMut) {
        let fov_height = self.config.view_height as usize;
        // the top row of the band is the stats, the one under it inventory
        let stats_row = fov_height + UI_HEIGHT - 1;
        let slots_row = fov_height;
        let night = self.is_night(self.state.time);
        let sight = self.vision_radius(self.state.time);
        // as `fov::encode_visible` centres the window
        let half_width = self.config.view_width / 2;
        let half_height = self.config.view_height / 2;

        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            // wall padding keeps the view window inside the map
            let mut view = timestep.obs.slice_mut(s![agent_id, .., ..fov_height, 0]);
            fov::encode_visible(
                &self.state.map,
                agent.position,
                &mut view,
                SurvivalObs::Mask,
                |tile| tile.opaque(),
            );
            if let Some(sight) = sight {
                // only the near and the lit show in the dark
                let corner = agent.position - Position::new(half_width, half_height);
                for ((x, y), cell) in view.indexed_iter_mut() {
                    let position = corner + Position::new(x as i32, y as i32);
                    let near = within(position - agent.position, sight);
                    if !near && !self.state.lit[position.idx()] {
                        *cell = SurvivalObs::Mask.into();
                    }
                }
            }

            let mut ui = timestep.obs.slice_mut(s![agent_id, .., fov_height.., 0]);
            ui.fill(SurvivalObs::UI.into());

            let mut stats = timestep.obs.slice_mut(s![agent_id, .., stats_row, 0]);
            for (col, label, value) in [
                (HEALTH_COL, SurvivalObs::UiHealth, agent.health),
                (HUNGER_COL, SurvivalObs::UiHunger, agent.hunger),
            ] {
                stats[col] = label.into();
                write_number(
                    stats.slice_mut(s![col + 1..col + 1 + STAT_DIGITS]),
                    u32::from(value),
                    &DIGIT_TILES,
                );
            }

            let mut slots = timestep.obs.slice_mut(s![agent_id, .., slots_row, 0]);
            for (col, label, item) in [
                (HANDS_COL, SurvivalObs::UiHands, agent.hands),
                (BACKPACK_COL, SurvivalObs::UiBackpack, agent.backpack),
            ] {
                slots[col] = label.into();
                if let Some(item) = item {
                    slots[col + 1] = item.tile().into();
                }
            }
            slots[CLOCK_COL] = if night {
                SurvivalObs::UiNight
            } else {
                SurvivalObs::UiDay
            }
            .into();
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length);
        timestep.task_ids.fill(0);
    }

    fn encode_action_mask(&self, timestep: &mut TimeStepMut) {
        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            let mut mask = timestep.action_mask.row_mut(agent_id);
            for &action in SurvivalAction::TABLE {
                mask[action as usize] = self.can(agent, action);
            }
        }
    }
}

impl Environment for Survival {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.state.rng = SmallRng::seed_from_u64(seed);
        self.state.time = 0;

        self.state.base_map.fill(SurvivalObs::TileWall);
        self.generate_map();

        // Base map finished
        self.state.map.assign(&self.state.base_map);
        self.state.fires.clear();
        self.state.regrowing.clear();
        self.state.lit.fill(false);
        self.state.spiders.clear();

        self.state.agents.clear();
        self.calculate_free_positions();
        // out of each other's sight, where the map has room for it
        let apart = self.pad_width.max(self.pad_height) + 1;
        let spawns = spread_out(&self.state.free_positions, self.num_agents(), apart);
        assert_eq!(
            spawns.len(),
            self.num_agents(),
            "no open ground left to spawn on"
        );
        for (agent_id, position) in spawns.into_iter().enumerate() {
            self.spawn(agent_id, position);
        }

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.state.agent_order.shuffle(&mut self.state.rng);
        for i in 0..self.state.agent_order.len() {
            let agent_id = self.state.agent_order[i];
            timestep.last_action[agent_id] = actions[agent_id];
            timestep.reward[agent_id] = 0.0;
            self.act(agent_id, SurvivalAction::from_id(actions[agent_id]));
        }

        self.tick_world();

        let dead: Vec<usize> = (0..self.num_agents())
            .filter(|&agent_id| self.tick_stats(agent_id))
            .collect();
        for &agent_id in &dead {
            self.kill(agent_id);
        }
        if !dead.is_empty() {
            self.calculate_free_positions();
            for &agent_id in &dead {
                let position = self
                    .state
                    .free_positions
                    .pop()
                    .expect("no open ground left to respawn on");
                self.spawn(agent_id, position);
            }
        }

        self.state.time += 1;
        self.encode_observations(timestep);
        for &agent_id in &dead {
            timestep.terminated[agent_id] = true;
        }
        self.encode_action_mask(timestep);
    }

    fn consume_metrics(&mut self) -> serde_json::Value {
        let metrics = std::mem::take(&mut self.metrics);
        let agents = self.num_agents().max(1) as f64;
        let achievements: serde_json::Map<_, _> = ACHIEVEMENTS
            .iter()
            .zip(metrics.achievements)
            .map(|(&name, count)| (name.to_owned(), serde_json::Value::from(count / agents)))
            .collect();
        serde_json::json!({
            "deaths": metrics.deaths / agents,
            "spider_bites": metrics.spider_bites / agents,
            "achievements": achievements,
        })
    }

    fn observation_spec(&self) -> ObservationSpec {
        self.obs_spec
    }

    fn action_spec(&self) -> ActionSpec {
        self.action_spec
    }

    fn num_agents(&self) -> usize {
        self.config.num_agents
    }

    fn obs_vocab(&self) -> &Vocabulary {
        &self.obs_vocab
    }

    fn action_vocab(&self) -> &Vocabulary {
        &self.action_vocab
    }

    fn get_render_settings(&self) -> GridRenderSettings {
        GridRenderSettings {
            obs_vocab: self.obs_vocab.clone(),
            tile_width: self.config.width as usize,
            tile_height: self.config.height as usize,
            view_width: self.config.view_width as usize,
            view_height: self.config.view_height as usize + UI_HEIGHT,
            ui_height: UI_HEIGHT,
        }
    }

    fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
        let dim = (self.config.width as usize, self.config.height as usize);
        let tilemap = &mut grid_render_state.tilemap;
        if tilemap.dim() != dim {
            *tilemap = Array2::zeros(dim);
        }
        let interior = self.state.map.slice(s![
            self.pad_width as usize..(self.width - self.pad_width) as usize,
            self.pad_height as usize..(self.height - self.pad_height) as usize,
        ]);
        tilemap.zip_mut_with(&interior, |dst, &tile| *dst = tile.into());

        grid_render_state.agent_positions.clear();
        for agent in &self.state.agents {
            let local_pos = agent.position - Position::new(self.pad_width, self.pad_height);
            grid_render_state.agent_positions.push(local_pos);
        }
    }

    fn num_tasks(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timestep::TimeStepBuffers;
    use SurvivalAction::*;
    use SurvivalObs::*;

    /// A bare walled floor with nothing scattered on it and no agents, for
    /// placing things by hand.
    fn empty_env_with(config: SurvivalConfig) -> Survival {
        let mut env = Survival::new(
            &SurvivalConfig {
                width: 21,
                height: 21,
                ..config
            },
            512,
        );

        env.state.base_map.fill(TileWall);
        env.state
            .base_map
            .slice_mut(s![
                env.pad_width as usize..(env.width - env.pad_width) as usize,
                env.pad_height as usize..(env.height - env.pad_height) as usize,
            ])
            .fill(TileEmpty);
        env.state.map.assign(&env.state.base_map);

        env
    }

    fn center(env: &Survival) -> Position {
        Position::new(env.width / 2, env.height / 2)
    }

    /// Puts an agent on the map facing `dir`; call in agent-id order, and as
    /// many times as the config has agents.
    fn spawn_facing(env: &mut Survival, position: Position, dir: u8) {
        let agent = Survivor::spawn(position, dir, &env.config);
        env.state.map[position.idx()] = agent.tile();
        env.state.agents.push(agent);
    }

    fn agent(env: &mut Survival, agent_id: usize) -> &mut Survivor {
        &mut env.state.agents[agent_id]
    }

    fn step(env: &mut Survival, buffers: &mut TimeStepBuffers, actions: &[SurvivalAction]) {
        let actions: Vec<VocabId> = actions.iter().map(|&a| a.into()).collect();
        env.step(&actions, &mut buffers.view_mut());
    }

    /// Encodes the current state without stepping, for checking a setup.
    fn observe(env: &Survival) -> TimeStepBuffers {
        let mut buffers = TimeStepBuffers::new(env);
        env.encode_observations(&mut buffers.view_mut());
        env.encode_action_mask(&mut buffers.view_mut());
        buffers
    }

    fn legal(buffers: &TimeStepBuffers, agent_id: usize, action: SurvivalAction) -> bool {
        buffers.action_mask[[agent_id, action as usize]]
    }

    fn id(tile: SurvivalObs) -> VocabId {
        tile.into()
    }

    fn achieved(env: &Survival, achievement: Achievement) -> f64 {
        env.metrics.achievements[achievement as usize]
    }

    const UP: Position = DIRECTIONS[0];
    const RIGHT: Position = DIRECTIONS[1];
    const DOWN: Position = DIRECTIONS[2];
    const LEFT: Position = DIRECTIONS[3];

    /// Sets a fire burning on `position`, lighting the ground around it.
    fn light_fire(env: &mut Survival, position: Position) {
        env.set_ground(position, TileFire);
        env.state.fires.push(Fire {
            position,
            burn_left: 100,
        });
        env.light_up();
    }

    /// Puts a spider from `nest` out on `position`.
    fn release_spider(env: &mut Survival, position: Position, nest: Position) {
        env.state.map[position.idx()] = Spider;
        env.state.spiders.push(super::Spider { position, nest });
    }

    fn spiders_on_map(env: &Survival) -> usize {
        env.state.map.iter().filter(|&&t| t == Spider).count()
    }

    /// What the agent's view shows at `offset` from it.
    fn seen(
        env: &Survival,
        buffers: &TimeStepBuffers,
        agent_id: usize,
        offset: Position,
    ) -> SurvivalObs {
        let x = env.config.view_width / 2 + offset.x;
        let y = env.config.view_height / 2 + offset.y;
        SurvivalObs::from_id(buffers.obs[[agent_id, x as usize, y as usize, 0]])
    }

    /// Reset leaves every agent standing on open ground, painted facing the
    /// way it faces, with fresh stats; the scatter puts some of everything on
    /// the map; and the walls inside are all diggable, inside an edge of solid
    /// wall.
    #[test]
    fn reset_spawns_agents_on_open_ground_among_the_scatter() {
        let config = SurvivalConfig {
            num_agents: 4,
            width: 40,
            height: 40,
            ..Default::default()
        };
        let mut env = Survival::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(3, &mut buffers.view_mut());

        for agent in &env.state.agents {
            assert_eq!(env.state.map[agent.position.idx()], agent.tile());
            assert!(env.state.base_map[agent.position.idx()].is_floor());
            assert_eq!(agent.hunger, config.start_hunger);
            assert_eq!(agent.health, config.start_health);
            assert_eq!((agent.hands, agent.backpack), (None, None));
        }

        for tile in [
            TileTree,
            TileBerryBush,
            ItemStick,
            ItemStone,
            TileDestructibleWall,
        ] {
            assert!(
                env.state.base_map.iter().any(|&t| t == tile),
                "reset scattered no {tile:?}"
            );
        }

        let interior = env.state.base_map.slice(s![
            env.pad_width as usize..(env.width - env.pad_width) as usize,
            env.pad_height as usize..(env.height - env.pad_height) as usize,
        ]);
        assert!(!interior.iter().any(|&t| t == TileWall));
        assert_eq!(env.state.base_map[[0, 0]], TileWall);

        assert_eq!(env.state.eggs.len(), config.num_spider_eggs);
        for &nest in &env.state.eggs {
            assert_eq!(env.state.base_map[nest.idx()], TileSpiderEggs);
            assert!(
                AROUND
                    .iter()
                    .all(|&d| env.state.base_map[(nest + d).idx()].walkable()),
                "a nest at {nest:?} hems in a path"
            );
        }
    }

    /// However the noise falls, every bit of open ground is reachable from
    /// every other, and the agents start out of each other's sight.
    #[test]
    fn reset_joins_the_open_ground_and_spreads_the_agents() {
        let mut env = Survival::new(&SurvivalConfig::default(), 512);
        let mut buffers = TimeStepBuffers::new(&env);
        for seed in 0..10 {
            env.reset(seed, &mut buffers.view_mut());

            let walkable = env.state.base_map.map(|tile| tile.walkable());
            let (_, sizes) = crate::envs::common::map_gen::label_regions(&walkable);
            let regions = sizes.iter().filter(|&&size| size > 0).count();
            assert_eq!(
                regions, 1,
                "seed {seed} left the ground in {regions} pieces"
            );

            let apart = env.pad_width + 1;
            for (i, a) in env.state.agents.iter().enumerate() {
                for b in &env.state.agents[i + 1..] {
                    let gap = (a.position.x - b.position.x)
                        .abs()
                        .max((a.position.y - b.position.y).abs());
                    assert!(gap >= apart, "seed {seed}: agents {gap} apart");
                }
            }
        }
    }

    /// A move into something solid turns the agent to face it without
    /// moving, which is how an agent lines up on a tree or a bush. Moves are
    /// never masked for that reason.
    #[test]
    fn a_blocked_move_turns_the_agent_in_place() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            ..Default::default()
        });
        let start = center(&env);
        env.set_ground(start + RIGHT, TileTree);
        spawn_facing(&mut env, start, 0);
        let mut buffers = TimeStepBuffers::new(&env);

        step(&mut env, &mut buffers, &[MoveRight]);
        assert_eq!(env.state.agents[0].position, start);
        assert_eq!(env.state.map[start.idx()], AgentRight);
        for action in [MoveUp, MoveRight, MoveDown, MoveLeft] {
            assert!(legal(&buffers, 0, action));
        }

        step(&mut env, &mut buffers, &[MoveUp]);
        assert_eq!(env.state.agents[0].position, start + UP);
        assert_eq!(env.state.map[(start + UP).idx()], AgentUp);
        assert_eq!(env.state.map[start.idx()], TileEmpty);
    }

    /// Items lie underfoot: an agent walks onto one, hiding it from every
    /// view while it stands there, and it shows again once the agent steps
    /// off. Standing on it doesn't reach it; only the tile in front does.
    #[test]
    fn an_agent_walks_over_an_item_and_hides_it() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            ..Default::default()
        });
        let start = center(&env);
        env.set_ground(start + UP, ItemStick);
        spawn_facing(&mut env, start, 0);
        let mut buffers = TimeStepBuffers::new(&env);

        step(&mut env, &mut buffers, &[MoveUp]);
        assert_eq!(env.state.agents[0].position, start + UP);
        assert_eq!(env.state.map[(start + UP).idx()], AgentUp);
        assert!(!legal(&buffers, 0, Grab), "the stick is underfoot");

        step(&mut env, &mut buffers, &[MoveUp]);
        assert_eq!(env.state.map[(start + UP).idx()], ItemStick);
    }

    /// Grab takes the item in front into empty hands and leaves floor
    /// behind; drop puts it back down in front. Collecting counts once per
    /// life however often the agent picks the same kind of thing up.
    #[test]
    fn grab_and_drop_work_on_the_tile_in_front() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            ..Default::default()
        });
        let start = center(&env);
        env.set_ground(start + UP, ItemStone);
        spawn_facing(&mut env, start, 0);

        let buffers = observe(&env);
        assert!(legal(&buffers, 0, Grab));
        assert!(!legal(&buffers, 0, Drop), "nothing in hand to drop");

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Grab]);
        assert_eq!(env.state.agents[0].hands, Some(Item::Stone));
        assert_eq!(env.state.map[(start + UP).idx()], TileEmpty);
        assert!(!legal(&buffers, 0, Grab), "nothing in front to grab");
        assert!(legal(&buffers, 0, Drop));

        step(&mut env, &mut buffers, &[Drop]);
        assert_eq!(env.state.agents[0].hands, None);
        assert_eq!(env.state.map[(start + UP).idx()], ItemStone);

        step(&mut env, &mut buffers, &[Grab]);
        assert_eq!(achieved(&env, Achievement::CollectStone), 1.0);
    }

    /// Only open ground takes a dropped item: not a tree, an agent, or
    /// another item.
    #[test]
    fn drop_needs_open_ground_in_front() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 2,
            ..Default::default()
        });
        let start = center(&env);
        env.set_ground(start + UP, TileTree);
        spawn_facing(&mut env, start, 0);
        spawn_facing(&mut env, start + RIGHT * 2, 3);
        env.set_ground(start + RIGHT, ItemStick);
        agent(&mut env, 0).hands = Some(Item::Stone);
        agent(&mut env, 1).hands = Some(Item::Stone);

        let buffers = observe(&env);
        assert!(!legal(&buffers, 0, Drop), "a tree is in front");
        assert!(!legal(&buffers, 1, Drop), "a stick is in front");
    }

    /// Two agents reaching for the same item: whoever goes first in the turn
    /// order gets it, and the other's grab does nothing.
    #[test]
    fn two_agents_cannot_take_the_same_item() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 2,
            ..Default::default()
        });
        let start = center(&env);
        env.set_ground(start + RIGHT, ItemStick);
        spawn_facing(&mut env, start, 1);
        spawn_facing(&mut env, start + RIGHT * 2, 3);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Grab, Grab]);

        let holding: Vec<_> = env.state.agents.iter().map(|a| a.hands).collect();
        assert!(
            holding == [Some(Item::Stick), None] || holding == [None, Some(Item::Stick)],
            "{holding:?}"
        );
        assert_eq!(env.state.map[(start + RIGHT).idx()], TileEmpty);
    }

    #[test]
    fn swap_trades_hands_and_backpack() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 0);
        assert!(!legal(&observe(&env), 0, Swap), "both slots are empty");

        agent(&mut env, 0).hands = Some(Item::Stick);
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Swap]);
        let survivor = env.state.agents[0];
        assert_eq!(
            (survivor.hands, survivor.backpack),
            (None, Some(Item::Stick))
        );
    }

    /// Recipes work with either ingredient in hand, put the result in hand,
    /// and empty the backpack. Combine is masked for pairs with no recipe.
    #[test]
    fn combine_follows_the_recipes_either_way_round() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 0);
        let mut buffers = TimeStepBuffers::new(&env);

        for (hands, backpack, made) in [
            (Item::Stick, Item::Stone, Item::Axe),
            (Item::Stone, Item::Stick, Item::Axe),
            (Item::Wood, Item::Stone, Item::Campfire),
            (Item::Stone, Item::Wood, Item::Campfire),
        ] {
            agent(&mut env, 0).hands = Some(hands);
            agent(&mut env, 0).backpack = Some(backpack);
            step(&mut env, &mut buffers, &[Combine]);
            let survivor = env.state.agents[0];
            assert_eq!((survivor.hands, survivor.backpack), (Some(made), None));
        }
        assert_eq!(achieved(&env, Achievement::MakeAxe), 1.0);
        assert_eq!(achieved(&env, Achievement::MakeCampfire), 1.0);

        agent(&mut env, 0).hands = Some(Item::Stick);
        agent(&mut env, 0).backpack = Some(Item::Stick);
        assert!(!legal(&observe(&env), 0, Combine));
    }

    /// Using the axe on a tree locks the agent in place, able only to wait,
    /// and after `chop_steps` in all a log lies where the tree stood. The axe
    /// stays in hand.
    #[test]
    fn chopping_holds_the_agent_until_the_log_drops() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            chop_steps: 4,
            ..Default::default()
        });
        let start = center(&env);
        let tree = start + UP;
        spawn_facing(&mut env, start, 0);
        agent(&mut env, 0).hands = Some(Item::Axe);
        assert!(!legal(&observe(&env), 0, Use), "no tree in front");

        env.set_ground(tree, TileTree);
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Use]);
        for _ in 0..2 {
            assert_eq!(env.state.map[tree.idx()], TileTree);
            let mask = buffers.action_mask.row(0);
            assert_eq!(mask.iter().filter(|&&legal| legal).count(), 1);
            assert!(mask[Noop as usize], "only waiting while at work");
            step(&mut env, &mut buffers, &[Noop]);
        }
        assert_eq!(env.state.map[tree.idx()], TileTree);

        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.map[tree.idx()], ItemWood);
        assert_eq!(env.state.agents[0].hands, Some(Item::Axe));
        assert_eq!(achieved(&env, Achievement::ChopTree), 1.0);
        assert!(legal(&buffers, 0, MoveDown), "free again");
    }

    /// Whatever a busy agent asks for, its turn goes into the job.
    #[test]
    fn a_busy_agent_only_works() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            chop_steps: 3,
            ..Default::default()
        });
        let start = center(&env);
        env.set_ground(start + UP, TileTree);
        spawn_facing(&mut env, start, 0);
        agent(&mut env, 0).hands = Some(Item::Axe);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Use]);
        step(&mut env, &mut buffers, &[MoveDown]);
        assert_eq!(env.state.agents[0].position, start);
        step(&mut env, &mut buffers, &[Swap]);
        assert_eq!(env.state.agents[0].hands, Some(Item::Axe));
        assert_eq!(env.state.map[(start + UP).idx()], ItemWood);
    }

    /// Two agents felling the same tree get one log between them: whoever
    /// finishes second finds the tree gone and gets nothing.
    #[test]
    fn two_agents_felling_one_tree_get_one_log() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 2,
            chop_steps: 2,
            ..Default::default()
        });
        let start = center(&env);
        let tree = start + RIGHT;
        env.set_ground(tree, TileTree);
        spawn_facing(&mut env, start, 1);
        spawn_facing(&mut env, start + RIGHT * 2, 3);
        agent(&mut env, 0).hands = Some(Item::Axe);
        agent(&mut env, 1).hands = Some(Item::Axe);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Use, Use]);
        step(&mut env, &mut buffers, &[Noop, Noop]);
        assert_eq!(env.state.map[tree.idx()], ItemWood);
        assert!(env.state.agents.iter().all(|a| a.work.is_none()));
        assert_eq!(achieved(&env, Achievement::ChopTree), 1.0);
    }

    /// A campfire is set down lit in front, and goes out after
    /// `fire_burn_steps`.
    #[test]
    fn a_campfire_is_set_down_lit_and_burns_out() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            fire_burn_steps: 3,
            fire_low_steps: 0,
            ..Default::default()
        });
        let start = center(&env);
        let fire = start + UP;
        spawn_facing(&mut env, start, 0);
        agent(&mut env, 0).hands = Some(Item::Campfire);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Use]);
        assert_eq!(env.state.map[fire.idx()], TileFire);
        assert_eq!(env.state.agents[0].hands, None);
        assert_eq!(achieved(&env, Achievement::PlaceFire), 1.0);

        step(&mut env, &mut buffers, &[Noop]);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.map[fire.idx()], TileFire);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.map[fire.idx()], TileEmpty);
    }

    /// Using a raw berry on the fire in front cooks it in hand at once.
    /// Away from a fire a berry has no use, and a cooked one none at all.
    #[test]
    fn a_berry_held_to_a_fire_cooks_at_once() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 0);
        agent(&mut env, 0).hands = Some(Item::Berry);
        assert!(!legal(&observe(&env), 0, Use), "no fire in front");

        light_fire(&mut env, start + UP);
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Use]);

        assert_eq!(env.state.agents[0].hands, Some(Item::CookedBerry));
        assert_eq!(achieved(&env, Achievement::CookBerry), 1.0);
        assert!(!legal(&buffers, 0, Use), "a cooked berry has no use");
    }

    /// Grabbing at a ripe bush picks a berry and leaves it bare, and a bare
    /// bush can't be picked again until it regrows.
    #[test]
    fn bushes_give_berries_and_regrow() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            bush_regrow_steps: 3,
            ..Default::default()
        });
        let start = center(&env);
        let bush = start + UP;
        env.set_ground(bush, TileBerryBush);
        spawn_facing(&mut env, start, 0);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Grab]);
        assert_eq!(env.state.agents[0].hands, Some(Item::Berry));
        assert_eq!(env.state.map[bush.idx()], TileBush);
        assert_eq!(achieved(&env, Achievement::CollectBerry), 1.0);

        step(&mut env, &mut buffers, &[Swap]);
        assert!(!legal(&buffers, 0, Grab), "the bush is bare");
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.map[bush.idx()], TileBush);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.map[bush.idx()], TileBerryBush);
        assert!(legal(&buffers, 0, Grab));
    }

    /// A fire burns low for its last `fire_low_steps`, and wood used on a low
    /// fire stokes it back up to full. Wood does nothing for a fire burning
    /// high, and a berry cooks on a low fire as on any other.
    #[test]
    fn a_fire_burns_low_and_wood_stokes_it() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            fire_burn_steps: 4,
            fire_low_steps: 2,
            ..Default::default()
        });
        let start = center(&env);
        let fire = start + UP;
        spawn_facing(&mut env, start, 0);
        agent(&mut env, 0).hands = Some(Item::Campfire);

        // shows for four steps, the last two low
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Use]);
        agent(&mut env, 0).hands = Some(Item::Wood);
        let mut burning = vec![env.state.map[fire.idx()]];
        for _ in 0..3 {
            let low = burning.last() == Some(&TileFireLow);
            assert_eq!(
                legal(&observe(&env), 0, Use),
                low,
                "wood stokes only a low fire"
            );
            step(&mut env, &mut buffers, &[Noop]);
            burning.push(env.state.map[fire.idx()]);
        }
        assert_eq!(burning, [TileFire, TileFire, TileFireLow, TileFireLow]);
        assert!(legal(&buffers, 0, Use));

        step(&mut env, &mut buffers, &[Use]);
        assert_eq!(env.state.map[fire.idx()], TileFire);
        assert_eq!(env.state.fires[0].burn_left, 3);
        assert_eq!(env.state.agents[0].hands, None);
        assert_eq!(achieved(&env, Achievement::RefuelFire), 1.0);

        step(&mut env, &mut buffers, &[Noop]);
        step(&mut env, &mut buffers, &[Noop]);
        agent(&mut env, 0).hands = Some(Item::Berry);
        assert_eq!(env.state.map[fire.idx()], TileFireLow);
        assert!(legal(&observe(&env), 0, Use), "a low fire still cooks");
    }

    /// The clock in the band shows the sun through `day_length` steps and
    /// the moon through `night_length`, then the sun again.
    #[test]
    fn night_follows_day_on_the_clock() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            day_length: 2,
            dusk_length: 0,
            night_length: 3,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 0);
        let slots_row = env.config.view_height as usize;
        let clock = move |buffers: &TimeStepBuffers| {
            SurvivalObs::from_id(buffers.obs[[0, CLOCK_COL, slots_row, 0]])
        };

        let mut buffers = observe(&env);
        let mut seen = vec![clock(&buffers)];
        for _ in 0..6 {
            step(&mut env, &mut buffers, &[Noop]);
            seen.push(clock(&buffers));
        }
        assert_eq!(
            seen,
            [UiDay, UiDay, UiNight, UiNight, UiNight, UiDay, UiDay]
        );
    }

    /// By night the agent sees only what is near it, and lit ground as far as
    /// by day: a fire in the distance shows, with what is around it.
    #[test]
    fn at_night_only_the_near_and_the_lit_show() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            day_length: 1,
            dusk_length: 0,
            night_length: 10,
            night_vision_radius: 1,
            fire_light_radius: 1,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 0);
        light_fire(&mut env, start + RIGHT * 5);

        let by_day = observe(&env);
        assert_eq!(seen(&env, &by_day, 0, RIGHT * 3), TileEmpty);
        assert_eq!(seen(&env, &by_day, 0, UP * 5), TileEmpty);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(seen(&env, &buffers, 0, RIGHT), TileEmpty, "near");
        assert_eq!(seen(&env, &buffers, 0, RIGHT * 3), Mask, "dark");
        assert_eq!(seen(&env, &buffers, 0, RIGHT * 4), TileEmpty, "lit");
        assert_eq!(seen(&env, &buffers, 0, RIGHT * 5), TileFire);
        assert_eq!(
            seen(&env, &buffers, 0, RIGHT * 5 + UP * 2),
            Mask,
            "past the light"
        );
        assert_eq!(seen(&env, &buffers, 0, UP * 5), Mask);
    }

    /// Through dusk, sight closes in a little every step, from the whole view
    /// down to the night's radius.
    #[test]
    fn dusk_closes_in_step_by_step() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            day_length: 10,
            dusk_length: 4,
            night_length: 5,
            night_vision_radius: 1,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 0);
        let fov = env.config.view_height as usize;
        let visible = |buffers: &TimeStepBuffers| {
            let view = buffers.obs.slice(s![0, .., ..fov, 0]);
            view.iter().filter(|&&id| id != VocabId::from(Mask)).count()
        };

        let mut buffers = observe(&env);
        let mut counts = vec![visible(&buffers)];
        for _ in 0..11 {
            step(&mut env, &mut buffers, &[Noop]);
            counts.push(visible(&buffers));
        }

        let whole = (env.config.view_width as usize) * fov;
        assert_eq!(counts[..6], [whole; 6], "full day");
        assert!(counts[5..11].is_sorted_by(|a, b| a > b), "dusk: {counts:?}");
        assert_eq!(counts[10], counts[11], "night holds steady");
        assert_eq!(
            counts[10], 9,
            "the night's radius-1 disc, rounded out to 3x3"
        );
    }

    /// Nests hatch a spider each at nightfall, beside them; the spiders hunt
    /// through the night and are home again soon after dawn, biting no one on
    /// the way.
    #[test]
    fn spiders_hatch_at_nightfall_and_go_home_at_dawn() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            day_length: 4,
            dusk_length: 0,
            night_length: 2,
            ..Default::default()
        });
        let start = center(&env);
        let nest = start + UP * 3;
        env.set_ground(nest, TileSpiderEggs);
        env.state.eggs.push(nest);
        spawn_facing(&mut env, start + DOWN * 4, 0);

        let mut buffers = TimeStepBuffers::new(&env);
        for _ in 0..3 {
            step(&mut env, &mut buffers, &[Noop]);
        }
        assert_eq!(spiders_on_map(&env), 0, "still day");

        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.spiders.len(), 1, "nightfall");
        assert_eq!(spiders_on_map(&env), 1);
        assert_eq!(env.state.spiders[0].nest, nest);

        // the night's second step, then dawn and the walk home
        let mut steps_out = 0;
        while !env.state.spiders.is_empty() {
            step(&mut env, &mut buffers, &[Noop]);
            steps_out += 1;
            assert!(steps_out < 6, "the spider never got home");
        }
        assert_eq!(spiders_on_map(&env), 0);
        assert_eq!(
            env.state.base_map[nest.idx()],
            TileSpiderEggs,
            "the nest stays"
        );
        assert_eq!(env.metrics.spider_bites, 0.0);
    }

    /// By day a spider takes the shortest way home, one cell closer every
    /// step, and burrows in on the step after it reaches the nest.
    #[test]
    fn a_spider_walks_the_shortest_way_home() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            day_length: 50,
            dusk_length: 0,
            ..Default::default()
        });
        let start = center(&env);
        let nest = start + UP * 5;
        env.set_ground(nest, TileSpiderEggs);
        env.state.eggs.push(nest);
        spawn_facing(&mut env, start + RIGHT * 6, 0);
        release_spider(&mut env, start + DOWN * 3, nest);

        let mut buffers = TimeStepBuffers::new(&env);
        let mut distances = Vec::new();
        while let Some(spider) = env.state.spiders.first() {
            let gap = spider.position - nest;
            distances.push(gap.x.abs() + gap.y.abs());
            step(&mut env, &mut buffers, &[Noop]);
        }
        assert_eq!(distances, [8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(spiders_on_map(&env), 0);
    }

    /// A spider caught by a fire lit beside it walks out of the light by the
    /// shortest way, and stays out.
    #[test]
    fn a_spider_in_the_light_walks_out_of_it() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            day_length: 1,
            dusk_length: 0,
            night_length: 50,
            fire_light_radius: 2,
            // nothing to track, so it only has the light to get away from
            spider_hunt_radius: 0,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start + LEFT * 8, 0);
        release_spider(&mut env, start, start + UP * 8);
        light_fire(&mut env, start + RIGHT);
        assert!(env.state.lit[start.idx()]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.spiders[0].position, start + LEFT);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.spiders[0].position, start + LEFT * 2);
        for _ in 0..10 {
            assert!(!env.state.lit[env.state.spiders[0].position.idx()]);
            step(&mut env, &mut buffers, &[Noop]);
        }
    }

    /// A nest whose spider is still out doesn't hatch another.
    #[test]
    fn a_nest_waits_for_its_spider() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            day_length: 1,
            dusk_length: 0,
            night_length: 5,
            ..Default::default()
        });
        let start = center(&env);
        let nest = start + UP * 3;
        env.set_ground(nest, TileSpiderEggs);
        env.state.eggs.push(nest);
        spawn_facing(&mut env, start + LEFT * 8, 0);
        release_spider(&mut env, start + DOWN * 5, nest);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.spiders.len(), 1);
    }

    /// A spider walks up to an agent in the dark and bites it. Once the agent
    /// stands in firelight, the spider can neither bite it nor step closer.
    #[test]
    fn spiders_hunt_in_the_dark_but_fear_fire() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            day_length: 1,
            dusk_length: 0,
            night_length: 50,
            start_hunger: 50,
            start_health: 100,
            spider_damage: 10,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 0);
        release_spider(&mut env, start + RIGHT * 3, start + UP * 8);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.spiders[0].position, start + RIGHT * 2);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.spiders[0].position, start + RIGHT);
        assert_eq!(env.state.agents[0].health, 100);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.agents[0].health, 90, "bitten");
        assert_eq!(env.metrics.spider_bites, 1.0);

        light_fire(&mut env, start + DOWN);
        for _ in 0..3 {
            step(&mut env, &mut buffers, &[Noop]);
        }
        assert_eq!(env.state.agents[0].health, 90, "safe in the light");
    }

    /// Spiders wander the dark but never set foot on lit ground, and can't
    /// track an agent standing in it.
    #[test]
    fn spiders_never_step_into_the_light() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            day_length: 1,
            dusk_length: 0,
            night_length: 100,
            fire_light_radius: 2,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 0);
        light_fire(&mut env, start + DOWN);
        release_spider(&mut env, start + RIGHT * 5, start + UP * 8);
        release_spider(&mut env, start + UP * 5, start + UP * 8);

        let mut buffers = TimeStepBuffers::new(&env);
        for _ in 0..40 {
            step(&mut env, &mut buffers, &[Noop]);
            for spider in &env.state.spiders {
                assert!(
                    !env.state.lit[spider.position.idx()],
                    "a spider at {:?} is in the light",
                    spider.position
                );
            }
        }
        assert_eq!(env.metrics.spider_bites, 0.0);
    }

    /// Eating uses the berry up and fills hunger, never past the cap. Only
    /// food can be eaten.
    #[test]
    fn eating_restores_hunger_up_to_the_cap() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            start_hunger: 10,
            // no drain to muddle the sums
            hunger_interval: 1000,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 0);
        assert!(!legal(&observe(&env), 0, Eat), "nothing in hand");
        agent(&mut env, 0).hands = Some(Item::Stick);
        assert!(!legal(&observe(&env), 0, Eat), "a stick is no food");
        agent(&mut env, 0).hands = Some(Item::Berry);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Eat]);
        assert_eq!(env.state.agents[0].hunger, 10 + env.config.berry_food);
        assert_eq!(env.state.agents[0].hands, None);
        assert_eq!(achieved(&env, Achievement::EatBerry), 1.0);

        agent(&mut env, 0).hunger = MAX_STAT - 5;
        agent(&mut env, 0).hands = Some(Item::CookedBerry);
        step(&mut env, &mut buffers, &[Eat]);
        assert_eq!(env.state.agents[0].hunger, MAX_STAT);
        assert_eq!(achieved(&env, Achievement::EatCookedBerry), 1.0);
    }

    /// Hunger drains one point per `hunger_interval` steps, and while it
    /// stays high, health grows back one per `regen_interval`.
    #[test]
    fn hunger_drains_and_a_fed_agent_heals() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            start_hunger: 120,
            hunger_interval: 4,
            start_health: 50,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 0);

        let mut buffers = TimeStepBuffers::new(&env);
        for _ in 0..8 {
            step(&mut env, &mut buffers, &[Noop]);
        }
        assert_eq!(env.state.agents[0].hunger, 118);
        assert_eq!(env.state.agents[0].health, 52);
    }

    /// At zero hunger health drains; at zero health the agent is flagged
    /// terminated, leaves what it carried where it stood, and respawns
    /// elsewhere with fresh stats and achievements.
    #[test]
    fn starving_agents_die_drop_their_things_and_respawn() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            start_hunger: 0,
            start_health: 2,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 0);
        let survivor = agent(&mut env, 0);
        survivor.hands = Some(Item::Stick);
        survivor.backpack = Some(Item::Stone);
        survivor.unlocked = !0;

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.agents[0].health, 1);
        assert!(!buffers.terminated[0]);

        step(&mut env, &mut buffers, &[Noop]);
        assert!(buffers.terminated[0]);
        assert_eq!(env.state.map[start.idx()], ItemStick);
        let dropped = DIRECTIONS
            .iter()
            .filter(|&&d| env.state.map[(start + d).idx()] == ItemStone)
            .count();
        assert_eq!(dropped, 1);

        let respawned = env.state.agents[0];
        assert_ne!(respawned.position, start);
        assert_eq!(env.state.map[respawned.position.idx()], respawned.tile());
        assert_eq!((respawned.health, respawned.hunger), (2, 0));
        assert_eq!((respawned.hands, respawned.backpack), (None, None));
        assert_eq!(respawned.unlocked, 0);

        let metrics = env.consume_metrics();
        assert_eq!(metrics["deaths"], 1.0);
        assert_eq!(metrics["achievements"]["make_axe"], 0.0);
    }

    /// The band's top row reads health and hunger as numbers after their
    /// labels; the row under it, the hands and backpack items after theirs,
    /// then the time of day.
    /// The view's centre is the agent, facing the way it faces.
    #[test]
    fn the_ui_band_shows_stats_and_inventory() {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 1);
        let survivor = agent(&mut env, 0);
        survivor.health = 150;
        survivor.hunger = 7;
        survivor.hands = Some(Item::Axe);

        let buffers = observe(&env);
        let obs = buffers.obs.slice(s![0, .., .., 0]);
        let fov = env.config.view_height as usize;
        let width = env.config.view_width as usize;

        let blank = |n: usize| std::iter::repeat_n(id(UI), n);
        let stats: Vec<VocabId> = [
            UiHealth, Digit1, Digit5, Digit0, UI, UiHunger, UI, UI, Digit7,
        ]
        .into_iter()
        .map(id)
        .chain(blank(width - 9))
        .collect();
        assert_eq!(obs.column(fov + 1).to_vec(), stats);

        let slots: Vec<VocabId> = [UiHands, ItemAxe, UI, UiBackpack, UI, UI, UiDay]
            .into_iter()
            .map(id)
            .chain(blank(width - 7))
            .collect();
        assert_eq!(obs.column(fov).to_vec(), slots);

        assert_eq!(obs[[width / 2, fov / 2]], id(AgentRight));
    }
}
