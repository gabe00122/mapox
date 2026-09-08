//! Port of the PufferLib pacman demo: the classic maze, four ghosts with
//! their scatter/chase/frightened AI, dots and power pellets.
//!
//! Two things differ from the C original on purpose. The observation is an
//! egocentric crop of the maze centred on the player — walls, dots, pellets
//! and ghosts as tiles — instead of the absolute-coordinate feature vector
//! plus a global dot bitmap; the view is the whole window around the agent,
//! with no line-of-sight masking. And the player is the only agent: ghosts
//! are driven by the maze's own AI, exactly as in the original.
//!
//! The maze is fixed (28x31, the original arcade layout) and wraps
//! horizontally through the tunnel; the x padding mirrors the far side of
//! the map so a player at either mouth sees what the other side really
//! holds. Like the other envs, y points up, so the ASCII rows below read
//! top-down the way the maze looks on screen and are flipped when indexed.

use ndarray::{Array2, s};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::common::{Position, vocab_enum::VocabEnum},
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_GHOST_BLINKY, AGENT_GHOST_CLYDE, AGENT_GHOST_EYES, AGENT_GHOST_FRIGHTENED,
        AGENT_GHOST_INKY, AGENT_GHOST_PINKY, AGENT_PACMAN, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT,
        MOVE_UP, TILE_EMPTY, TILE_FOOD, TILE_POWER, TILE_UI, TILE_WALL,
    },
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
    vocab_enum,
};

const MAP_WIDTH: i32 = 28;
const MAP_HEIGHT: i32 = 31;

/// The classic maze, rows in screen order (first row is the top of the
/// screen, i.e. the highest `y`). `#` wall, `.` dot, `x` power pellet,
/// `p` the player's spawn, `1`/`2`/`3`/`4` the Inky/Blinky/Pinky/Clyde
/// spawns; spaces are plain floor.
const ORIGINAL_MAP: [&str; MAP_HEIGHT as usize] = [
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
];

const NUM_GHOSTS: usize = 4;
/// Indexes into the ghost array; the order matches the C original.
const PINKY: usize = 0;
const BLINKY: usize = 1;
const INKY: usize = 2;
const CLYDE: usize = 3;

/// Each ghost's scatter corner, in screen coordinates (`y` down, may sit
/// off the map); converted to the env's padded y-up frame in [`Pacman::new`].
const GHOST_CORNERS_DOWN: [(i32, i32); NUM_GHOSTS] = [
    (3, -3),
    (MAP_WIDTH - 4, -3),
    (MAP_WIDTH - 1, MAP_HEIGHT),
    (0, MAP_HEIGHT),
];

const PINKY_TARGET_LEAD: i32 = 4;
const INKY_TARGET_LEAD: i32 = 2;
const CLYDE_TARGET_RADIUS: i32 = 8;

/// Heading i moves by `DIRECTIONS[i]`, in `MOVES` order: up, right, down,
/// left. [`REVERSED`] is the matching half-turn.
const DIRECTIONS: [Position; 4] = [
    Position { x: 0, y: 1 },
    Position { x: 1, y: 0 },
    Position { x: 0, y: -1 },
    Position { x: -1, y: 0 },
];

/// Copy the live map's edge columns into the x padding, so a window
/// cropped across the tunnel seam reads the far side as it is right now:
/// padding column `x` holds playable column `x + MAP_WIDTH`, and the right
/// padding mirrors the opening columns the same way one step further out.
fn mirror_x_padding(map: &mut Array2<PacmanObs>, pad_width: i32) {
    let height = map.dim().1 as i32;
    for x in 0..pad_width {
        for y in 0..height {
            let far = Position::new(x + MAP_WIDTH, y);
            map[[x as usize, y as usize]] = map[far.idx()];
            let near = Position::new(pad_width + x, y);
            map[(near + Position::new(MAP_WIDTH, 0)).idx()] = map[near.idx()];
        }
    }
}
const REVERSED: [usize; 4] = [2, 3, 0, 1];

fn distance_squared(from: Position, to: Position) -> i32 {
    let delta = from - to;
    delta.x * delta.x + delta.y * delta.y
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PacmanConfig {
    pub view_width: i32,
    pub view_height: i32,

    pub randomize_starting_position: bool,
    /// Randomised per-episode delay before each ghost starts moving.
    pub min_start_timeout: i32,
    pub max_start_timeout: i32,
    /// Steps the ghosts spend frightened after a power pellet.
    pub frightened_time: i32,
    /// After this many scatter/chase flips the ghosts scatter forever.
    pub max_mode_changes: i32,
    pub scatter_mode_length: i32,
    pub chase_mode_length: i32,

    pub dot_reward: f32,
    pub ghost_reward: f32,
}

impl Default for PacmanConfig {
    fn default() -> Self {
        Self {
            view_width: 11,
            view_height: 11,
            randomize_starting_position: false,
            min_start_timeout: 0,
            max_start_timeout: 49,
            frightened_time: 35,
            max_mode_changes: 6,
            scatter_mode_length: 700,
            chase_mode_length: 70,
            dot_reward: 1.0,
            ghost_reward: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GhostKind {
    Pinky,
    Blinky,
    Inky,
    Clyde,
}

impl GhostKind {
    const ALL: [GhostKind; NUM_GHOSTS] = [
        GhostKind::Pinky,
        GhostKind::Blinky,
        GhostKind::Inky,
        GhostKind::Clyde,
    ];

    /// The maze glyph marking this ghost's spawn.
    fn marker(self) -> char {
        match self {
            GhostKind::Pinky => '3',
            GhostKind::Blinky => '2',
            GhostKind::Inky => '1',
            GhostKind::Clyde => '4',
        }
    }

    fn tile(self) -> PacmanObs {
        match self {
            GhostKind::Pinky => PacmanObs::GhostPinky,
            GhostKind::Blinky => PacmanObs::GhostBlinky,
            GhostKind::Inky => PacmanObs::GhostInky,
            GhostKind::Clyde => PacmanObs::GhostClyde,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Ghost {
    kind: GhostKind,
    position: Position,
    spawn: Position,
    target: Position,
    /// Heading index into [`DIRECTIONS`].
    direction: usize,
    /// Counts down every step; the ghost stays parked while it is >= 0.
    start_timeout: i32,
    frightened: bool,
    return_to_spawn: bool,
    /// Frightened ghosts move on every other step; this is the coin flip.
    half_move: bool,
}

#[derive(Debug, Clone)]
struct PacmanState {
    rngs: SmallRng,
    time: usize,
    remaining_pickups: usize,

    player_position: Position,
    player_direction: usize,

    scatter_mode: bool,
    mode_time_left: i32,
    mode_changes: i32,
    frightened_time_left: i32,
    /// Set on a mode flip or a power pellet: ghosts reverse before hunting.
    reverse_directions: bool,
    player_caught: bool,

    ghosts: [Ghost; NUM_GHOSTS],

    base_map: Array2<PacmanObs>, // terrain and uneaten pickups
    map: Array2<PacmanObs>,      // the base plus every ghost and the player
}

vocab_enum!(PacmanObs {
    UI => TILE_UI,
    TileEmpty => TILE_EMPTY,
    TileWall => TILE_WALL,
    TileDot => TILE_FOOD,
    TilePower => TILE_POWER,
    AgentPacman => AGENT_PACMAN,
    GhostPinky => AGENT_GHOST_PINKY,
    GhostBlinky => AGENT_GHOST_BLINKY,
    GhostInky => AGENT_GHOST_INKY,
    GhostClyde => AGENT_GHOST_CLYDE,
    GhostFrightened => AGENT_GHOST_FRIGHTENED,
    GhostEyes => AGENT_GHOST_EYES,
});

impl PacmanObs {
    fn wall(self) -> bool {
        self == PacmanObs::TileWall
    }
}

vocab_enum!(PacmanAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
});

/// The classic maze with the classic ghost AI, observed from the player's
/// own square.
///
/// The player is the only agent: it is paid `dot_reward` for every dot or
/// power pellet eaten and `ghost_reward` for every frightened ghost, and it
/// acts on all four moves every step. Walking into a wall keeps the current
/// heading, so an action is a wish to turn, not a permission to step.
///
/// The episode ends when the player is caught, when the last pickup is
/// eaten, or at `length` steps; unlike the C demo there is no auto-reset —
/// the caller resets, as for every env here.
#[derive(Debug, Clone)]
pub struct Pacman {
    pub config: PacmanConfig,
    state: PacmanState,

    // max steps for a single episode
    length: usize,

    pad_width: i32,
    pad_height: i32,

    // full map size including padding on both sides
    width: i32,
    height: i32,
    /// The pristine maze: reset copies it back over the working map.
    original: Array2<PacmanObs>,
    player_spawn: Position,
    /// Where a dot sits at the start of the round; the randomised player
    /// spawn draws from these, as in the C original.
    dot_positions: Vec<Position>,
    /// Dots plus pellets on the pristine maze; the x padding mirrors
    /// pickups into it, so the count lives here rather than in a scan.
    pickup_count: usize,
    corners: [Position; NUM_GHOSTS],

    obs_spec: ObservationSpec,
    action_spec: ActionSpec,

    obs_vocab: Vocabulary,
    action_vocab: Vocabulary,
}

impl Pacman {
    pub fn new(config: &PacmanConfig, length: usize) -> Self {
        assert!(
            config.min_start_timeout <= config.max_start_timeout,
            "empty ghost delay range {}..{}",
            config.min_start_timeout,
            config.max_start_timeout,
        );
        let pad_width = config.view_width / 2;
        assert!(
            pad_width < MAP_WIDTH,
            "a {}-column view needs more x padding than the maze is wide",
            config.view_width,
        );
        let pad_height = config.view_height / 2;

        let width = MAP_WIDTH + 2 * pad_width;
        let height = MAP_HEIGHT + 2 * pad_height;
        let dim = (width as usize, height as usize);

        let mut original = Array2::from_elem(dim, PacmanObs::TileWall);
        let mut player_spawn = Position::default();
        let mut ghost_spawns = [Position::default(); NUM_GHOSTS];
        let mut dot_positions = Vec::new();
        let mut power_count = 0usize;

        for (row, line) in ORIGINAL_MAP.iter().enumerate() {
            // screen row 0 is the top of the maze, i.e. the highest y
            let y = pad_height + (MAP_HEIGHT - 1 - row as i32);
            for (x, glyph) in line.chars().enumerate() {
                let position = Position::new(pad_width + x as i32, y);
                let tile = match glyph {
                    '#' => PacmanObs::TileWall,
                    '.' => {
                        dot_positions.push(position);
                        PacmanObs::TileDot
                    }
                    'x' => {
                        power_count += 1;
                        PacmanObs::TilePower
                    }
                    'p' => {
                        player_spawn = position;
                        PacmanObs::TileEmpty
                    }
                    _ => PacmanObs::TileEmpty,
                };
                original[position.idx()] = tile;

                for (i, kind) in GhostKind::ALL.iter().enumerate() {
                    if glyph == kind.marker() {
                        ghost_spawns[i] = position;
                    }
                }
            }
        }
        let pickup_count = dot_positions.len() + power_count;

        // Seed the padding from the pristine maze; repaint keeps it in
        // step with the live map, so the seam never shows stale pickups.
        mirror_x_padding(&mut original, pad_width);

        let corners = GHOST_CORNERS_DOWN.map(|(x, y_down)| {
            Position::new(pad_width + x, pad_height + (MAP_HEIGHT - 1 - y_down))
        });

        let action_vocab = PacmanAction::vocab();
        let obs_vocab = PacmanObs::vocab();

        let view_height = config.view_height + 2;
        let obs_spec = ObservationSpec::new(config.view_width, view_height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        let base_map = original.clone();
        let map = base_map.clone();

        Self {
            config: config.clone(),
            state: PacmanState {
                rngs: SmallRng::seed_from_u64(0),
                time: 0,
                remaining_pickups: 0,
                player_position: player_spawn,
                player_direction: PacmanAction::MoveRight as usize,
                scatter_mode: false,
                mode_time_left: 0,
                mode_changes: 0,
                frightened_time_left: 0,
                reverse_directions: false,
                player_caught: false,
                ghosts: std::array::from_fn(|i| Ghost {
                    kind: GhostKind::ALL[i],
                    position: ghost_spawns[i],
                    spawn: ghost_spawns[i],
                    target: ghost_spawns[i],
                    direction: PacmanAction::MoveUp as usize,
                    start_timeout: 0,
                    frightened: false,
                    return_to_spawn: false,
                    half_move: false,
                }),
                base_map,
                map,
            },
            length,

            pad_width,
            pad_height,
            width,
            height,

            original,
            player_spawn,
            dot_positions,
            pickup_count,
            corners,

            obs_spec,
            action_spec,

            obs_vocab,
            action_vocab,
        }
    }

    /// Folds a position that stepped past a tunnel mouth back onto the map.
    /// A single step never overshoots by more than one column.
    fn wrap(&self, position: Position) -> Position {
        let mut x = position.x;
        if x < self.pad_width {
            x += MAP_WIDTH;
        } else if x >= self.pad_width + MAP_WIDTH {
            x -= MAP_WIDTH;
        }
        Position::new(x, position.y)
    }

    fn tile_at(&self, position: Position) -> PacmanObs {
        self.state.base_map[position.idx()]
    }

    fn is_wall(&self, position: Position) -> bool {
        self.tile_at(position).wall()
    }

    fn can_move(&self, position: Position, direction: usize) -> bool {
        !self.is_wall(self.wrap(position + DIRECTIONS[direction]))
    }

    /// The delay range is inclusive of `min` and exclusive of `max`, as in
    /// the C original — but drawn from the env's own rng, not global state.
    fn rand_range(&mut self, min: i32, max: i32) -> i32 {
        if min == max {
            min
        } else {
            min + self.state.rngs.random_range(0..max - min)
        }
    }

    fn reset_round(&mut self) {
        self.state.time = 0;
        self.state.scatter_mode = false;
        self.state.mode_time_left = 0;
        self.state.mode_changes = 0;
        self.state.frightened_time_left = 0;
        self.state.reverse_directions = false;
        self.state.player_caught = false;
        self.state.base_map.assign(&self.original);
        self.state.remaining_pickups = self.pickup_count;

        let (min, max) = (self.config.min_start_timeout, self.config.max_start_timeout);
        let timeouts: Vec<i32> = (0..NUM_GHOSTS).map(|_| self.rand_range(min, max)).collect();
        for (ghost, start_timeout) in self.state.ghosts.iter_mut().zip(timeouts) {
            ghost.position = ghost.spawn;
            ghost.target = ghost.spawn;
            ghost.direction = PacmanAction::MoveUp as usize;
            ghost.start_timeout = start_timeout;
            ghost.frightened = false;
            ghost.return_to_spawn = false;
            ghost.half_move = false;
        }

        if self.config.randomize_starting_position {
            let pick = self.state.rngs.random_range(0..self.dot_positions.len());
            self.state.player_position = self.dot_positions[pick];
        } else {
            self.state.player_position = self.player_spawn;
        }
        self.state.player_direction = PacmanAction::MoveRight as usize;
    }

    /// Repaint the top layer: terrain plus ghosts, plus the player on top.
    /// The x padding is re-mirrored last, so the tunnel seam always shows
    /// the live map — eaten dots gone, and an agent stood at one mouth
    /// visible from the other.
    fn repaint(&mut self) {
        self.state.map.assign(&self.state.base_map);
        for ghost in self.state.ghosts.iter() {
            let tile = if ghost.return_to_spawn {
                PacmanObs::GhostEyes
            } else if ghost.frightened {
                PacmanObs::GhostFrightened
            } else {
                ghost.kind.tile()
            };
            self.state.map[ghost.position.idx()] = tile;
        }
        self.state.map[self.state.player_position.idx()] = PacmanObs::AgentPacman;
        mirror_x_padding(&mut self.state.map, self.pad_width);
    }

    fn set_frightened(&mut self) {
        self.state.frightened_time_left = self.config.frightened_time;
        self.state.reverse_directions = true;
        for ghost in self.state.ghosts.iter_mut() {
            // eyes heading home stay eyes
            ghost.frightened = !ghost.return_to_spawn;
        }
    }

    /// The player acts on its wish: turn into a wall and it keeps sliding
    /// along its current heading instead. Eating is a side effect of the
    /// square it lands on. Returns the reward collected this step.
    fn player_move(&mut self, action: usize) -> f32 {
        let mut target = self.wrap(self.state.player_position + DIRECTIONS[action]);
        if self.is_wall(target) {
            target =
                self.wrap(self.state.player_position + DIRECTIONS[self.state.player_direction]);
        } else {
            self.state.player_direction = action;
        }

        if self.is_wall(target) {
            return 0.0;
        }
        self.state.player_position = target;

        match self.tile_at(target) {
            PacmanObs::TilePower => {
                self.set_frightened();
                self.eat(target)
            }
            PacmanObs::TileDot => self.eat(target),
            _ => 0.0,
        }
    }

    fn eat(&mut self, position: Position) -> f32 {
        self.state.base_map[position.idx()] = PacmanObs::TileEmpty;
        self.state.remaining_pickups -= 1;
        self.config.dot_reward
    }

    fn check_mode_change(&mut self) {
        if self.state.mode_changes > self.config.max_mode_changes {
            return;
        }

        self.state.mode_time_left -= 1;
        if self.state.mode_time_left <= 0 {
            self.state.scatter_mode = !self.state.scatter_mode;
            self.state.reverse_directions = true;
            self.state.mode_changes += 1;

            self.state.mode_time_left = if self.state.scatter_mode {
                self.config.scatter_mode_length
            } else {
                self.config.chase_mode_length
            };
        }
    }

    fn set_targets(&mut self) {
        if self.state.scatter_mode {
            for (i, ghost) in self.state.ghosts.iter_mut().enumerate() {
                ghost.target = self.corners[i];
            }
            return;
        }

        let player = self.state.player_position;
        let heading = DIRECTIONS[self.state.player_direction];
        let blinky = self.state.ghosts[BLINKY].position;
        let clyde = self.state.ghosts[CLYDE].position;

        self.state.ghosts[PINKY].target = player + heading * PINKY_TARGET_LEAD;
        self.state.ghosts[BLINKY].target = player;
        self.state.ghosts[INKY].target = {
            let lead = player + heading * INKY_TARGET_LEAD;
            Position::new(2 * lead.x - blinky.x, 2 * lead.y - blinky.y)
        };
        self.state.ghosts[CLYDE].target =
            if distance_squared(player, clyde) > CLYDE_TARGET_RADIUS * CLYDE_TARGET_RADIUS {
                player
            } else {
                self.corners[CLYDE]
            };
    }

    /// The ghost's next heading: reverse on command, flee at random when
    /// frightened, otherwise greedily close in on its target. Ties break
    /// toward the earlier heading in `MOVES` order.
    fn ghost_direction(&mut self, ghost: &Ghost) -> usize {
        if self.state.reverse_directions && !ghost.return_to_spawn {
            return REVERSED[ghost.direction];
        }

        let reverse = REVERSED[ghost.direction];
        let mut options = [0usize; 4];
        let mut n = 0;
        for direction in 0..4 {
            if direction != reverse && self.can_move(ghost.position, direction) {
                options[n] = direction;
                n += 1;
            }
        }

        match n {
            0 => ghost.direction, // boxed in; keep heading (the C original reads uninitialised memory here)
            1 => options[0],
            _ if ghost.frightened => options[self.state.rngs.random_range(0..n)],
            _ => {
                let mut best = options[0];
                let mut best_distance =
                    distance_squared(ghost.position + DIRECTIONS[best], ghost.target);
                for &direction in &options[1..n] {
                    let distance =
                        distance_squared(ghost.position + DIRECTIONS[direction], ghost.target);
                    if distance < best_distance {
                        best = direction;
                        best_distance = distance;
                    }
                }
                best
            }
        }
    }

    /// One ghost's step: wait out the start delay, skip every other step
    /// while frightened, then chase, and resolve what it collided with.
    /// Returns the reward the player collects for eating it, if any.
    fn ghost_move(&mut self, index: usize) -> f32 {
        let mut ghost = self.state.ghosts[index];
        ghost.start_timeout -= 1;

        if ghost.frightened && ghost.half_move {
            ghost.half_move = false;
        } else {
            ghost.half_move = true;

            if ghost.return_to_spawn {
                ghost.target = ghost.spawn;
            }

            ghost.direction = self.ghost_direction(&ghost);

            if ghost.start_timeout < 0 {
                let target = self.wrap(ghost.position + DIRECTIONS[ghost.direction]);
                if !self.is_wall(target) {
                    ghost.position = target;
                }
            }
        }

        let mut reward = 0.0;
        if ghost.return_to_spawn {
            if ghost.position == ghost.spawn {
                ghost.return_to_spawn = false;
            }
        } else if self.collides(ghost.position) {
            if ghost.frightened {
                ghost.frightened = false;
                ghost.half_move = false;
                ghost.return_to_spawn = true;
                reward = self.config.ghost_reward;
            } else {
                self.state.player_caught = true;
            }
        }

        self.state.ghosts[index] = ghost;
        reward
    }

    /// Contact means sharing a row within a column of each other, or
    /// sharing a column within a row — the C original's generous graze.
    fn collides(&self, ghost: Position) -> bool {
        let player = self.state.player_position;
        (player.x >= ghost.x - 1 && player.x <= ghost.x + 1 && player.y == ghost.y)
            || (player.y >= ghost.y - 1 && player.y <= ghost.y + 1 && player.x == ghost.x)
    }

    fn encode_observations(&self, timestep: &mut TimeStepMut) {
        let fov_height = self.config.view_height as usize;
        let player = self.state.player_position;

        // wall padding keeps the window inside the map, so the view is a
        // plain crop centred on the player: nothing to raycast or mask.
        let x0 = (player.x - self.pad_width) as usize;
        let y0 = (player.y - self.pad_height) as usize;

        let mut view = timestep.obs.slice_mut(s![0, .., ..fov_height, 0]);
        let window = self.state.map.slice(s![
            x0..x0 + self.config.view_width as usize,
            y0..y0 + fov_height,
        ]);
        view.zip_mut_with(&window, |dst, &tile| *dst = tile.into());

        let mut ui = timestep.obs.slice_mut(s![0, .., fov_height.., 0]);
        ui.fill(PacmanObs::UI as VocabId);

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(
            self.state.player_caught
                || self.state.remaining_pickups == 0
                || self.state.time == self.length,
        );
        timestep.task_ids.fill(0);
    }
}

impl Environment for Pacman {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.state.rngs = SmallRng::seed_from_u64(seed);
        self.reset_round();

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.repaint();
        self.encode_observations(timestep);
        timestep.action_mask.fill(true);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        let action = PacmanAction::from_id(actions[0]) as usize;
        timestep.last_action[0] = actions[0];
        timestep.reward[0] = 0.0;

        self.state.time += 1;
        self.state.reverse_directions = false;
        self.state.player_caught = false;

        if self.state.frightened_time_left > 0 {
            self.state.frightened_time_left -= 1;
        } else {
            for ghost in self.state.ghosts.iter_mut() {
                ghost.frightened = false;
                ghost.half_move = false;
            }
        }

        self.check_mode_change();
        self.set_targets();

        timestep.reward[0] += self.player_move(action);
        for index in 0..NUM_GHOSTS {
            timestep.reward[0] += self.ghost_move(index);
        }

        self.repaint();
        self.encode_observations(timestep);
        timestep.action_mask.fill(true);
    }

    fn observation_spec(&self) -> ObservationSpec {
        self.obs_spec
    }

    fn action_spec(&self) -> ActionSpec {
        self.action_spec
    }

    fn num_agents(&self) -> usize {
        1
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
            tile_width: MAP_WIDTH as usize,
            tile_height: MAP_HEIGHT as usize,
            view_width: self.config.view_width as usize,
            view_height: (self.config.view_height + 2) as usize,
            ui_height: 2,
        }
    }

    fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
        let dim = (MAP_WIDTH as usize, MAP_HEIGHT as usize);
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
        grid_render_state
            .agent_positions
            .push(self.state.player_position - Position::new(self.pad_width, self.pad_height));
    }

    fn num_tasks(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timestep::TimeStepBuffers;
    use PacmanObs::*;

    /// A fresh env, already reset with a fixed seed.
    fn env() -> Pacman {
        let mut env = Pacman::new(&PacmanConfig::default(), 1024);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(0, &mut buffers.view_mut());
        env
    }

    fn step(env: &mut Pacman, buffers: &mut TimeStepBuffers, action: PacmanAction) {
        env.step(&[action.into()], &mut buffers.view_mut());
    }

    /// Reads an observation cell at an offset from the agent. The crop is
    /// the window minus its UI band, so the centre is the fov's centre.
    fn seen(env: &Pacman, buffers: &TimeStepBuffers, dx: i32, dy: i32) -> VocabId {
        let half = (env.config.view_width / 2, env.config.view_height / 2);
        buffers.obs[[0, (half.0 + dx) as usize, (half.1 + dy) as usize, 0]]
    }

    fn id(tile: PacmanObs) -> VocabId {
        tile.into()
    }

    /// Stops the scatter/chase clock from flipping on the next step, so a
    /// test can hold `reverse_directions` false.
    fn freeze_modes(env: &mut Pacman) {
        env.state.mode_changes = env.config.max_mode_changes + 1;
    }

    #[test]
    fn the_maze_holds_the_original_pickup_count() {
        let env = env();
        assert_eq!(env.dot_positions.len(), 240);
        assert_eq!(
            env.original
                .slice(s![
                    env.pad_width as usize..(env.width - env.pad_width) as usize,
                    env.pad_height as usize..(env.height - env.pad_height) as usize,
                ])
                .iter()
                .filter(|&&tile| tile == PacmanObs::TilePower)
                .count(),
            4
        );
        assert_eq!(env.state.remaining_pickups, 244);
    }

    /// The observation is the egocentric crop: the player sits dead centre,
    /// and the maze around it is what the map says, not a feature vector.
    #[test]
    fn the_player_observes_its_own_neighbourhood() {
        let env = env();
        let mut buffers = TimeStepBuffers::new(&env);
        let mut timestep = buffers.view_mut();
        env.encode_observations(&mut timestep);
        drop(timestep);

        assert_eq!(seen(&env, &buffers, 0, 0), id(AgentPacman));
        // the spawn square sits in a corridor: wall above, dot to the left
        assert_eq!(seen(&env, &buffers, 0, 1), id(TileWall));
        assert_eq!(seen(&env, &buffers, -1, 0), id(TileDot));
        // and the UI band tops the window
        assert_eq!(
            buffers.obs[[0, 0, (env.config.view_height + 1) as usize, 0]],
            id(UI)
        );
    }

    #[test]
    fn walking_onto_a_dot_collects_it() {
        let mut env = env();
        let mut buffers = TimeStepBuffers::new(&env);

        let before = env.state.remaining_pickups;
        step(&mut env, &mut buffers, PacmanAction::MoveLeft);

        assert_eq!(buffers.reward[0], env.config.dot_reward);
        assert_eq!(env.state.remaining_pickups, before - 1);
        // the square the player left is plain floor, not the dot it came from
        assert_eq!(seen(&env, &buffers, 1, 0), id(TileEmpty));
    }

    /// An action into a wall is a failed wish to turn: the player keeps
    /// sliding along its heading instead of stopping.
    #[test]
    fn a_blocked_action_keeps_the_current_heading() {
        let mut env = env();
        let mut buffers = TimeStepBuffers::new(&env);

        let from = env.state.player_position;
        step(&mut env, &mut buffers, PacmanAction::MoveUp); // wall above the spawn

        assert_eq!(
            env.state.player_position,
            from + DIRECTIONS[PacmanAction::MoveRight as usize]
        );
        assert_eq!(env.state.player_direction, PacmanAction::MoveRight as usize);
    }

    #[test]
    fn the_tunnel_wraps_both_ways() {
        let mut env = env();
        let mut buffers = TimeStepBuffers::new(&env);

        // the open row across the whole width, screen row 14 from the top
        let tunnel_y = env.pad_height + (MAP_HEIGHT - 1 - 14);
        env.state.player_position = Position::new(env.pad_width, tunnel_y);
        env.state.player_direction = PacmanAction::MoveRight as usize;

        step(&mut env, &mut buffers, PacmanAction::MoveLeft);
        assert_eq!(
            env.state.player_position,
            Position::new(env.pad_width + MAP_WIDTH - 1, tunnel_y)
        );

        step(&mut env, &mut buffers, PacmanAction::MoveRight);
        assert_eq!(
            env.state.player_position,
            Position::new(env.pad_width, tunnel_y)
        );
    }

    /// The wrap-around view reads the live map, not the pristine one:
    /// a dot eaten at the far mouth is gone from the seam crop too.
    #[test]
    fn the_seam_shows_the_current_far_side() {
        let mut env = env();
        let mut buffers = TimeStepBuffers::new(&env);

        // screen row 1 holds a dot at column 23, which the left padding
        // mirrors into column 0; stand the player at that seam
        let row_y = env.pad_height + (MAP_HEIGHT - 1 - 1);
        let far = Position::new(env.pad_width + 23, row_y);
        env.state.player_position = Position::new(env.pad_width, row_y);
        let encode = |env: &Pacman, buffers: &mut TimeStepBuffers| {
            let mut timestep = buffers.view_mut();
            env.encode_observations(&mut timestep);
        };
        encode(&env, &mut buffers);
        assert_eq!(seen(&env, &buffers, -env.pad_width, 0), id(TileDot));

        env.state.base_map[far.idx()] = PacmanObs::TileEmpty;
        env.repaint();
        encode(&env, &mut buffers);
        assert_eq!(seen(&env, &buffers, -env.pad_width, 0), id(TileEmpty));
    }

    /// A power pellet frightens every ghost that is not already eyes.
    #[test]
    fn a_power_pellet_frightens_the_pack() {
        let mut env = env();
        let mut buffers = TimeStepBuffers::new(&env);

        let power = env
            .original
            .indexed_iter()
            .find(|(_, tile)| **tile == PacmanObs::TilePower)
            .map(|(idx, _)| Position::new(idx.0 as i32, idx.1 as i32))
            .unwrap();
        env.state.player_position = power + DIRECTIONS[PacmanAction::MoveLeft as usize];
        env.state.player_direction = PacmanAction::MoveRight as usize;

        step(&mut env, &mut buffers, PacmanAction::MoveRight);

        assert_eq!(env.state.frightened_time_left, env.config.frightened_time);
        assert!(env.state.ghosts.iter().all(|g| g.frightened));
        assert_eq!(buffers.reward[0], env.config.dot_reward);
    }

    /// A mode flip (or a pellet) sends the whole pack reversing before it
    /// starts hunting again.
    #[test]
    fn a_mode_flip_reverses_the_ghosts() {
        let mut env = env();
        let mut buffers = TimeStepBuffers::new(&env);

        step(&mut env, &mut buffers, PacmanAction::MoveLeft);

        assert!(env.state.scatter_mode);
        assert_eq!(env.state.mode_time_left, env.config.scatter_mode_length);
        assert!(
            env.state
                .ghosts
                .iter()
                .all(|g| g.direction == REVERSED[PacmanAction::MoveUp as usize])
        );
        assert_eq!(env.state.ghosts[PINKY].target, env.corners[PINKY]);
    }

    /// Eating a frightened ghost pays, and demotes it to eyes that walk
    /// home; a normal ghost in the same spot would have killed instead.
    #[test]
    fn a_frightened_ghost_eaten_sends_it_home() {
        let mut env = env();
        let mut buffers = TimeStepBuffers::new(&env);
        freeze_modes(&mut env);

        let player = env.state.player_position;
        let blinky = &mut env.state.ghosts[BLINKY];
        blinky.position = player + DIRECTIONS[PacmanAction::MoveRight as usize];
        blinky.start_timeout = -1;
        blinky.frightened = true;
        blinky.half_move = true; // skips this step, so it is still there to eat
        env.repaint();

        env.state.frightened_time_left = env.config.frightened_time;

        step(&mut env, &mut buffers, PacmanAction::MoveRight);
        let blinky = &env.state.ghosts[BLINKY];
        assert_eq!(buffers.reward[0], env.config.ghost_reward);
        assert!(!blinky.frightened);
        assert!(blinky.return_to_spawn);
        assert!(!buffers.terminated[0]);
    }

    #[test]
    fn a_normal_ghost_catches_the_player() {
        let mut env = env();
        let mut buffers = TimeStepBuffers::new(&env);
        freeze_modes(&mut env);

        let player = env.state.player_position;
        let blinky = &mut env.state.ghosts[BLINKY];
        blinky.position = player + DIRECTIONS[PacmanAction::MoveRight as usize];
        blinky.start_timeout = -1;
        env.repaint();

        step(&mut env, &mut buffers, PacmanAction::MoveRight);

        assert!(buffers.terminated[0]);
    }

    /// The round is won by clearing the board: the last pickup ends it.
    #[test]
    fn the_last_dot_ends_the_episode() {
        let mut env = env();
        let mut buffers = TimeStepBuffers::new(&env);
        env.state.remaining_pickups = 1;

        step(&mut env, &mut buffers, PacmanAction::MoveLeft); // the dot beside the spawn

        assert_eq!(env.state.remaining_pickups, 0);
        assert!(buffers.terminated[0]);
    }

    /// Parked ghosts count their delay down and do not move; once it is up
    /// they hunt.
    #[test]
    fn a_parked_ghost_waits_before_hunting() {
        let mut env = env();
        let mut buffers = TimeStepBuffers::new(&env);
        freeze_modes(&mut env);

        let parked = env.state.ghosts[PINKY].position;
        env.state.ghosts[PINKY].start_timeout = 3;

        for _ in 0..3 {
            step(&mut env, &mut buffers, PacmanAction::MoveLeft);
            assert_eq!(env.state.ghosts[PINKY].position, parked);
        }
        step(&mut env, &mut buffers, PacmanAction::MoveLeft);
        assert_ne!(env.state.ghosts[PINKY].position, parked);
    }

    /// Ghosts are painted into the crop the player sees: one shared
    /// frightened tile, and eyes on the way home.
    #[test]
    fn ghost_states_show_as_their_own_tiles_in_the_view() {
        let mut env = env();
        let mut buffers = TimeStepBuffers::new(&env);

        let player = env.state.player_position;
        let pinky = &mut env.state.ghosts[PINKY];
        pinky.position = player + DIRECTIONS[PacmanAction::MoveUp as usize];
        pinky.frightened = true;
        env.repaint();
        let mut timestep = buffers.view_mut();
        env.encode_observations(&mut timestep);
        drop(timestep);
        assert_eq!(seen(&env, &buffers, 0, 1), id(GhostFrightened));

        let pinky = &mut env.state.ghosts[PINKY];
        pinky.frightened = false;
        pinky.return_to_spawn = true;
        env.repaint();
        let mut timestep = buffers.view_mut();
        env.encode_observations(&mut timestep);
        drop(timestep);
        assert_eq!(seen(&env, &buffers, 0, 1), id(GhostEyes));
    }

    /// A randomised start puts the player on some dot square.
    #[test]
    fn a_randomised_start_sits_on_a_dot() {
        let mut env = Pacman::new(
            &PacmanConfig {
                randomize_starting_position: true,
                ..Default::default()
            },
            1024,
        );
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(7, &mut buffers.view_mut());

        assert!(env.dot_positions.contains(&env.state.player_position));
        assert_eq!(seen(&env, &buffers, 0, 0), id(AgentPacman));
    }
}
