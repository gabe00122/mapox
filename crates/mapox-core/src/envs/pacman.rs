use ndarray::{Array2, s};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::common::{Position, UI_HEIGHT, ui::write_number, vocab_enum::VocabEnum},
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_GHOST_BLINKY, AGENT_GHOST_CLYDE, AGENT_GHOST_EYES, AGENT_GHOST_FRIGHTENED,
        AGENT_GHOST_INKY, AGENT_GHOST_PINKY, AGENT_PACMAN_DOWN, AGENT_PACMAN_LEFT,
        AGENT_PACMAN_RIGHT, AGENT_PACMAN_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, TILE_EMPTY,
        TILE_PELLET, TILE_POWER_PELLET, TILE_UI, TILE_WALL, UI_DIGITS,
    },
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
    vocab_enum,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PacmanConfig {
    pub view_width: i32,
    pub view_height: i32,

    /// Start each round on a random pellet instead of the classic spawn.
    pub randomize_starting_position: bool,
    /// Each ghost waits a uniform `min..max` steps (`min` when they are
    /// equal) at the start of a round before it moves.
    pub min_start_timeout: i32,
    pub max_start_timeout: i32,
    /// Steps a power pellet keeps the ghosts frightened.
    pub frightened_time: i32,
    /// The ghosts alternate scatter and chase this many times, then chase
    /// for the rest of the round.
    pub max_mode_changes: i32,
    pub scatter_mode_length: i32,
    pub chase_mode_length: i32,

    /// Paid for every pellet, power pellets included.
    pub pellet_reward: f32,
    /// Paid for every frightened ghost eaten.
    pub ghost_reward: f32,
    /// Paid on the step a ghost catches Pac-Man.
    pub death_reward: f32,
}

impl Default for PacmanConfig {
    fn default() -> Self {
        Self {
            view_width: 15,
            view_height: 15,
            randomize_starting_position: false,
            min_start_timeout: 0,
            max_start_timeout: 49,
            frightened_time: 35,
            max_mode_changes: 6,
            scatter_mode_length: 70,
            chase_mode_length: 140,
            pellet_reward: 1.0,
            ghost_reward: 1.0,
            death_reward: 0.0,
        }
    }
}

/// The classic maze, top row first. `#` wall, `.` pellet, `x` power pellet,
/// `p` Pac-Man's spawn, `1`-`4` the ghost spawns (see [`GHOST_MARKERS`]).
/// Row 14 is open at both ends: the tunnel that wraps x around.
const MAP: [&str; 31] = [
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

const MAP_WIDTH: i32 = 28;
const MAP_HEIGHT: i32 = MAP.len() as i32;

/// A [`MAP`] cell as a map position: the rows are written top first, and map
/// y points up.
const fn screen(x: i32, row: i32) -> Position {
    Position {
        x,
        y: MAP_HEIGHT - 1 - row,
    }
}

/// Row i is the (dx, dy) delta for move action i, in `MOVES` order: up,
/// right, down, left. Headings index this table too, and a heading's reverse
/// is two rows on.
const DIRECTIONS: [Position; 4] = [
    Position { x: 0, y: 1 },
    Position { x: 1, y: 0 },
    Position { x: 0, y: -1 },
    Position { x: -1, y: 0 },
];
const UP: u8 = 0;
const RIGHT: u8 = 1;

fn reverse(dir: u8) -> u8 {
    (dir + 2) % 4
}

/// The order a ghost weighs its exits in, so ties break toward the first:
/// down, up, right, left, as the reference implementation lists them.
const GHOST_EXIT_ORDER: [u8; 4] = [2, 0, 1, 3];

const NUM_GHOSTS: usize = 4;
const PINKY: usize = 0;
const BLINKY: usize = 1;
const INKY: usize = 2;
const CLYDE: usize = 3;

/// The [`MAP`] character marking each ghost's spawn, by ghost index.
const GHOST_MARKERS: [u8; NUM_GHOSTS] = *b"3214";

/// Scatter-mode targets, by ghost index: points off the maze beyond each
/// ghost's home corner, which it circles without ever reaching.
const GHOST_CORNERS: [Position; NUM_GHOSTS] = [
    screen(3, -3),
    screen(MAP_WIDTH - 4, -3),
    screen(MAP_WIDTH - 1, MAP_HEIGHT),
    screen(0, MAP_HEIGHT),
];

const PINKY_TARGET_LEAD: i32 = 4;
const INKY_TARGET_LEAD: i32 = 2;
const CLYDE_TARGET_RADIUS: i32 = 8;

vocab_enum!(PacmanObs {
    TileUI => TILE_UI,
    TileEmpty => TILE_EMPTY,
    TileWall => TILE_WALL,
    Pellet => TILE_PELLET,
    PowerPellet => TILE_POWER_PELLET,
    PacmanUp => AGENT_PACMAN_UP,
    PacmanRight => AGENT_PACMAN_RIGHT,
    PacmanDown => AGENT_PACMAN_DOWN,
    PacmanLeft => AGENT_PACMAN_LEFT,
    GhostPinky => AGENT_GHOST_PINKY,
    GhostBlinky => AGENT_GHOST_BLINKY,
    GhostInky => AGENT_GHOST_INKY,
    GhostClyde => AGENT_GHOST_CLYDE,
    GhostFrightened => AGENT_GHOST_FRIGHTENED,
    GhostEyes => AGENT_GHOST_EYES,
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

/// Pac-Man's tile by heading, in `DIRECTIONS` order.
const PACMAN_TILES: [PacmanObs; 4] = [
    PacmanObs::PacmanUp,
    PacmanObs::PacmanRight,
    PacmanObs::PacmanDown,
    PacmanObs::PacmanLeft,
];

/// Each ghost's own tile when it is neither frightened nor eaten.
const GHOST_TILES: [PacmanObs; NUM_GHOSTS] = [
    PacmanObs::GhostPinky,
    PacmanObs::GhostBlinky,
    PacmanObs::GhostInky,
    PacmanObs::GhostClyde,
];

const DIGIT_TILES: [PacmanObs; 10] = [
    PacmanObs::Digit0,
    PacmanObs::Digit1,
    PacmanObs::Digit2,
    PacmanObs::Digit3,
    PacmanObs::Digit4,
    PacmanObs::Digit5,
    PacmanObs::Digit6,
    PacmanObs::Digit7,
    PacmanObs::Digit8,
    PacmanObs::Digit9,
];

vocab_enum!(
    #[allow(clippy::enum_variant_names)]
    PacmanAction {
        MoveUp => MOVE_UP,
        MoveRight => MOVE_RIGHT,
        MoveDown => MOVE_DOWN,
        MoveLeft => MOVE_LEFT,
    }
);

#[derive(Debug, Clone)]
struct Ghost {
    spawn: Position,
    pos: Position,
    target: Position,
    /// Heading as a `DIRECTIONS` index. A ghost never turns back on itself
    /// except when the mode flips or a power pellet lands.
    dir: u8,
    /// Steps left before the ghost first moves this round; negative once it
    /// is free.
    start_timeout: i32,
    frightened: bool,
    /// Eaten and heading home as a pair of eyes; harmless, and immune to
    /// power pellets until it arrives.
    returning: bool,
    /// Frightened ghosts move every other step; this is set on the steps
    /// they move.
    half_move: bool,
}

#[derive(Debug, Clone)]
struct PacmanState {
    /// The maze as it stands: walls, floor, and the pellets not yet eaten.
    tiles: Array2<PacmanObs>,
    /// `tiles` with the ghosts and Pac-Man drawn over it, which is what the
    /// views and the renderer read. Rebuilt after every reset and step.
    painted: Array2<PacmanObs>,

    player: Position,
    player_dir: u8,
    ghosts: [Ghost; NUM_GHOSTS],

    remaining_pellets: usize,
    /// Pellets eaten this round, shown in the UI band.
    score: u32,

    scatter_mode: bool,
    mode_time_left: i32,
    mode_changes: i32,
    frightened_time_left: i32,
    /// Every ghost turns around this step: set by a mode flip or a power
    /// pellet, cleared at the start of the next step.
    reverse_directions: bool,

    time: usize,
}

#[derive(Debug, Clone, Default)]
struct PacmanMetrics {
    reward: f64,
    pellets_eaten: f64,
    ghosts_eaten: f64,
    deaths: f64,
    clears: f64,
}

/// Pac-Man on the classic maze, ported from the PufferLib C env: the same
/// four ghost personalities, scatter/chase schedule, power pellets, and
/// half-speed frightened ghosts.
///
/// The maze wraps around through the side tunnel, and so does the view: the
/// window is a crop of a cylinder centred on Pac-Man, with x taken modulo the
/// maze width, so whatever sits across the tunnel is visible from this side
/// exactly as it would be walked to. The maze does not wrap vertically, and
/// rows beyond its top and bottom read as wall. There is no wall padding to
/// crop from; every view cell maps into the maze on its own.
///
/// The UI band carries what the maze does not show: its top row is the
/// round's score (the count of pellets eaten, right-aligned), and the row
/// beneath it counts down the power pellet's remaining steps (blank when no
/// pellet is active).
///
/// A round ends when a ghost catches Pac-Man or the last pellet is eaten.
/// Pac-Man is flagged terminated on that step and the next round starts on
/// a fresh maze at once, as the dead snake respawns in the snake env; the
/// episode itself runs until `length`, like every env here.
#[derive(Debug, Clone)]
pub struct Pacman {
    pub config: PacmanConfig,
    state: PacmanState,
    metrics: PacmanMetrics,
    rng: SmallRng,

    // max steps for a single episode
    length: usize,

    /// The maze at the start of a round.
    initial_tiles: Array2<PacmanObs>,
    player_spawn: Position,
    /// Where a randomized start may land: every plain pellet.
    pellet_spawns: Vec<Position>,

    obs_spec: ObservationSpec,
    action_spec: ActionSpec,

    obs_vocab: Vocabulary,
    action_vocab: Vocabulary,
}

impl Pacman {
    pub fn new(config: &PacmanConfig, length: usize) -> Self {
        assert!(
            config.min_start_timeout <= config.max_start_timeout,
            "min_start_timeout exceeds max_start_timeout"
        );

        let action_vocab = PacmanAction::vocab();
        let obs_vocab = PacmanObs::vocab();

        let view_height = config.view_height + UI_HEIGHT as i32;
        let obs_spec = ObservationSpec::new(config.view_width, view_height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        let dim = (MAP_WIDTH as usize, MAP_HEIGHT as usize);
        let mut initial_tiles = Array2::from_elem(dim, PacmanObs::TileEmpty);
        let mut player_spawn = None;
        let mut ghost_spawns = [None; NUM_GHOSTS];
        let mut pellet_spawns = Vec::new();

        for (row, line) in MAP.iter().enumerate() {
            assert_eq!(line.len(), MAP_WIDTH as usize, "map row {row} is ragged");
            for (x, cell) in line.bytes().enumerate() {
                let pos = screen(x as i32, row as i32);
                initial_tiles[pos.idx()] = match cell {
                    b'#' => PacmanObs::TileWall,
                    b'.' => PacmanObs::Pellet,
                    b'x' => PacmanObs::PowerPellet,
                    _ => PacmanObs::TileEmpty,
                };
                match cell {
                    b'.' => pellet_spawns.push(pos),
                    b'p' => player_spawn = Some(pos),
                    _ => {}
                }
                if let Some(ghost) = GHOST_MARKERS.iter().position(|&m| m == cell) {
                    ghost_spawns[ghost] = Some(pos);
                }
            }
        }

        let player_spawn = player_spawn.expect("the map marks Pac-Man's spawn");
        let ghosts = ghost_spawns.map(|spawn| {
            let spawn = spawn.expect("the map marks every ghost's spawn");
            Ghost {
                spawn,
                pos: spawn,
                target: spawn,
                dir: UP,
                start_timeout: 0,
                frightened: false,
                returning: false,
                half_move: false,
            }
        });

        let state = PacmanState {
            tiles: initial_tiles.clone(),
            painted: initial_tiles.clone(),
            player: player_spawn,
            player_dir: RIGHT,
            ghosts,
            remaining_pellets: 0,
            score: 0,
            scatter_mode: false,
            mode_time_left: 0,
            mode_changes: 0,
            frightened_time_left: 0,
            reverse_directions: false,
            time: 0,
        };

        Self {
            config: config.clone(),
            state,
            metrics: PacmanMetrics::default(),
            rng: SmallRng::seed_from_u64(0),
            length,

            initial_tiles,
            player_spawn,
            pellet_spawns,

            obs_spec,
            action_spec,

            obs_vocab,
            action_vocab,
        }
    }

    /// `p` with x wrapped around the tunnel. y never wraps.
    fn wrap(p: Position) -> Position {
        Position::new(p.x.rem_euclid(MAP_WIDTH), p.y)
    }

    /// The maze tile under `p`, x wrapped; rows off the top or bottom are
    /// wall.
    fn tile(&self, p: Position) -> PacmanObs {
        if p.y < 0 || p.y >= MAP_HEIGHT {
            return PacmanObs::TileWall;
        }
        self.state.tiles[Self::wrap(p).idx()]
    }

    fn can_move(&self, from: Position, dir: u8) -> bool {
        self.tile(from + DIRECTIONS[dir as usize]) != PacmanObs::TileWall
    }

    /// Adjacent or on the same cell, along a row or a column. Adjacency
    /// counts across the tunnel, and catches a ghost and Pac-Man that swap
    /// cells in one step.
    fn touching(a: Position, b: Position) -> bool {
        let dx = (a.x - b.x).rem_euclid(MAP_WIDTH);
        let dy = (a.y - b.y).abs();
        (dy == 0 && (dx <= 1 || dx >= MAP_WIDTH - 1)) || (dx == 0 && dy <= 1)
    }

    fn start_timeout(&mut self) -> i32 {
        let (min, max) = (self.config.min_start_timeout, self.config.max_start_timeout);
        if min == max {
            min
        } else {
            self.rng.random_range(min..max)
        }
    }

    /// A fresh maze, with every ghost home and the schedule at its start.
    /// Runs at every reset and whenever a round ends mid-episode.
    fn reset_round(&mut self) {
        self.state.tiles.assign(&self.initial_tiles);
        self.state.remaining_pellets = self
            .initial_tiles
            .iter()
            .filter(|&&t| matches!(t, PacmanObs::Pellet | PacmanObs::PowerPellet))
            .count();
        self.state.score = 0;

        self.state.scatter_mode = false;
        self.state.mode_time_left = 0;
        self.state.mode_changes = 0;
        self.state.frightened_time_left = 0;
        self.state.reverse_directions = false;

        for i in 0..NUM_GHOSTS {
            let start_timeout = self.start_timeout();
            let ghost = &mut self.state.ghosts[i];
            ghost.pos = ghost.spawn;
            ghost.target = ghost.spawn;
            ghost.dir = UP;
            ghost.start_timeout = start_timeout;
            ghost.frightened = false;
            ghost.returning = false;
            ghost.half_move = false;
        }

        self.state.player = if self.config.randomize_starting_position {
            self.pellet_spawns[self.rng.random_range(0..self.pellet_spawns.len())]
        } else {
            self.player_spawn
        };
        self.state.player_dir = RIGHT;
    }

    /// Scatter and chase alternate on a timer until `max_mode_changes` flips
    /// have happened; every flip turns the ghosts around.
    fn check_mode_change(&mut self) {
        if self.state.mode_changes >= self.config.max_mode_changes {
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

    /// Each ghost's chase target: Blinky goes straight for Pac-Man, Pinky
    /// ambushes a few cells ahead of him, Inky mirrors Blinky through a point
    /// just ahead of him, and Clyde chases until close, then retreats to his
    /// corner.
    fn set_chase_targets(&mut self) {
        let player = self.state.player;
        let heading = DIRECTIONS[self.state.player_dir as usize];
        let ghosts = &mut self.state.ghosts;

        ghosts[PINKY].target = player + heading * PINKY_TARGET_LEAD;
        ghosts[BLINKY].target = player;

        let pivot = player + heading * INKY_TARGET_LEAD;
        ghosts[INKY].target = pivot * 2 - ghosts[BLINKY].pos;

        let to_clyde = player - ghosts[CLYDE].pos;
        let clyde_distance = to_clyde.x * to_clyde.x + to_clyde.y * to_clyde.y;
        ghosts[CLYDE].target = if clyde_distance > CLYDE_TARGET_RADIUS * CLYDE_TARGET_RADIUS {
            player
        } else {
            GHOST_CORNERS[CLYDE]
        };
    }

    /// Moves Pac-Man one cell. A move into a wall keeps him going the way he
    /// was heading instead, and if that is a wall too he stays put. Returns
    /// the reward for what he ate.
    fn player_move(&mut self, action: VocabId) -> f32 {
        let player = self.state.player;
        let is_move = usize::from(action) < PacmanAction::TABLE.len();

        let dir = if is_move && self.can_move(player, action as u8) {
            action as u8
        } else {
            self.state.player_dir
        };
        self.state.player_dir = dir;
        if !self.can_move(player, dir) {
            return 0.0;
        }

        let next = Self::wrap(player + DIRECTIONS[dir as usize]);
        self.state.player = next;

        let tile = self.state.tiles[next.idx()];
        if !matches!(tile, PacmanObs::Pellet | PacmanObs::PowerPellet) {
            return 0.0;
        }

        self.state.tiles[next.idx()] = PacmanObs::TileEmpty;
        self.state.remaining_pellets -= 1;
        self.state.score += 1;
        self.metrics.pellets_eaten += 1.0;

        if tile == PacmanObs::PowerPellet {
            self.state.frightened_time_left = self.config.frightened_time;
            self.state.reverse_directions = true;
            for ghost in &mut self.state.ghosts {
                ghost.frightened = !ghost.returning;
            }
        }

        self.config.pellet_reward
    }

    /// The way ghost `i` turns this step: straight back on a reversal,
    /// otherwise the exit, never backwards, that leaves it nearest its target
    /// — or a random one while frightened.
    fn ghost_direction(&mut self, i: usize) -> u8 {
        let Ghost {
            pos,
            target,
            dir,
            frightened,
            returning,
            ..
        } = self.state.ghosts[i];
        let back = reverse(dir);
        if self.state.reverse_directions && !returning {
            return back;
        }

        let mut exits = [0u8; 4];
        let mut count = 0;
        for exit in GHOST_EXIT_ORDER {
            if exit != back && self.can_move(pos, exit) {
                exits[count] = exit;
                count += 1;
            }
        }

        match count {
            // A dead end: the one way out is back. The classic maze has
            // none, but a ghost must still go somewhere.
            0 => back,
            1 => exits[0],
            _ if frightened => exits[self.rng.random_range(0..count)],
            // min_by_key keeps the first of equal keys, so ties go to the
            // earlier exit in GHOST_EXIT_ORDER
            _ => *exits[..count]
                .iter()
                .min_by_key(|&&exit| {
                    let d = pos + DIRECTIONS[exit as usize] - target;
                    d.x * d.x + d.y * d.y
                })
                .expect("count > 0"),
        }
    }

    /// Moves ghost `i` and settles any contact with Pac-Man. Returns the
    /// reward for eating it, and whether it caught him.
    fn ghost_move(&mut self, i: usize) -> (f32, bool) {
        self.state.ghosts[i].start_timeout -= 1;

        let ghost = &mut self.state.ghosts[i];
        if ghost.frightened && ghost.half_move {
            ghost.half_move = false;
        } else {
            ghost.half_move = true;
            if ghost.returning {
                ghost.target = ghost.spawn;
            }

            let dir = self.ghost_direction(i);
            let ghost = &self.state.ghosts[i];
            let can_move = ghost.start_timeout < 0 && self.can_move(ghost.pos, dir);

            let ghost = &mut self.state.ghosts[i];
            ghost.dir = dir;
            if can_move {
                ghost.pos = Self::wrap(ghost.pos + DIRECTIONS[dir as usize]);
            }
        }

        let player = self.state.player;
        let ghost = &mut self.state.ghosts[i];
        if ghost.returning {
            if ghost.pos == ghost.spawn {
                ghost.returning = false;
            }
        } else if Self::touching(ghost.pos, player) {
            if ghost.frightened {
                ghost.frightened = false;
                ghost.half_move = false;
                ghost.returning = true;
                self.metrics.ghosts_eaten += 1.0;
                return (self.config.ghost_reward, false);
            }
            return (0.0, true);
        }
        (0.0, false)
    }

    /// Draws the ghosts and then Pac-Man over the maze, so he is always on
    /// top of his own cell.
    fn repaint(&mut self) {
        self.state.painted.assign(&self.state.tiles);
        for (i, ghost) in self.state.ghosts.iter().enumerate() {
            self.state.painted[ghost.pos.idx()] = if ghost.returning {
                PacmanObs::GhostEyes
            } else if ghost.frightened {
                PacmanObs::GhostFrightened
            } else {
                GHOST_TILES[i]
            };
        }
        self.state.painted[self.state.player.idx()] = PACMAN_TILES[self.state.player_dir as usize];
    }

    fn encode_observations(&self, timestep: &mut TimeStepMut) {
        let view_width = self.config.view_width;
        let fov_height = self.config.view_height;
        // an even window runs [-half, half - 1], matching the other envs
        let x0 = self.state.player.x - view_width / 2;
        let y0 = self.state.player.y - fov_height / 2;

        let mut obs = timestep.obs.slice_mut(s![0, .., .., 0]);
        for vx in 0..view_width {
            let x = (x0 + vx).rem_euclid(MAP_WIDTH) as usize;
            // the maze is indexed [x, y], so map column x is array row x
            let column = self.state.painted.row(x);
            for vy in 0..fov_height {
                let y = y0 + vy;
                let tile = if (0..MAP_HEIGHT).contains(&y) {
                    column[y as usize]
                } else {
                    PacmanObs::TileWall
                };
                obs[[vx as usize, vy as usize]] = tile.into();
            }
        }

        let fov_height = fov_height as usize;
        obs.slice_mut(s![.., fov_height..])
            .fill(PacmanObs::TileUI.into());

        // The top row of the band is the score, the one under it the power
        // pellet's countdown.
        let score_row = fov_height + UI_HEIGHT - 1;
        let power_row = fov_height;
        write_number(
            obs.slice_mut(s![.., score_row]),
            self.state.score,
            &DIGIT_TILES,
        );
        if self.state.frightened_time_left > 0 {
            write_number(
                obs.slice_mut(s![.., power_row]),
                self.state.frightened_time_left as u32,
                &DIGIT_TILES,
            );
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length);
        timestep.task_ids.fill(0);
    }
}

impl Environment for Pacman {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.rng = SmallRng::seed_from_u64(seed);
        self.state.time = 0;
        self.reset_round();
        self.repaint();

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        timestep.action_mask.fill(true);
        self.encode_observations(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        let action = actions.first().copied().unwrap_or(VocabId::MAX);

        self.state.reverse_directions = false;

        if self.state.frightened_time_left > 0 {
            self.state.frightened_time_left -= 1;
        } else {
            for ghost in &mut self.state.ghosts {
                ghost.frightened = false;
                ghost.half_move = false;
            }
        }

        self.check_mode_change();
        if self.state.scatter_mode {
            for (ghost, corner) in self.state.ghosts.iter_mut().zip(GHOST_CORNERS) {
                ghost.target = corner;
            }
        } else {
            self.set_chase_targets();
        }

        let mut reward = self.player_move(action);

        let mut caught = false;
        for i in 0..NUM_GHOSTS {
            let (ghost_reward, caught_by) = self.ghost_move(i);
            reward += ghost_reward;
            caught |= caught_by;
        }

        let cleared = self.state.remaining_pellets == 0;
        if caught {
            reward += self.config.death_reward;
            self.metrics.deaths += 1.0;
        } else if cleared {
            self.metrics.clears += 1.0;
        }
        if caught || cleared {
            self.reset_round();
        }
        self.repaint();

        self.state.time += 1;
        self.metrics.reward += f64::from(reward);
        timestep.reward.fill(reward);
        timestep.last_action.fill(action);

        self.encode_observations(timestep);
        timestep.terminated[0] |= caught || cleared;
    }

    fn consume_metrics(&mut self) -> serde_json::Value {
        let metrics = std::mem::take(&mut self.metrics);
        serde_json::json!({
            "reward": metrics.reward,
            "pellets_eaten": metrics.pellets_eaten,
            "ghosts_eaten": metrics.ghosts_eaten,
            "deaths": metrics.deaths,
            "clears": metrics.clears,
        })
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
            view_height: self.config.view_height as usize + UI_HEIGHT,
            ui_height: UI_HEIGHT,
        }
    }

    fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
        let tilemap = &mut grid_render_state.tilemap;
        if tilemap.dim() != self.state.painted.dim() {
            *tilemap = Array2::zeros(self.state.painted.dim());
        }
        tilemap.zip_mut_with(&self.state.painted, |dst, &tile| *dst = tile.into());

        grid_render_state.agent_positions.clear();
        grid_render_state.agent_positions.push(self.state.player);
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
    use ndarray::ArrayView2;

    const DOWN: u8 = 2;
    const LEFT: u8 = 3;

    /// A start timeout no test outlives: the ghost turns in place but never
    /// moves.
    const PARKED: i32 = 1_000_000;

    /// Pellets plus power pellets on the classic maze.
    const PICKUPS: usize = 244;

    fn pos(x: i32, y: i32) -> Position {
        Position::new(x, y)
    }

    fn id(tile: PacmanObs) -> VocabId {
        tile.into()
    }

    fn test_env() -> Pacman {
        Pacman::new(&PacmanConfig::default(), 512)
    }

    /// A fresh round with Pac-Man placed by hand and every ghost parked at
    /// home, so only what a test moves by hand can reach him.
    fn setup(env: &mut Pacman, player: Position, dir: u8) -> TimeStepBuffers {
        let mut buffers = TimeStepBuffers::new(&*env);
        env.reset(0, &mut buffers.view_mut());
        env.state.player = player;
        env.state.player_dir = dir;
        for ghost in &mut env.state.ghosts {
            ghost.start_timeout = PARKED;
        }
        env.repaint();
        env.encode_observations(&mut buffers.view_mut());
        buffers
    }

    fn step(env: &mut Pacman, buffers: &mut TimeStepBuffers, action: u8) {
        env.step(&[action as VocabId], &mut buffers.view_mut());
    }

    fn view(buffers: &TimeStepBuffers) -> ArrayView2<'_, VocabId> {
        buffers.obs.slice(s![0, .., .., 0])
    }

    /// The view cell `(dx, dy)` away from Pac-Man.
    fn cell(env: &Pacman, dx: i32, dy: i32) -> [usize; 2] {
        [
            (env.config.view_width / 2 + dx) as usize,
            (env.config.view_height / 2 + dy) as usize,
        ]
    }

    fn pellets_left(env: &Pacman) -> usize {
        env.state
            .tiles
            .iter()
            .filter(|&&t| matches!(t, Pellet | PowerPellet))
            .count()
    }

    #[test]
    fn the_maze_parses() {
        let env = test_env();
        assert_eq!(env.initial_tiles.dim(), (28, 31));
        assert_eq!(env.player_spawn, pos(13, 7));
        assert_eq!(env.pellet_spawns.len(), 240);
        assert_eq!(
            env.state.ghosts.each_ref().map(|g| g.spawn),
            [pos(14, 19), pos(13, 19), pos(12, 19), pos(15, 19)],
        );
        // the tunnel is open at both ends
        assert_eq!(env.initial_tiles[[0, 16]], TileEmpty);
        assert_eq!(env.initial_tiles[[27, 16]], TileEmpty);
    }

    #[test]
    fn reset_lays_out_a_full_maze() {
        let mut env = test_env();
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(3, &mut buffers.view_mut());

        assert_eq!(pellets_left(&env), PICKUPS);
        assert_eq!(env.state.remaining_pellets, PICKUPS);
        assert_eq!(env.state.player, env.player_spawn);
        assert_eq!(view(&buffers)[cell(&env, 0, 0)], id(PacmanRight));
        assert!(buffers.action_mask.iter().all(|&legal| legal));
        assert_eq!(buffers.obs.dim(), (1, 15, 15 + UI_HEIGHT, 1));
    }

    #[test]
    fn a_randomized_start_lands_on_a_pellet() {
        let config = PacmanConfig {
            randomize_starting_position: true,
            ..PacmanConfig::default()
        };
        let mut env = Pacman::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        for seed in 0..20 {
            env.reset(seed, &mut buffers.view_mut());
            assert_eq!(env.initial_tiles[env.state.player.idx()], Pellet);
        }
    }

    /// Standing in the tunnel mouth, the far side of the maze is right there
    /// to the left: the window is a crop of a cylinder, not a padded plane.
    #[test]
    fn the_view_wraps_through_the_tunnel() {
        let mut env = test_env();
        let mut buffers = setup(&mut env, pos(0, 16), LEFT);
        env.state.ghosts[BLINKY].pos = pos(26, 16);
        env.repaint();
        env.encode_observations(&mut buffers.view_mut());
        let view = view(&buffers);

        assert_eq!(view[cell(&env, 0, 0)], id(PacmanLeft));
        // x = 27, across the tunnel, and Blinky just beyond it at x = 26
        assert_eq!(view[cell(&env, -1, 0)], id(TileEmpty));
        assert_eq!(view[cell(&env, -2, 0)], id(GhostBlinky));
        // the tunnel's pellets on both sides: x = 21 to the left, x = 6 right
        assert_eq!(view[cell(&env, -7, 0)], id(Pellet));
        assert_eq!(view[cell(&env, 6, 0)], id(Pellet));
        // the walls either side of the tunnel mouth, wrapped as well
        assert_eq!(view[cell(&env, -1, 1)], id(TileWall));
        assert_eq!(view[cell(&env, -1, -1)], id(TileWall));
    }

    /// Every view cell agrees with the maze at its x taken modulo the width,
    /// wherever Pac-Man stands.
    #[test]
    fn every_view_cell_is_the_maze_modulo_its_width() {
        let mut env = test_env();
        for &player in &[pos(0, 16), pos(27, 16), pos(3, 16), pos(13, 7), pos(26, 29)] {
            let buffers = setup(&mut env, player, RIGHT);
            let view = view(&buffers);
            for dx in -7..=7 {
                for dy in -7..=7 {
                    let y = player.y + dy;
                    let expected = if (0..MAP_HEIGHT).contains(&y) {
                        env.state.painted
                            [[(player.x + dx).rem_euclid(MAP_WIDTH) as usize, y as usize]]
                    } else {
                        TileWall
                    };
                    assert_eq!(
                        view[cell(&env, dx, dy)],
                        id(expected),
                        "{player:?} + ({dx}, {dy})"
                    );
                }
            }
        }
    }

    /// The maze does not wrap vertically: the rows past its edge are wall.
    #[test]
    fn rows_beyond_the_maze_read_as_wall() {
        let mut env = test_env();
        let buffers = setup(&mut env, pos(1, 1), LEFT);
        let view = view(&buffers);

        for dy in -7..=-2 {
            for dx in -7..=7 {
                assert_eq!(view[cell(&env, dx, dy)], id(TileWall), "({dx}, {dy})");
            }
        }
        // the bottom corridor itself, one row up from the bottom wall
        assert_eq!(view[cell(&env, 1, 0)], id(Pellet));
    }

    #[test]
    fn pacman_walks_through_the_tunnel() {
        let mut env = test_env();
        let mut buffers = setup(&mut env, pos(0, 16), LEFT);
        step(&mut env, &mut buffers, LEFT);
        assert_eq!(env.state.player, pos(27, 16));

        step(&mut env, &mut buffers, RIGHT);
        assert_eq!(env.state.player, pos(0, 16));
    }

    /// A move into a wall keeps Pac-Man going the way he was heading.
    #[test]
    fn a_move_into_a_wall_keeps_the_heading() {
        let mut env = test_env();
        // x = 14 is open on the spawn row; above and below are wall
        let mut buffers = setup(&mut env, pos(13, 7), RIGHT);
        step(&mut env, &mut buffers, UP);
        assert_eq!(env.state.player, pos(14, 7));
        assert_eq!(env.state.player_dir, RIGHT);

        // a non-move action does the same
        step(&mut env, &mut buffers, 9);
        assert_eq!(env.state.player, pos(15, 7));
    }

    #[test]
    fn walled_in_both_ways_pacman_stays_put() {
        let mut env = test_env();
        // the bottom-left corner: wall to the left and below
        let mut buffers = setup(&mut env, pos(1, 1), LEFT);
        step(&mut env, &mut buffers, DOWN);
        assert_eq!(env.state.player, pos(1, 1));
        assert_eq!(buffers.reward[0], 0.0);
    }

    #[test]
    fn pellets_pay_and_count_toward_the_score() {
        let mut env = test_env();
        let mut buffers = setup(&mut env, pos(14, 7), RIGHT);
        step(&mut env, &mut buffers, RIGHT);

        assert_eq!(env.state.player, pos(15, 7));
        assert_eq!(buffers.reward[0], env.config.pellet_reward);
        assert_eq!(env.state.tiles[[15, 7]], TileEmpty);
        assert_eq!(env.state.score, 1);
        assert_eq!(env.state.remaining_pellets, PICKUPS - 1);
        assert_eq!(pellets_left(&env), PICKUPS - 1);

        // backtracking over the eaten cell pays nothing
        step(&mut env, &mut buffers, LEFT);
        step(&mut env, &mut buffers, RIGHT);
        assert_eq!(buffers.reward[0], 0.0);
        assert_eq!(env.state.score, 1);
    }

    /// The band holds only UI and digits, and none of those leak into the
    /// field of view. Its top row is the score and the row under it the power
    /// countdown, both right-aligned.
    #[test]
    fn the_ui_band_shows_the_score_and_the_power_countdown() {
        let config = PacmanConfig {
            ghost_reward: 5.0,
            ..PacmanConfig::default()
        };
        let mut env = Pacman::new(&config, 512);
        // one step right of the power pellet in the lower-left corner
        let mut buffers = setup(&mut env, pos(2, 7), LEFT);
        env.state.score = 41;
        step(&mut env, &mut buffers, LEFT);

        let view = view(&buffers);
        let fov = env.config.view_height as usize;
        let width = env.config.view_width as usize;
        let row = |y: usize| (0..width).map(|x| view[[x, y]]).collect::<Vec<_>>();

        let blank = |n: usize| std::iter::repeat_n(id(TileUI), n);
        let score: Vec<_> = blank(width - 2)
            .chain([id(Digit4), id(Digit2)])
            .collect();
        let power: Vec<_> = blank(width - 2)
            .chain([id(Digit3), id(Digit5)])
            .collect();
        assert_eq!(row(fov + 1), score);
        assert_eq!(row(fov), power);

        let ui_only = [id(TileUI)]
            .into_iter()
            .chain(DIGIT_TILES.iter().map(|&d| id(d)))
            .collect::<Vec<_>>();
        for y in 0..fov {
            assert!(
                row(y).iter().all(|tile| !ui_only.contains(tile)),
                "UI leaked into fov row {y}"
            );
        }

        // The countdown clears once the power runs out.
        for _ in 0..env.config.frightened_time {
            step(&mut env, &mut buffers, LEFT);
        }
        let view = self::view(&buffers);
        assert!((0..width).all(|x| view[[x, fov]] == id(TileUI)));
    }

    #[test]
    fn a_power_pellet_makes_ghosts_edible() {
        let config = PacmanConfig {
            ghost_reward: 5.0,
            ..PacmanConfig::default()
        };
        let mut env = Pacman::new(&config, 512);
        let mut buffers = setup(&mut env, pos(2, 7), LEFT);
        // Blinky waits just above the power pellet; Pinky is out of reach.
        env.state.ghosts[BLINKY].pos = pos(1, 8);
        step(&mut env, &mut buffers, LEFT);

        assert_eq!(env.state.player, pos(1, 7));
        assert_eq!(buffers.reward[0], 1.0 + 5.0);
        assert!(!buffers.terminated[0]);
        assert!(env.state.ghosts[BLINKY].returning);
        assert!(!env.state.ghosts[BLINKY].frightened);
        assert!(env.state.ghosts[PINKY].frightened);
        assert_eq!(env.state.painted[[1, 8]], GhostEyes);
        assert_eq!(
            env.state.painted[env.state.ghosts[PINKY].pos.idx()],
            GhostFrightened
        );
        assert_eq!(env.metrics.ghosts_eaten, 1.0);
    }

    /// Eyes are harmless: walking into a returning ghost is no death.
    #[test]
    fn returning_eyes_do_not_catch() {
        let mut env = test_env();
        let mut buffers = setup(&mut env, pos(14, 7), RIGHT);
        env.state.ghosts[BLINKY].pos = pos(16, 7);
        env.state.ghosts[BLINKY].returning = true;
        step(&mut env, &mut buffers, RIGHT);

        assert!(!buffers.terminated[0]);
        assert_eq!(env.metrics.deaths, 0.0);
    }

    /// Caught, the round ends: Pac-Man is flagged terminated and the next
    /// round starts on a full maze in the same step.
    #[test]
    fn a_catch_ends_the_round() {
        let config = PacmanConfig {
            death_reward: -3.0,
            ..PacmanConfig::default()
        };
        let mut env = Pacman::new(&config, 512);
        let mut buffers = setup(&mut env, pos(18, 7), RIGHT);
        env.state.ghosts[BLINKY].pos = pos(20, 7);
        step(&mut env, &mut buffers, RIGHT);

        assert!(buffers.terminated[0]);
        // the pellet at x = 19 was eaten on the way in
        assert_eq!(buffers.reward[0], 1.0 - 3.0);
        assert_eq!(env.state.player, env.player_spawn);
        assert_eq!(env.state.score, 0);
        assert_eq!(pellets_left(&env), PICKUPS);
        assert_eq!(env.state.ghosts[BLINKY].pos, env.state.ghosts[BLINKY].spawn);
        assert_eq!(view(&buffers)[cell(&env, 0, 0)], id(PacmanRight));
        assert_eq!(env.metrics.deaths, 1.0);

        // the round goes on from there
        step(&mut env, &mut buffers, RIGHT);
        assert!(!buffers.terminated[0]);
    }

    /// A ghost and Pac-Man walking through each other still collide.
    #[test]
    fn a_swap_is_a_catch() {
        let mut env = test_env();
        let mut buffers = setup(&mut env, pos(18, 7), RIGHT);
        let blinky = &mut env.state.ghosts[BLINKY];
        blinky.pos = pos(19, 7);
        blinky.dir = LEFT;
        blinky.start_timeout = 0;
        blinky.target = pos(0, 7);
        step(&mut env, &mut buffers, RIGHT);

        assert!(buffers.terminated[0]);
    }

    /// Contact counts across the tunnel too.
    #[test]
    fn a_catch_reaches_across_the_tunnel() {
        let mut env = test_env();
        let mut buffers = setup(&mut env, pos(1, 16), LEFT);
        env.state.ghosts[BLINKY].pos = pos(27, 16);
        step(&mut env, &mut buffers, LEFT);

        assert!(buffers.terminated[0]);
    }

    #[test]
    fn eating_the_last_pellet_clears_the_round() {
        let mut env = test_env();
        let mut buffers = setup(&mut env, pos(14, 7), RIGHT);
        env.state.tiles.mapv_inplace(|t| match t {
            Pellet | PowerPellet => TileEmpty,
            t => t,
        });
        env.state.tiles[[15, 7]] = Pellet;
        env.state.remaining_pellets = 1;
        step(&mut env, &mut buffers, RIGHT);

        assert!(buffers.terminated[0]);
        assert_eq!(buffers.reward[0], env.config.pellet_reward);
        assert_eq!(pellets_left(&env), PICKUPS);
        assert_eq!(env.state.remaining_pellets, PICKUPS);
        assert_eq!(
            env.consume_metrics(),
            serde_json::json!({
                "reward": 1.0,
                "pellets_eaten": 1.0,
                "ghosts_eaten": 0.0,
                "deaths": 0.0,
                "clears": 1.0,
            })
        );
    }

    /// The schedule opens in scatter, and every flip turns the ghosts around.
    #[test]
    fn the_ghosts_scatter_first_and_reverse_on_each_flip() {
        let config = PacmanConfig {
            scatter_mode_length: 3,
            chase_mode_length: 5,
            ..PacmanConfig::default()
        };
        let mut env = Pacman::new(&config, 512);
        let mut buffers = setup(&mut env, pos(13, 7), RIGHT);

        step(&mut env, &mut buffers, RIGHT);
        assert!(env.state.scatter_mode);
        assert!(env.state.reverse_directions);
        for (ghost, corner) in env.state.ghosts.iter().zip(GHOST_CORNERS) {
            assert_eq!(ghost.target, corner);
        }

        step(&mut env, &mut buffers, RIGHT);
        step(&mut env, &mut buffers, RIGHT);
        assert!(env.state.scatter_mode);
        // targets are set before Pac-Man moves, so Blinky chases where he was
        let player = env.state.player;
        step(&mut env, &mut buffers, RIGHT);
        assert!(!env.state.scatter_mode);
        assert!(env.state.reverse_directions);
        assert_eq!(env.state.ghosts[BLINKY].target, player);
    }

    /// After `max_mode_changes` flips the schedule settles on chase and stays
    /// there, as the config documents.
    #[test]
    fn the_schedule_settles_on_chase_after_the_last_mode_change() {
        let config = PacmanConfig {
            scatter_mode_length: 1,
            chase_mode_length: 1,
            max_mode_changes: 2,
            ..PacmanConfig::default()
        };
        let mut env = Pacman::new(&config, 512);
        let mut buffers = setup(&mut env, pos(13, 7), RIGHT);

        step(&mut env, &mut buffers, RIGHT);
        assert!(env.state.scatter_mode);
        assert_eq!(env.state.mode_changes, 1);

        step(&mut env, &mut buffers, RIGHT);
        assert!(!env.state.scatter_mode);
        assert_eq!(env.state.mode_changes, 2);

        // the schedule is frozen now, so no further flips
        for _ in 0..4 {
            step(&mut env, &mut buffers, RIGHT);
            assert!(!env.state.scatter_mode);
            assert_eq!(env.state.mode_changes, 2);
        }
    }

    #[test]
    fn chase_targets_follow_each_personality() {
        let mut env = test_env();
        setup(&mut env, pos(6, 5), UP);
        env.state.ghosts[BLINKY].pos = pos(10, 13);
        env.state.ghosts[CLYDE].pos = pos(26, 29);
        env.set_chase_targets();

        let ghosts = &env.state.ghosts;
        assert_eq!(ghosts[BLINKY].target, pos(6, 5));
        assert_eq!(ghosts[PINKY].target, pos(6, 9));
        // twice the vector from Blinky to two cells ahead of Pac-Man
        assert_eq!(ghosts[INKY].target, pos(2, 1));
        // far away, Clyde chases; up close he heads for his corner
        assert_eq!(ghosts[CLYDE].target, pos(6, 5));
        env.state.ghosts[CLYDE].pos = pos(6, 9);
        env.set_chase_targets();
        assert_eq!(env.state.ghosts[CLYDE].target, GHOST_CORNERS[CLYDE]);
    }

    #[test]
    fn frightened_ghosts_move_at_half_speed() {
        let mut env = test_env();
        let mut buffers = setup(&mut env, pos(2, 7), LEFT);
        let blinky = &mut env.state.ghosts[BLINKY];
        blinky.start_timeout = 0;
        // on the top corridor, heading right, far from Pac-Man
        blinky.pos = pos(1, 25);
        blinky.dir = RIGHT;
        step(&mut env, &mut buffers, LEFT);
        assert!(env.state.ghosts[BLINKY].frightened);

        let mut moves = 0;
        for _ in 0..8 {
            let before = env.state.ghosts[BLINKY].pos;
            step(&mut env, &mut buffers, LEFT);
            moves += usize::from(env.state.ghosts[BLINKY].pos != before);
        }
        assert_eq!(moves, 4);
    }

    /// Fuzz the bookkeeping with live ghosts: whatever happens, nobody stands
    /// in a wall, the pellet count matches the maze, and the paint shows
    /// Pac-Man exactly once.
    #[test]
    fn rollout_invariants() {
        let config = PacmanConfig {
            randomize_starting_position: true,
            max_start_timeout: 10,
            ..PacmanConfig::default()
        };
        let mut env = Pacman::new(&config, 5000);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(5, &mut buffers.view_mut());

        let mut rng = SmallRng::seed_from_u64(17);
        let mut rounds = 0;
        for _ in 0..5000 {
            step(&mut env, &mut buffers, rng.random_range(0..4));
            rounds += usize::from(buffers.terminated[0]);

            let player = env.state.player;
            assert_ne!(env.tile(player), TileWall, "Pac-Man in a wall");
            for ghost in &env.state.ghosts {
                assert_ne!(env.tile(ghost.pos), TileWall, "a ghost in a wall");
                assert!((0..MAP_WIDTH).contains(&ghost.pos.x));
            }
            assert_eq!(env.state.remaining_pellets, pellets_left(&env));
            assert_eq!(env.state.score as usize + pellets_left(&env), PICKUPS);
            let pacmen = env
                .state
                .painted
                .iter()
                .filter(|t| PACMAN_TILES.contains(t))
                .count();
            assert_eq!(pacmen, 1);
            assert_eq!(
                view(&buffers)[cell(&env, 0, 0)],
                id(PACMAN_TILES[env.state.player_dir as usize])
            );
        }
        assert!(rounds > 0, "random play never died in 5000 steps");
        assert!(buffers.terminated[0], "the last step ends the episode");
    }

    #[test]
    fn episode_ends_on_the_last_step() {
        let mut env = Pacman::new(&PacmanConfig::default(), 4);
        let mut buffers = setup(&mut env, pos(13, 7), RIGHT);

        for time in 0..4i32 {
            step(&mut env, &mut buffers, RIGHT);
            assert_eq!(buffers.time[0], time + 1);
            assert_eq!(buffers.terminated[0], time == 3);
        }
    }

    #[test]
    fn the_render_state_is_the_painted_maze() {
        let mut env = test_env();
        setup(&mut env, pos(0, 16), LEFT);
        let mut render_state = GridRenderState::default();
        env.render_state_into(&mut render_state);

        assert_eq!(render_state.tilemap.dim(), (28, 31));
        assert_eq!(render_state.agent_positions, vec![pos(0, 16)]);
        assert_eq!(render_state.tilemap[[0, 16]], id(PacmanLeft));
        assert_eq!(render_state.tilemap[[13, 19]], id(GhostBlinky));

        let settings = env.get_render_settings();
        assert_eq!((settings.tile_width, settings.tile_height), (28, 31));
        assert_eq!(settings.view_height, 15 + UI_HEIGHT);
        assert_eq!(settings.fov_height(), 15);
    }
}
