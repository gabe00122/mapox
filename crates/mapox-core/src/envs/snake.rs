use std::collections::{HashMap, VecDeque};

use ndarray::{Array2, s};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::common::{Position, vocab_enum::VocabEnum},
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_SNAKE_BLUE, AGENT_SNAKE_GOLD, AGENT_SNAKE_GRAY, AGENT_SNAKE_GREEN,
        AGENT_SNAKE_ORANGE, AGENT_SNAKE_PINK, AGENT_SNAKE_PURPLE, AGENT_SNAKE_RED,
        AGENT_SNAKE_WHITE, AGENT_SNAKE_YELLOW, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP,
        TILE_EMPTY, TILE_FOOD, TILE_UI, TILE_WALL,
    },
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
    vocab_enum,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SnakeConfig {
    pub num_agents: usize,
    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,

    /// Every open tile independently grows a pellet with this probability
    /// on each step; occupied tiles never do. The board always starts bare.
    pub food_spawn_prob: f32,
    pub food_reward: f32,
    pub death_reward: f32,
}

impl Default for SnakeConfig {
    fn default() -> Self {
        Self {
            num_agents: 4,
            width: 24,
            height: 24,
            view_width: 11,
            view_height: 11,
            food_spawn_prob: 0.002,
            food_reward: 1.0,
            death_reward: -1.0,
        }
    }
}

/// Row i is the (dx, dy) delta for move action i, in `MOVES` order: up,
/// right, down, left. The [`SnakeAction`] variants carry the same order, so
/// a heading is also an index into this table.
const DIRECTIONS: [Position; 4] = [
    Position { x: 0, y: 1 },
    Position { x: 1, y: 0 },
    Position { x: 0, y: -1 },
    Position { x: -1, y: 0 },
];

vocab_enum!(SnakeObs {
    TileUI => TILE_UI,
    TileEmpty => TILE_EMPTY,
    TileWall => TILE_WALL,
    TileFood => TILE_FOOD,
    SnakeRed => AGENT_SNAKE_RED,
    SnakeOrange => AGENT_SNAKE_ORANGE,
    SnakeYellow => AGENT_SNAKE_YELLOW,
    SnakeGold => AGENT_SNAKE_GOLD,
    SnakeGreen => AGENT_SNAKE_GREEN,
    SnakeBlue => AGENT_SNAKE_BLUE,
    SnakePurple => AGENT_SNAKE_PURPLE,
    SnakePink => AGENT_SNAKE_PINK,
    SnakeGray => AGENT_SNAKE_GRAY,
    SnakeWhite => AGENT_SNAKE_WHITE,
});

/// The ten snake tiles in vocab order. Agent `i` wears colour `i % 10`, so
/// every body cell identifies its owner inside the single tile channel —
/// heads and tails share one tile per snake.
const SNAKE_TILES: [SnakeObs; 10] = [
    SnakeObs::SnakeRed,
    SnakeObs::SnakeOrange,
    SnakeObs::SnakeYellow,
    SnakeObs::SnakeGold,
    SnakeObs::SnakeGreen,
    SnakeObs::SnakeBlue,
    SnakeObs::SnakePurple,
    SnakeObs::SnakePink,
    SnakeObs::SnakeGray,
    SnakeObs::SnakeWhite,
];

/// The paint tile for an owner-grid value (agent index + 1).
fn snake_tile(owner: u16) -> SnakeObs {
    SNAKE_TILES[((owner - 1) as usize) % SNAKE_TILES.len()]
}

vocab_enum!(
    #[allow(clippy::enum_variant_names)]
    SnakeAction {
        MoveUp => MOVE_UP,
        MoveRight => MOVE_RIGHT,
        MoveDown => MOVE_DOWN,
        MoveLeft => MOVE_LEFT,
    }
);

#[derive(Debug, Clone)]
struct SnakeAgent {
    /// The body cells, tail first and head last. Grows without a bound: the
    /// length cap the jax version carries is a fixed-shape-array limitation,
    /// not a game rule.
    body: VecDeque<Position>,
    /// Current heading as a `DIRECTIONS` index (the move action id).
    dir: u8,
}

impl SnakeAgent {
    fn head(&self) -> Position {
        *self.body.back().expect("a snake always has a head")
    }

    fn tail(&self) -> Position {
        *self.body.front().expect("a snake always has a tail")
    }
}

/// Where one snake is going this step. Every snake is planned before any of
/// them moves: all snakes move at once, and collisions resolve against the
/// board as it looks after this step's tails have vacated.
#[derive(Debug, Clone, Copy)]
struct MovePlan {
    new_head: Position,
    /// The direction the snake actually takes (reverse/non-move coerced).
    heading: u8,
    /// Provisional: the target cell has food. A snake that eats still holds
    /// its tail for collision purposes even if it dies on the way, so this
    /// feeds the occupancy checks before being cleared for the dying.
    eats: bool,
    died: bool,
}

/// Slot marking an open cell that is not in the pool.
const POOL_NONE: u32 = u32::MAX;

#[derive(Debug, Clone)]
struct SnakeState {
    snakes: Vec<SnakeAgent>,
    plans: Vec<MovePlan>,
    time: usize,

    /// The painted map the views and the renderer read: walls, floor, food,
    /// and colour-coded snake bodies. An occupied cell paints its owner's
    /// colour over the food it stands on, so food can hide under a snake
    /// until it moves off again.
    map: Array2<SnakeObs>,
    /// Body collision layer: 0 for empty, else the owning agent index + 1.
    owner: Array2<u16>,
    /// Authoritative food layer, kept separate from the painted map because
    /// food may sit under a snake.
    food: Array2<bool>,

    /// Every open cell (no wall, no body, no food) in one pool, so spawns
    /// draw in O(1) and the food roll sweeps open tiles only.
    free: Vec<Position>,
    /// Each cell's slot in `free`, or [`POOL_NONE`], for O(1) removal.
    free_at: Array2<u32>,

    /// Scratch for the head-on checks: flat cell -> first agent. Reused and
    /// cleared each step to keep per-step allocation at zero.
    head_first: HashMap<usize, usize>,
    target_first: HashMap<usize, usize>,
}

impl SnakeState {
    /// Takes `cell` out of the open pool if it is in it.
    fn withdraw(&mut self, cell: Position) {
        let slot = self.free_at[cell.idx()];
        if slot == POOL_NONE {
            return;
        }
        let last = self.free.pop().expect("pool slot without pool");
        if last != cell {
            self.free[slot as usize] = last;
            self.free_at[last.idx()] = slot;
        }
        self.free_at[cell.idx()] = POOL_NONE;
    }

    /// Puts `cell` into the open pool if nothing is on it. Callers may hand
    /// it repeated segments of a stacked respawn body; the membership check
    /// keeps the pool duplicate-free.
    fn release(&mut self, cell: Position) {
        let idx = cell.idx();
        if self.owner[idx] == 0 && !self.food[idx] && self.free_at[idx] == POOL_NONE {
            self.free_at[idx] =
                u32::try_from(self.free.len()).expect("open pool exceeds u32 cells");
            self.free.push(cell);
        }
    }

    /// Removes and returns a uniformly chosen open cell, if one exists.
    fn draw_open(&mut self, rng: &mut SmallRng) -> Option<Position> {
        if self.free.is_empty() {
            return None;
        }
        let k = rng.random_range(0..self.free.len());
        let cell = self.free.swap_remove(k);
        if let Some(&moved) = self.free.get(k) {
            self.free_at[moved.idx()] = u32::try_from(k).expect("open pool exceeds u32 cells");
        }
        self.free_at[cell.idx()] = POOL_NONE;
        Some(cell)
    }
}

/// Multiplayer snake on an empty walled board.
///
/// Every snake paints one tile type in one of ten colours — agent `i` wears
/// colour `i % 10` — so a body cell identifies its owner in the observation
/// without a head/body or team channel. Beyond ten agents, colour repeats
/// and only the collision grid still tells the snakes apart.
///
/// All snakes move at once. Collisions are resolved against post-tail-move
/// occupancy, so a snake may safely follow a tail that vacates this step,
/// but the body of a snake dying this same step still kills. Half of a dead
/// snake's body (alternating segments from the tail) turns into food and the
/// snake respawns as a single cell on a random open tile. Reversing into your
/// own neck is masked out and, if forced, keeps the snake moving straight.
/// A snake is flagged terminated on the step it dies; every agent is
/// flagged on the episode's last step, and the board is rebuilt by `reset`.
///
/// Every body is a growable deque and every open cell sits in an index pool,
/// so a step costs work per snake plus the view-sized observation crops and
/// one pass over the open pool that rolls every tile's food spawn.
#[derive(Debug, Clone)]
pub struct Snake {
    pub config: SnakeConfig,
    state: SnakeState,
    rngs: SmallRng,

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

impl Snake {
    pub fn new(config: &SnakeConfig, length: usize) -> Self {
        assert!(
            config.num_agents < u16::MAX as usize,
            "the owner grid packs agent ids into a u16"
        );

        let action_vocab = SnakeAction::vocab();
        let obs_vocab = SnakeObs::vocab();

        let pad_width = config.view_width / 2;
        let pad_height = config.view_height / 2;
        let ui_height = 2;
        let view_height = config.view_height + ui_height;

        let width = config.width + 2 * pad_width;
        let height = config.height + 2 * pad_height;

        let interior = (config.width * config.height) as usize;
        assert!(
            interior >= config.num_agents,
            "the board cannot hold {num_agents} snakes at {w}x{h}",
            num_agents = config.num_agents,
            w = config.width,
            h = config.height,
        );

        let dim = (width as usize, height as usize);
        let obs_spec = ObservationSpec::new(config.view_width, view_height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        let plan = MovePlan {
            new_head: Position::default(),
            heading: 0,
            eats: false,
            died: false,
        };

        Self {
            config: config.clone(),
            state: SnakeState {
                snakes: Vec::with_capacity(config.num_agents),
                plans: vec![plan; config.num_agents],
                time: 0,
                map: Array2::from_elem(dim, SnakeObs::TileWall),
                owner: Array2::zeros(dim),
                food: Array2::from_elem(dim, false),
                free: Vec::with_capacity(interior),
                free_at: Array2::from_elem(dim, POOL_NONE),
                head_first: HashMap::with_capacity(config.num_agents),
                target_first: HashMap::with_capacity(config.num_agents),
            },
            rngs: SmallRng::seed_from_u64(0),
            length,

            pad_width,
            pad_height,
            width,
            height,

            obs_spec,
            action_spec,

            action_vocab,
            obs_vocab,
        }
    }

    /// The wall padding is the only wall on this empty map.
    fn is_wall(&self, p: Position) -> bool {
        p.x < self.pad_width
            || p.y < self.pad_height
            || p.x >= self.width - self.pad_width
            || p.y >= self.height - self.pad_height
    }

    fn flat(&self, p: Position) -> usize {
        p.x as usize * self.height as usize + p.y as usize
    }

    /// Clears a dead snake off the body and paint layers, turning every
    /// other segment (counting from the tail) into food. The body
    /// list is kept: the caller respawns it, and the corpse cells are the
    /// fallback spawn candidates when the open pool is empty.
    fn dissolve(&mut self, agent_id: usize) {
        let len = self.state.snakes[agent_id].body.len();
        for j in 0..len {
            let cell = self.state.snakes[agent_id].body[j];
            self.state.owner[cell.idx()] = 0;
        }
        for j in 0..len {
            let cell = self.state.snakes[agent_id].body[j];
            if j % 2 == 0 {
                self.state.food[cell.idx()] = true;
            }
            let has_food = self.state.food[cell.idx()];
            self.state.map[cell.idx()] = if has_food {
                SnakeObs::TileFood
            } else {
                SnakeObs::TileEmpty
            };
            if has_food {
                self.state.withdraw(cell);
            } else {
                self.state.release(cell);
            }
        }
    }

    /// Moves one live snake onto its planned head cell.
    fn advance(&mut self, agent_id: usize) {
        let plan = self.state.plans[agent_id];
        let new_head = plan.new_head;
        let grows = plan.eats;
        let id = (agent_id + 1) as u16;

        if !grows {
            // The tail vacates; without growth the body slides forward.
            let tail = self.state.snakes[agent_id].body.pop_front().unwrap();
            // The cell only empties if this snake still owns it: an
            // earlier advance this phase may have moved another snake's
            // head onto the vacating tail.
            let mine = self.state.owner[tail.idx()] == id;
            if mine {
                self.state.owner[tail.idx()] = 0;
                // Food can hide under a body (a head can enter a cell that
                // turned into food on the very step it was entered), so a
                // vacated cell repaints to whatever the food layer says.
                let has_food = self.state.food[tail.idx()];
                self.state.map[tail.idx()] = if has_food {
                    SnakeObs::TileFood
                } else {
                    SnakeObs::TileEmpty
                };
                self.state.release(tail);
            }
        }

        // Head, neck, and tail all paint the same colour tile, so the only
        // repaints this move needs are the vacated tail and the new head.
        if grows {
            self.state.food[new_head.idx()] = false;
        }
        self.state.snakes[agent_id].body.push_back(new_head);
        self.state.owner[new_head.idx()] = id;
        self.state.map[new_head.idx()] = snake_tile(id);
        self.state.withdraw(new_head);
        self.state.snakes[agent_id].dir = plan.heading;
    }

    /// Respawns a dead snake as a single cell on a fresh open tile; the
    /// body grows out of the spawn from there. If the pool is empty — the
    /// board entirely full of bodies and pellets — fall back to one of the
    /// corpse's own cells, skipping the tail: the tail may have been
    /// legally entered by a living snake this step ("follow a vacating
    /// tail"), while phase 2 kills anything aiming at a non-tail body
    /// cell, so a mid-body cell cannot carry a living head. A length-1
    /// corpse has only the tail, so the fallback still cannot save that
    /// case on a truly full board.
    fn respawn(&mut self, agent_id: usize) {
        let spawn = self.state.draw_open(&mut self.rngs).unwrap_or_else(|| {
            let corpse = &self.state.snakes[agent_id].body;
            corpse
                .iter()
                .skip(1)
                .copied()
                .find(|cell| {
                    !self
                        .state
                        .plans
                        .iter()
                        .enumerate()
                        .any(|(j, plan)| j != agent_id && !plan.died && plan.new_head == *cell)
                })
                .or_else(|| corpse.front().copied())
                .expect("a dead snake still holds its corpse")
        });
        let dir = self.rngs.random_range(0..4) as u8;

        self.state.snakes[agent_id].body = [spawn].into();
        self.state.snakes[agent_id].dir = dir;

        self.state.owner[spawn.idx()] = (agent_id + 1) as u16;
        self.state.map[spawn.idx()] = snake_tile((agent_id + 1) as u16);
        self.state.withdraw(spawn);
    }

    /// Rolls every open tile for a pellet, each tile independently with
    /// `food_spawn_prob`. Occupied tiles are not in the pool, so they never
    /// spawn; walking the pool back to front means `withdraw`'s swap of the
    /// last cell into the hole only ever moves cells the cursor has already
    /// rolled past, so no tile is skipped and none is rolled twice.
    fn spawn_food(&mut self) {
        let prob = f64::from(self.config.food_spawn_prob);
        if prob <= 0.0 {
            return;
        }
        for slot in (0..self.state.free.len()).rev() {
            if self.rngs.random_bool(prob) {
                let cell = self.state.free[slot];
                self.state.food[cell.idx()] = true;
                self.state.map[cell.idx()] = SnakeObs::TileFood;
                self.state.withdraw(cell);
            }
        }
    }

    fn encode_observations(&self, timestep: &mut TimeStepMut) {
        let fov_height = self.config.view_height as usize;

        for (agent_id, snake) in self.state.snakes.iter().enumerate() {
            // Wall padding keeps the window inside the map, so every view is
            // a plain crop centred on the head: nothing to raycast or mask.
            let head = snake.head();
            let x0 = (head.x - self.pad_width) as usize;
            let y0 = (head.y - self.pad_height) as usize;

            let mut view = timestep.obs.slice_mut(s![agent_id, .., ..fov_height, 0]);
            let window = self.state.map.slice(s![
                x0..x0 + self.config.view_width as usize,
                y0..y0 + fov_height,
            ]);
            view.zip_mut_with(&window, |dst, &tile| *dst = tile.into());

            let mut ui = timestep.obs.slice_mut(s![agent_id, .., fov_height.., 0]);
            ui.fill(SnakeObs::TileUI as VocabId);
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length);
        timestep.task_ids.fill(0);
    }

    /// Every move is legal except the one that reverses into the neck.
    fn encode_action_mask(&self, timestep: &mut TimeStepMut) {
        timestep.action_mask.fill(true);

        for (agent_id, snake) in self.state.snakes.iter().enumerate() {
            timestep.action_mask.row_mut(agent_id)[(snake.dir as usize + 2) % 4] = false;
        }
    }
}

impl Environment for Snake {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.rngs = SmallRng::seed_from_u64(seed);
        self.state.time = 0;

        let dim = (self.width as usize, self.height as usize);
        if self.state.map.dim() != dim {
            self.state.map = Array2::from_elem(dim, SnakeObs::TileWall);
            self.state.owner = Array2::zeros(dim);
            self.state.food = Array2::from_elem(dim, false);
            self.state.free_at = Array2::from_elem(dim, POOL_NONE);
        }

        // A simple empty map: floor inside, wall padding outside.
        self.state.map.fill(SnakeObs::TileWall);
        self.state
            .map
            .slice_mut(s![
                self.pad_width as usize..(self.width - self.pad_width) as usize,
                self.pad_height as usize..(self.height - self.pad_height) as usize,
            ])
            .fill(SnakeObs::TileEmpty);
        self.state.owner.fill(0);
        self.state.food.fill(false);

        self.state.free.clear();
        self.state.free_at.fill(POOL_NONE);
        for x in self.pad_width..self.width - self.pad_width {
            for y in self.pad_height..self.height - self.pad_height {
                let cell = Position::new(x, y);
                self.state.free_at[cell.idx()] =
                    u32::try_from(self.state.free.len()).expect("open pool exceeds u32 cells");
                self.state.free.push(cell);
            }
        }

        // Every snake starts as a single cell on its own spawn, and the
        // board starts bare: pellets arrive only through the food roll and
        // dissolved corpses.
        self.state.snakes.clear();
        for agent_id in 0..self.config.num_agents {
            let spawn = self
                .state
                .draw_open(&mut self.rngs)
                .expect("checked for room in new()");
            let dir = self.rngs.random_range(0..4) as u8;

            self.state.snakes.push(SnakeAgent {
                body: [spawn].into(),
                dir,
            });
            self.state.owner[spawn.idx()] = (agent_id + 1) as u16;
            self.state.map[spawn.idx()] = snake_tile((agent_id + 1) as u16);
        }

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        let n = self.config.num_agents;

        // Phase 1: plan where every snake is going. A reversal or a
        // non-move action keeps the snake moving straight.
        for i in 0..n {
            let dir = self.state.snakes[i].dir;
            let action = actions.get(i).copied().unwrap_or(u16::MAX);
            let is_move = (action as usize) < SnakeAction::TABLE.len();
            let heading = if is_move && action as u8 != (dir + 2) % 4 {
                action as u8
            } else {
                dir
            };

            let new_head = self.state.snakes[i].head() + DIRECTIONS[heading as usize];
            let eats = self.state.food[new_head.idx()];

            self.state.plans[i] = MovePlan {
                new_head,
                heading,
                eats,
                died: false,
            };
        }

        // Phase 2: resolve deaths against the post-tail-move board, so a
        // snake may follow a tail that vacates this very step, while the
        // body of a snake dying the same step still kills.
        self.state.head_first.clear();
        self.state.target_first.clear();
        for i in 0..n {
            let head = self.state.snakes[i].head();
            self.state.head_first.insert(self.flat(head), i);
        }

        for i in 0..n {
            let target = self.state.plans[i].new_head;

            if self.is_wall(target) {
                self.state.plans[i].died = true;
                continue;
            }

            // A cell is blocked unless it is the tail of a snake that is
            // not growing this step — growing covers eating, there is no
            // length cap to make the two diverge.
            let owner = self.state.owner[target.idx()];
            if owner > 0 {
                let j = (owner - 1) as usize;
                let vacating = target == self.state.snakes[j].tail() && !self.state.plans[j].eats;
                if !vacating {
                    self.state.plans[i].died = true;
                    continue;
                }
            }

            // Two heads into one cell.
            if let Some(j) = self.state.target_first.insert(self.flat(target), i) {
                self.state.plans[i].died = true;
                self.state.plans[j].died = true;
                continue;
            }

            // Two heads trading places.
            if let Some(&j) = self.state.head_first.get(&self.flat(target))
                && self.state.plans[j].new_head == self.state.snakes[i].head()
            {
                self.state.plans[i].died = true;
                self.state.plans[j].died = true;
            }
        }

        // Phase 3: the dying neither eat nor grow.
        for i in 0..n {
            if self.state.plans[i].died {
                self.state.plans[i].eats = false;
            }
        }

        // Phase 4: mutate the world in the order the layers need. The dead
        // first, freeing their cells and dropping corpse food; then the
        // living, whose heads may land exactly where a vacated tail was;
        // then the respawns and the food roll over the settled board.
        for i in 0..n {
            if self.state.plans[i].died {
                self.dissolve(i);
            }
        }
        for i in 0..n {
            if !self.state.plans[i].died {
                self.advance(i);
            }
        }
        for i in 0..n {
            if self.state.plans[i].died {
                self.respawn(i);
            }
        }

        self.spawn_food();

        // Phase 5: report.
        self.state.time += 1;
        for i in 0..n {
            let plan = self.state.plans[i];
            timestep.last_action[i] = actions.get(i).copied().unwrap_or(0);
            timestep.reward[i] = if plan.died {
                self.config.death_reward
            } else if plan.eats {
                self.config.food_reward
            } else {
                0.0
            };
        }

        self.encode_observations(timestep);
        for i in 0..n {
            timestep.terminated[i] |= self.state.plans[i].died;
        }
        self.encode_action_mask(timestep);
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
            view_height: (self.config.view_height + 2) as usize,
            ui_height: 2,
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
        for snake in &self.state.snakes {
            let local_pos = snake.head() - Position::new(self.pad_width, self.pad_height);
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
    use SnakeObs::*;
    use rand::RngExt;
    use std::collections::HashSet;

    const UP: u8 = 0;
    const RIGHT: u8 = 1;
    const DOWN: u8 = 2;
    const LEFT: u8 = 3;
    /// Out of the action vocab: the snake coerces it to straight.
    const NON_MOVE: u8 = 9;

    fn pos(x: i32, y: i32) -> Position {
        Position::new(x, y)
    }

    fn test_config() -> SnakeConfig {
        SnakeConfig {
            num_agents: 2,
            width: 12,
            height: 12,
            food_spawn_prob: 0.0,
            ..Default::default()
        }
    }

    fn test_env() -> Snake {
        Snake::new(&test_config(), 512)
    }

    /// A bare board with exact snake placements — cells tail first, head
    /// last — and pellets, bypassing the randomised reset.
    fn setup(env: &mut Snake, snakes: &[(&[Position], u8)], food: &[Position]) {
        let mut buffers = TimeStepBuffers::new(&*env);
        env.reset(0, &mut buffers.view_mut());

        env.state.map.fill(TileWall);
        env.state.owner.fill(0);
        env.state.food.fill(false);
        env.state.free.clear();
        env.state.free_at.fill(POOL_NONE);
        for x in env.pad_width..env.width - env.pad_width {
            for y in env.pad_height..env.height - env.pad_height {
                let cell = pos(x, y);
                env.state.map[cell.idx()] = TileEmpty;
                env.state.free_at[cell.idx()] = env.state.free.len() as u32;
                env.state.free.push(cell);
            }
        }

        env.state.snakes.clear();
        for (agent_id, (cells, dir)) in snakes.iter().enumerate() {
            let id = (agent_id + 1) as u16;
            for &cell in *cells {
                env.state.owner[cell.idx()] = id;
                env.state.map[cell.idx()] = snake_tile(id);
                env.state.withdraw(cell);
            }
            env.state.snakes.push(SnakeAgent {
                body: cells.iter().copied().collect(),
                dir: *dir,
            });
        }

        for &cell in food {
            env.state.food[cell.idx()] = true;
            env.state.map[cell.idx()] = TileFood;
            env.state.withdraw(cell);
        }

        env.state.time = 0;
    }

    fn step(env: &mut Snake, buffers: &mut TimeStepBuffers, actions: &[u8]) {
        let actions: Vec<VocabId> = actions.iter().map(|&a| a as VocabId).collect();
        env.step(&actions, &mut buffers.view_mut());
    }

    /// The cells the agent's body covers, tail first.
    fn cells(env: &Snake, agent: usize) -> Vec<Position> {
        env.state.snakes[agent].body.iter().copied().collect()
    }

    fn id(tile: SnakeObs) -> VocabId {
        tile.into()
    }

    /// The layers agree with each other and with the bodies: the owner grid
    /// is exactly the segment multiset, the paint follows it, and the open
    /// pool holds exactly the cells nothing is on.
    fn check_consistency(env: &Snake) {
        let (w, h) = env.state.map.dim();
        let mut owned: HashSet<usize> = HashSet::new();

        for (agent_id, snake) in env.state.snakes.iter().enumerate() {
            assert!(!snake.body.is_empty());
            for j in 1..snake.body.len() {
                let (a, b) = (snake.body[j - 1], snake.body[j]);
                assert!(
                    (a.x - b.x).abs() + (a.y - b.y).abs() <= 1,
                    "segments {a:?} {b:?} are not adjacent"
                );
            }
            let head = snake.head();
            assert!(!env.is_wall(head), "head sits on a wall");
            let id = (agent_id + 1) as u16;
            for j in 0..snake.body.len() {
                let cell = snake.body[j];
                assert_eq!(env.state.map[cell.idx()], snake_tile(id));
            }
            let mut snake_cells: HashSet<usize> = HashSet::new();
            for j in 0..snake.body.len() {
                let cell = snake.body[j];
                let flat = env.flat(cell);
                assert_eq!(env.state.owner[cell.idx()], (agent_id + 1) as u16);
                // Repeats within one snake are legal only as the stacked
                // cells of a respawn body; across snakes nothing may share.
                assert!(
                    snake_cells.insert(flat) || owned.contains(&flat),
                    "snakes share a cell"
                );
                owned.insert(flat);
            }
        }

        for x in 0..w {
            for y in 0..h {
                let cell = pos(x as i32, y as i32);
                let open = !env.is_wall(cell)
                    && env.state.owner[cell.idx()] == 0
                    && !env.state.food[cell.idx()];
                let in_pool = env.state.free_at[cell.idx()] != POOL_NONE;
                assert_eq!(in_pool, open, "open pool disagrees at {cell:?}");
                if in_pool {
                    let slot = env.state.free_at[cell.idx()] as usize;
                    assert_eq!(env.state.free[slot], cell, "stale pool slot at {cell:?}");
                }
                if env.state.food[cell.idx()] {
                    // Food is either painted or hidden under a snake.
                    assert!(!matches!(env.state.map[cell.idx()], TileEmpty | TileWall));
                }
                if env.is_wall(cell) {
                    assert_eq!(env.state.map[cell.idx()], TileWall);
                    assert_eq!(env.state.owner[cell.idx()], 0);
                }
            }
        }
    }

    #[test]
    fn moves_forward() {
        let mut env = test_env();
        setup(
            &mut env,
            &[(&[pos(8, 8), pos(9, 8)], RIGHT), (&[pos(15, 15)], UP)],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[RIGHT, UP]);

        assert_eq!(cells(&env, 0), vec![pos(9, 8), pos(10, 8)]);
        assert!(!buffers.terminated[0]);
        assert_eq!(buffers.reward[0], 0.0);
        check_consistency(&env);
    }

    #[test]
    fn reverse_is_coerced_straight() {
        let mut env = test_env();
        setup(
            &mut env,
            &[(&[pos(8, 8), pos(9, 8)], RIGHT), (&[pos(15, 15)], UP)],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[LEFT, UP]);

        assert_eq!(cells(&env, 0), vec![pos(9, 8), pos(10, 8)]);
        assert_eq!(env.state.snakes[0].dir, RIGHT);
        assert!(!buffers.terminated[0]);
    }

    #[test]
    fn non_move_is_coerced_straight() {
        let mut env = test_env();
        setup(
            &mut env,
            &[(&[pos(8, 8), pos(9, 8)], RIGHT), (&[pos(15, 15)], UP)],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[NON_MOVE, UP]);

        assert_eq!(env.state.snakes[0].head(), pos(10, 8));
        check_consistency(&env);
    }

    #[test]
    fn eating_grows() {
        let mut env = test_env();
        setup(
            &mut env,
            &[(&[pos(8, 8), pos(9, 8)], RIGHT), (&[pos(15, 15)], UP)],
            &[pos(10, 8)],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[RIGHT, UP]);

        assert_eq!(cells(&env, 0), vec![pos(8, 8), pos(9, 8), pos(10, 8)]);
        assert_eq!(buffers.reward[0], env.config.food_reward);
        assert!(!env.state.food[pos(10, 8).idx()]);
        check_consistency(&env);
    }

    #[test]
    fn growth_is_unbounded() {
        // The jax port caps bodies at max_length to keep the state a fixed
        // shape; the rust bodies are growable vectors and do not.
        let config = SnakeConfig {
            num_agents: 1,
            width: 120,
            height: 24,
            ..test_config()
        };
        let mut env = Snake::new(&config, 512);
        let lane: Vec<Position> = (14..14 + 80).map(|x| pos(x, 8)).collect();
        setup(
            &mut env,
            &[(&[pos(11, 8), pos(12, 8), pos(13, 8)], RIGHT)],
            &lane,
        );

        let mut buffers = TimeStepBuffers::new(&env);
        for _ in 0..80 {
            step(&mut env, &mut buffers, &[RIGHT]);
        }

        assert_eq!(env.state.snakes[0].head(), pos(93, 8));
        assert_eq!(env.state.snakes[0].body.len(), 83);
        assert_eq!(buffers.reward[0], env.config.food_reward);
        check_consistency(&env);
    }

    #[test]
    fn wall_death_drops_food() {
        // Playable x spans 5..=16, so moving right from 16 hits the wall.
        let mut env = test_env();
        setup(
            &mut env,
            &[(&[pos(15, 8), pos(16, 8)], RIGHT), (&[pos(15, 15)], UP)],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[RIGHT, UP]);

        assert!(buffers.terminated[0]);
        assert_eq!(buffers.reward[0], env.config.death_reward);
        // Alternating segments from the tail become food: here just the tail.
        assert!(env.state.food[pos(15, 8).idx()]);
        assert!(!env.state.food[pos(16, 8).idx()]);
        // Respawned fresh as a lone cell somewhere else.
        assert_eq!(env.state.snakes[0].body.len(), 1);
        assert_ne!(env.state.snakes[0].head(), pos(17, 8));
        check_consistency(&env);
    }

    #[test]
    fn respawn_fallback_picks_untargeted_corpse_cell() {
        // The open pool must be empty at respawn time: fill every
        // non-body cell with food, with hidden pellets under the dying
        // snake's tail (feeding the follower, so its tail holds and the
        // pool cannot re-open) and head (so dissolving cannot re-open it
        // either). The fallback then picks an un-targeted corpse cell.
        let mut env = test_env();
        let tail = pos(15, 8);
        let head = pos(16, 8);
        let follower = pos(14, 8);
        let mut food = vec![tail, head];
        for x in env.pad_width..env.width - env.pad_width {
            for y in env.pad_height..env.height - env.pad_height {
                let cell = pos(x, y);
                if cell != tail && cell != head && cell != follower {
                    food.push(cell);
                }
            }
        }
        setup(
            &mut env,
            &[(&[tail, head], RIGHT), (&[follower], RIGHT)],
            &food,
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[RIGHT, RIGHT]);

        assert!(buffers.terminated[0]);
        assert!(!buffers.terminated[1]);
        // The follower grew onto the old tail. The fallback must land the
        // respawn on the old head, the only un-targeted corpse cell, not
        // on the tail the living snake now owns.
        assert_eq!(env.state.snakes[1].head(), tail);
        assert_eq!(buffers.reward[1], env.config.food_reward);
        assert_eq!(env.state.owner[tail.idx()], 2);
        assert_eq!(cells(&env, 0), vec![head]);
        assert_eq!(env.state.owner[head.idx()], 1);
        check_consistency(&env);
    }

    #[test]
    fn body_collision_kills() {
        let mut env = test_env();
        setup(
            &mut env,
            &[
                (&[pos(8, 8), pos(9, 8)], RIGHT),
                (&[pos(10, 10), pos(10, 9), pos(10, 8)], DOWN),
            ],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[RIGHT, DOWN]);

        assert!(buffers.terminated[0]);
        assert!(!buffers.terminated[1]);
        check_consistency(&env);
    }

    #[test]
    fn head_on_collision_kills_both() {
        let mut env = test_env();
        setup(
            &mut env,
            &[
                (&[pos(8, 8), pos(9, 8)], RIGHT),
                (&[pos(12, 8), pos(11, 8)], LEFT),
            ],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[RIGHT, LEFT]);

        assert!(buffers.terminated[0]);
        assert!(buffers.terminated[1]);
        assert_ne!(env.state.snakes[0].head(), env.state.snakes[1].head());
        assert!(env.state.snakes.iter().all(|s| s.body.len() == 1));
        check_consistency(&env);
    }

    #[test]
    fn swap_collision_kills_both() {
        // Length-1 snakes passing through each other: only the swap rule
        // catches this, since both cells vacate this step.
        let mut env = test_env();
        setup(
            &mut env,
            &[(&[pos(9, 8)], RIGHT), (&[pos(10, 8)], LEFT)],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[RIGHT, LEFT]);

        assert!(buffers.terminated[0]);
        assert!(buffers.terminated[1]);
        check_consistency(&env);
    }

    #[test]
    fn can_follow_own_tail() {
        // A 4-cell snake looping through a 2x2 block: the head enters the
        // tail cell exactly as it vacates.
        let mut env = test_env();
        setup(
            &mut env,
            &[
                (&[pos(8, 8), pos(8, 9), pos(9, 9), pos(9, 8)], DOWN),
                (&[pos(15, 15)], UP),
            ],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[LEFT, UP]);

        assert!(!buffers.terminated[0]);
        assert_eq!(env.state.snakes[0].head(), pos(8, 8));
        check_consistency(&env);
    }

    #[test]
    fn own_body_collision_kills() {
        // Same loop, one cell longer: the target is the neck, which does not
        // vacate this step.
        let mut env = test_env();
        setup(
            &mut env,
            &[
                (
                    &[pos(8, 7), pos(8, 8), pos(8, 9), pos(9, 9), pos(9, 8)],
                    DOWN,
                ),
                (&[pos(15, 15)], UP),
            ],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[LEFT, UP]);

        assert!(buffers.terminated[0]);
        check_consistency(&env);
    }

    #[test]
    fn dying_next_to_a_corpse_still_dies() {
        // Both snakes drive into cells of the one that dies this step.
        let mut env = test_env();
        setup(
            &mut env,
            &[
                (&[pos(8, 8), pos(9, 8)], RIGHT),
                (&[pos(12, 8), pos(11, 8), pos(10, 8)], LEFT),
            ],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[RIGHT, LEFT]);

        assert!(buffers.terminated[0]);
        assert!(buffers.terminated[1]);
        check_consistency(&env);
    }

    #[test]
    fn action_mask_blocks_reverse() {
        let mut env = test_env();
        setup(
            &mut env,
            &[(&[pos(8, 8), pos(9, 8)], RIGHT), (&[pos(15, 15)], UP)],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[RIGHT, UP]);

        assert_eq!(buffers.action_mask.ncols(), 4);
        let mask = buffers.action_mask.row(0);
        assert!(!mask[LEFT as usize]);
        assert!(mask[UP as usize] && mask[RIGHT as usize] && mask[DOWN as usize]);
    }

    #[test]
    fn observation_is_egocentric() {
        let mut env = test_env();
        setup(
            &mut env,
            &[
                (&[pos(8, 8), pos(9, 8)], RIGHT),
                (&[pos(12, 8), pos(11, 8)], LEFT),
            ],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        env.encode_observations(&mut buffers.view_mut());

        let (cx, cy) = (env.config.view_width / 2, env.config.view_height / 2);
        let view = buffers.obs.slice(s![0, .., .., 0]);
        // Own head at the centre and body behind it, both in own colour.
        assert_eq!(view[[cx as usize, cy as usize]], id(SnakeRed));
        assert_eq!(view[[(cx - 1) as usize, cy as usize]], id(SnakeRed));
        // The other snake, two cells ahead, wears the next colour.
        assert_eq!(view[[(cx + 2) as usize, cy as usize]], id(SnakeOrange));
        // The wall padding frames the view.
        assert_eq!(view[[0, cy as usize]], id(TileWall));
    }

    /// The UI band is a placeholder, but it has to be a consistent one: the
    /// bottom rows of every agent's window carry the UI tile and nothing
    /// else, and the field of view stops short of them rather than running
    /// under.
    #[test]
    fn the_ui_band_caps_every_window() {
        let mut env = test_env();
        setup(
            &mut env,
            &[(&[pos(8, 8), pos(9, 8)], RIGHT), (&[pos(15, 15)], UP)],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        env.encode_observations(&mut buffers.view_mut());

        let fov_height = env.config.view_height as usize;
        assert_eq!(buffers.obs.dim().1, env.config.view_width as usize);
        assert_eq!(buffers.obs.dim().2, fov_height + 2);

        for agent_id in 0..env.num_agents() {
            let window = buffers.obs.slice(s![agent_id, .., .., 0]);
            for y in 0..fov_height + 2 {
                let row_is_ui =
                    (0..env.config.view_width as usize).all(|x| window[[x, y]] == id(TileUI));
                let row_has_ui =
                    (0..env.config.view_width as usize).any(|x| window[[x, y]] == id(TileUI));

                if y >= fov_height {
                    assert!(row_is_ui, "row {y} of agent {agent_id} is not all UI");
                } else {
                    assert!(!row_has_ui, "UI leaked into fov row {y}");
                }
            }
        }
    }

    #[test]
    fn reset_starts_lone_snakes_on_a_bare_board() {
        let mut env = test_env();
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(7, &mut buffers.view_mut());

        assert_eq!(env.state.snakes.len(), env.config.num_agents);
        let heads: HashSet<_> = env.state.snakes.iter().map(|s| s.head().idx()).collect();
        assert_eq!(heads.len(), env.config.num_agents, "snakes share a spawn");
        for snake in &env.state.snakes {
            assert_eq!(snake.body.len(), 1);
        }
        assert_eq!(env.state.food.iter().filter(|&&f| f).count(), 0);
        assert_eq!(buffers.reward.iter().sum::<f32>(), 0.0);
        check_consistency(&env);
    }

    /// At probability 1 the food roll covers every open tile: each tile is
    /// rolled independently, and the tiles the snakes stand on never spawn.
    #[test]
    fn food_grows_on_every_open_tile() {
        let config = SnakeConfig {
            food_spawn_prob: 1.0,
            ..test_config()
        };
        let mut env = Snake::new(&config, 512);
        setup(
            &mut env,
            &[(&[pos(9, 8)], RIGHT), (&[pos(15, 15)], UP)],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[RIGHT, UP]);

        assert!(env.state.free.is_empty());
        for x in env.pad_width..env.width - env.pad_width {
            for y in env.pad_height..env.height - env.pad_height {
                let cell = pos(x, y);
                let open = env.state.owner[cell.idx()] == 0;
                assert_eq!(env.state.food[cell.idx()], open, "tile {cell:?}");
            }
        }
        check_consistency(&env);
    }

    #[test]
    fn food_never_spawns_at_probability_zero() {
        let mut env = test_env();
        setup(
            &mut env,
            &[(&[pos(9, 8)], RIGHT), (&[pos(15, 10)], LEFT)],
            &[],
        );
        let mut buffers = TimeStepBuffers::new(&env);
        for _ in 0..6 {
            step(&mut env, &mut buffers, &[RIGHT, LEFT]);
        }

        assert_eq!(env.state.food.iter().filter(|&&f| f).count(), 0);
        check_consistency(&env);
    }

    #[test]
    fn rollout_invariants() {
        // Fuzz the bookkeeping: random legal actions for many steps,
        // checking every layer agrees after each one.
        let config = SnakeConfig {
            num_agents: 4,
            width: 12,
            height: 12,
            food_spawn_prob: 0.3,
            ..Default::default()
        };
        let mut env = Snake::new(&config, 100);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(3, &mut buffers.view_mut());

        let mut rng = SmallRng::seed_from_u64(11);
        for _ in 0..200 {
            let actions: Vec<u8> = (0..env.num_agents())
                .map(|i| {
                    let legal: Vec<u8> = (0..4)
                        .filter(|a| buffers.action_mask[[i, *a as usize]])
                        .map(|a| a as u8)
                        .collect();
                    let pick = rng.random_range(0..legal.len());
                    legal[pick]
                })
                .collect();
            step(&mut env, &mut buffers, &actions);
            check_consistency(&env);
        }
        assert!(env.state.time >= 200);
    }

    /// An episode ends on its `length`-th step — the flag fires when `time`
    /// reaches `length`, the convention every env in this crate shares.
    #[test]
    fn episode_ends_on_the_last_step() {
        let mut env = Snake::new(&test_config(), 4);
        setup(&mut env, &[(&[pos(10, 5)], UP), (&[pos(15, 5)], UP)], &[]);
        let mut buffers = TimeStepBuffers::new(&env);

        for time in 0..4i32 {
            step(&mut env, &mut buffers, &[UP, UP]);
            assert_eq!(buffers.time[0], time + 1);
            assert_eq!(buffers.terminated[0], time == 3);
        }
    }
}
