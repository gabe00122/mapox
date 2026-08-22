use ndarray::{Array2, s};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::common::{
        Position, fov,
        map_gen::{fractal_noise, sprinkle_decor},
        vocab_enum::VocabEnum,
    },
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_GENERIC, DIG_ACTION, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, NOOP, PLACE_PIPE,
        TILE_DECOR_1, TILE_DECOR_2, TILE_DECOR_3, TILE_DECOR_4, TILE_DESTRUCTIBLE_WALL, TILE_EMPTY,
        TILE_FLAG, TILE_FLAG_UNLOCKED, TILE_MASK, TILE_PIPE_HORIZONTAL, TILE_PIPE_VIRTICAL,
        TILE_UI, TILE_WALL, TILE_WATER,
    },
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
    vocab_enum,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FindReturnConfig {
    pub num_agents: usize,
    pub num_flags: usize,

    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,

    pub mapgen_threshold: f32,
    pub water_threshold: f32,
    pub digging_timeout: u32,
    pub preparation_steps: usize,
    pub treasure_reward: f32,
}

impl Default for FindReturnConfig {
    fn default() -> Self {
        Self {
            num_agents: 8,
            num_flags: 1,
            width: 40,
            height: 40,
            view_width: 11,
            view_height: 11,
            mapgen_threshold: 0.3,
            water_threshold: -0.45,
            digging_timeout: 5,
            preparation_steps: 256,
            treasure_reward: 1.0,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct FindReturnAgent {
    position: Position,
    dir: Position,
    timeout: u32,
}

impl FindReturnAgent {
    fn slide_target(&self, map: &Array2<FindReturnObs>, dir: Position) -> Position {
        let mut target = self.position + dir;
        while (dir.y == 0 && map[target.idx()] == FindReturnObs::PipeHorizontal)
            | (dir.x == 0 && map[target.idx()] == FindReturnObs::PipeVirtical)
        {
            target += dir;
        }
        target
    }
}

#[derive(Debug, Clone)]
struct FindReturnState {
    rngs: SmallRng,
    agents: Vec<FindReturnAgent>,
    agent_order: Vec<usize>, // agent turn order
    time: usize,

    base_map: Array2<FindReturnObs>, // the bottom layer of the map without agents
    map: Array2<FindReturnObs>,      // the base map plus the agents
    free_positions: Vec<Position>,   // these are used to calculate spawn positions
    flag_positions: Vec<Position>,   // where the flags sit, for unlocking them in place
}

vocab_enum!(FindReturnObs {
    UI => TILE_UI,
    Mask => TILE_MASK,
    TileEmpty => TILE_EMPTY,
    TileDestructibleWall => TILE_DESTRUCTIBLE_WALL,
    TileWall => TILE_WALL,
    TileWater => TILE_WATER,
    TileFlag => TILE_FLAG,
    TileFlagUnlocked => TILE_FLAG_UNLOCKED,
    TileDecor1 => TILE_DECOR_1,
    TileDecor2 => TILE_DECOR_2,
    TileDecor3 => TILE_DECOR_3,
    TileDecor4 => TILE_DECOR_4,
    PipeHorizontal => TILE_PIPE_HORIZONTAL,
    PipeVirtical => TILE_PIPE_VIRTICAL,
    AgentGeneric => AGENT_GENERIC,
});

impl FindReturnObs {
    fn blocked(self) -> bool {
        use FindReturnObs::*;
        matches!(
            self,
            TileWall | TileDestructibleWall | TileWater | AgentGeneric // | PipeHorizontal | PipeVirtical
        )
    }

    /// Water is the one blocking tile an agent can see straight over.
    fn opaque(self) -> bool {
        use FindReturnObs::*;
        matches!(self, TileWall | TileDestructibleWall)
    }

    fn destructible(self) -> bool {
        use FindReturnObs::*;
        matches!(self, TileDestructibleWall | PipeHorizontal | PipeVirtical)
    }
}

vocab_enum!(FindReturnAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
    PlacePipe => PLACE_PIPE,
    Dig => DIG_ACTION,
    Noop => NOOP,
});

impl FindReturnAction {
    fn direction(self) -> Position {
        use FindReturnAction::*;
        match self {
            MoveUp => Position::new(0, 1),
            MoveRight => Position::new(1, 0),
            MoveDown => Position::new(0, -1),
            MoveLeft => Position::new(-1, 0),
            _ => Position::new(0, 0),
        }
    }

    fn is_move(self) -> bool {
        use FindReturnAction::*;
        matches!(self, MoveUp | MoveRight | MoveDown | MoveLeft)
    }
}

#[derive(Debug, Clone)]
pub struct FindReturn {
    pub config: FindReturnConfig,
    state: FindReturnState,

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

impl FindReturn {
    pub fn new(config: &FindReturnConfig, length: usize) -> Self {
        let action_vocab = FindReturnAction::vocab();
        let obs_vocab = FindReturnObs::vocab();

        let pad_width = config.view_width / 2;
        let pad_height = config.view_height / 2;

        let width = config.width + 2 * pad_width;
        let height = config.height + 2 * pad_height;

        let obs_spec = ObservationSpec::new(config.view_width, config.view_height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        Self {
            config: config.clone(),
            state: FindReturnState {
                agents: Vec::with_capacity(config.num_agents),
                agent_order: (0..config.num_agents).collect(),
                base_map: Array2::from_elem(
                    (width as usize, height as usize),
                    FindReturnObs::TileEmpty,
                ),
                free_positions: Vec::new(),
                flag_positions: Vec::with_capacity(config.num_flags),
                map: Array2::from_elem((width as usize, height as usize), FindReturnObs::TileEmpty),
                rngs: SmallRng::seed_from_u64(0),
                time: 0,
            },
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

    fn calculate_free_positions(&mut self) {
        self.state.free_positions.clear();
        for x in self.pad_width..self.width - self.pad_width {
            for y in self.pad_height..self.height - self.pad_height {
                let position = Position::new(x, y);
                let tile = self.state.map[position.idx()];

                if !tile.blocked() {
                    self.state.free_positions.push(position);
                }
            }
        }

        self.state.free_positions.shuffle(&mut self.state.rngs);
    }

    fn unlock_flags(&mut self) {
        let FindReturnState {
            flag_positions,
            base_map,
            map,
            ..
        } = &mut self.state;

        for position in flag_positions.iter() {
            base_map[position.idx()] = FindReturnObs::TileFlagUnlocked;
            if map[position.idx()] == FindReturnObs::TileFlag {
                map[position.idx()] = FindReturnObs::TileFlagUnlocked;
            }
        }
    }

    fn encode_observations(&self, timestep: &mut TimeStepMut) {
        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            // wall padding keeps the view window inside the map
            let mut view = timestep.obs.slice_mut(s![agent_id, .., ..15, 0]);
            fov::encode_visible(
                &self.state.map,
                agent.position,
                &mut view,
                FindReturnObs::Mask,
                |tile| tile.opaque(),
            );

            let mut ui = timestep.obs.slice_mut(s![agent_id, .., 15.., 0]);
            ui.fill(FindReturnObs::UI as VocabId);
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length - 1);
        timestep.task_ids.fill(0);
    }

    fn encode_action_mask(&self, timestep: &mut TimeStepMut) {
        timestep.action_mask.fill(true);

        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            let mut mask = timestep.action_mask.row_mut(agent_id);

            if agent.timeout > 0 {
                mask.fill(false);
                mask[FindReturnAction::Noop as usize] = true;
                continue;
            }

            for &action in FindReturnAction::TABLE {
                let valid = match action {
                    FindReturnAction::MoveUp
                    | FindReturnAction::MoveRight
                    | FindReturnAction::MoveDown
                    | FindReturnAction::MoveLeft => {
                        let target = agent.slide_target(&self.state.map, action.direction());
                        !self.state.map[target.idx()].blocked()
                    }
                    FindReturnAction::Dig => {
                        let target = agent.position + agent.dir;
                        self.state.map[target.idx()].destructible()
                    }
                    FindReturnAction::PlacePipe => {
                        let target = agent.position + agent.dir;
                        !self.state.map[target.idx()].blocked()
                    }
                    FindReturnAction::Noop => true,
                };
                mask[action as usize] = valid;
            }
        }
    }
}

impl Environment for FindReturn {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.state.rngs = SmallRng::seed_from_u64(seed);

        self.state.time = 0;

        let dim = (self.width as usize, self.height as usize);
        if self.state.base_map.dim() != dim {
            self.state.map = Array2::from_elem(dim, FindReturnObs::TileEmpty);
            self.state.base_map = Array2::from_elem(dim, FindReturnObs::TileEmpty);
        }

        self.state.base_map.fill(FindReturnObs::TileWall);
        let mut interior = self.state.base_map.slice_mut(s![
            self.pad_width as usize..(self.width - self.pad_width) as usize,
            self.pad_height as usize..(self.height - self.pad_height) as usize,
        ]);

        fractal_noise(
            self.config.width as usize,
            self.config.height as usize,
            &mut self.state.rngs,
            |x, y, sample| {
                interior[[x, y]] = if sample > self.config.mapgen_threshold {
                    FindReturnObs::TileDestructibleWall
                } else if sample < self.config.water_threshold {
                    FindReturnObs::TileWater
                } else {
                    FindReturnObs::TileEmpty
                }
            },
        );

        sprinkle_decor(
            interior,
            FindReturnObs::TileEmpty,
            &[
                FindReturnObs::TileDecor1,
                FindReturnObs::TileDecor2,
                FindReturnObs::TileDecor3,
                FindReturnObs::TileDecor4,
            ],
            &mut self.state.rngs,
        );

        // Base map finished
        self.state.map.assign(&self.state.base_map);

        self.state.agents.clear();
        self.calculate_free_positions();

        // Place the flags, locked until the preparation phase is over
        self.state.flag_positions.clear();
        for _ in 0..self.config.num_flags {
            let flag_position = self.state.free_positions.pop().unwrap();
            self.state.base_map[flag_position.idx()] = FindReturnObs::TileFlag;
            self.state.map[flag_position.idx()] = FindReturnObs::TileFlag;
            self.state.flag_positions.push(flag_position);
        }

        if self.config.preparation_steps == 0 {
            self.unlock_flags();
        }

        // Place the agents
        for _ in 0..self.num_agents() {
            let position = self.state.free_positions.pop().unwrap();
            self.state.agents.push(FindReturnAgent {
                position,
                ..Default::default()
            });
            self.state.map[position.idx()] = FindReturnObs::AgentGeneric;
        }

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        if self.state.time == self.config.preparation_steps {
            self.unlock_flags();
        }

        let mut agent_respawn_ids: Vec<usize> = Vec::new();
        let mut agent_moved_ids: Vec<usize> = Vec::new();
        self.state.agent_order.shuffle(&mut self.state.rngs);

        for &agent_id in &self.state.agent_order {
            let (agent, map, base_map) = (
                &mut self.state.agents[agent_id],
                &mut self.state.map,
                &mut self.state.base_map,
            );

            timestep.last_action[agent_id] = actions[agent_id];
            timestep.reward[agent_id] = 0.0;

            if agent.timeout > 0 {
                agent.timeout -= 1;
                continue;
            }

            let action = FindReturnAction::from_id(actions[agent_id]);
            if action.is_move() {
                agent.dir = action.direction();
            }

            let target = agent.slide_target(map, agent.dir);
            match action {
                FindReturnAction::MoveUp
                | FindReturnAction::MoveRight
                | FindReturnAction::MoveDown
                | FindReturnAction::MoveLeft => {
                    if !map[target.idx()].blocked() {
                        // unpaint the agent because it's moving
                        map[agent.position.idx()] = base_map[agent.position.idx()];
                        agent.position = target;
                        agent_moved_ids.push(agent_id);

                        // a locked flag is scenery: only the unlocked tile pays
                        let found_flag =
                            base_map[agent.position.idx()] == FindReturnObs::TileFlagUnlocked;
                        if found_flag {
                            agent_respawn_ids.push(agent_id);
                            timestep.reward[agent_id] = self.config.treasure_reward;
                        }
                    }
                }
                FindReturnAction::Dig => {
                    if map[target.idx()].destructible() {
                        map[target.idx()] = FindReturnObs::TileEmpty;
                        base_map[target.idx()] = FindReturnObs::TileEmpty;
                        agent.timeout = self.config.digging_timeout;
                    }
                }
                FindReturnAction::Noop => {}
                FindReturnAction::PlacePipe => {
                    let target_tile = &mut map[target.idx()];

                    if !target_tile.blocked() {
                        *target_tile = if agent.dir.x == 0 {
                            FindReturnObs::PipeHorizontal
                        } else {
                            FindReturnObs::PipeVirtical
                        };
                        base_map[target.idx()] = *target_tile;
                    }
                }
            }
        }

        if !agent_respawn_ids.is_empty() {
            self.calculate_free_positions();

            let free_positions = &mut self.state.free_positions;
            for &agent_id in &agent_respawn_ids {
                let agent = &mut self.state.agents[agent_id];
                agent.position = free_positions.pop().unwrap();
                self.state.map[agent.position.idx()] = FindReturnObs::AgentGeneric;
            }
        }

        for &agent_id in &agent_moved_ids {
            self.state.map[self.state.agents[agent_id].position.idx()] =
                FindReturnObs::AgentGeneric;
        }

        self.state.time += 1;
        self.encode_observations(timestep);
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
            view_height: self.config.view_height as usize,
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
        for agent in &self.state.agents {
            let local_pos = agent.position - Position::new(self.pad_width, self.pad_height);
            grid_render_state.agent_positions.push(local_pos);
        }
    }
}
