use ndarray::{Array2, s};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::common::Position,
    map_gen::{fractal_noise, sprinkle_decor},
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_GENERIC, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, TILE_DECOR,
        TILE_DESTRUCTIBLE_WALL, TILE_EMPTY, TILE_FLAG, TILE_WALL,
    },
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FindReturnConfig {
    pub num_agents: usize,
    pub num_flags: usize,

    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,

    pub mapgen_threshold: f32,
    pub digging_timeout: u32,
    pub treasure_reward: f32,
}

impl Default for FindReturnConfig {
    fn default() -> Self {
        Self {
            num_agents: 32,
            num_flags: 1,
            width: 40,
            height: 40,
            view_width: 11,
            view_height: 11,
            mapgen_threshold: 0.3,
            digging_timeout: 5,
            treasure_reward: 1.0,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct FindReturnAgent {
    position: Position,
    timeout: u32,
}

#[derive(Debug, Clone)]
struct FindReturnState {
    rngs: SmallRng,
    agents: Vec<FindReturnAgent>,
    agent_order: Vec<usize>, // agent turn order
    time: i32,

    base_map: Array2<VocabId>, // the bottom layer of the map without agents
    map: Array2<VocabId>,      // the base map plus the agents
    free_positions: Vec<Position>, // these are used to calculate spawn positions
}

#[derive(Debug, Clone)]
struct FindReturnSymbols {
    obs_tile_empty: VocabId,
    obs_tile_destructible_wall: VocabId,
    obs_tile_wall: VocabId,
    obs_tile_flag: VocabId,
    obs_tile_decor: [VocabId; 4],
    obs_agent_generic: VocabId,
    action_move_up: VocabId,
    action_move_right: VocabId,
    action_move_down: VocabId,
    action_move_left: VocabId,
}

#[derive(Debug, Clone)]
pub struct FindReturn {
    pub config: FindReturnConfig,
    state: FindReturnState,

    pad_width: i32,
    pad_height: i32,

    // full map size including wall padding on both sides
    width: i32,
    height: i32,

    obs_spec: ObservationSpec,
    action_spec: ActionSpec,

    obs_vocab: Vocabulary,
    action_vocab: Vocabulary,
    symbols: FindReturnSymbols,
}

impl FindReturnSymbols {
    fn direction(&self, action: VocabId) -> Position {
        if action == self.action_move_up {
            Position::new(0, 1)
        } else if action == self.action_move_right {
            Position::new(1, 0)
        } else if action == self.action_move_down {
            Position::new(0, -1)
        } else if action == self.action_move_left {
            Position::new(-1, 0)
        } else {
            Position::new(0, 0)
        }
    }

    fn blocked(&self, tile: VocabId) -> bool {
        tile == self.obs_tile_wall
            || tile == self.obs_tile_destructible_wall
            || tile == self.obs_agent_generic
    }
}

impl FindReturn {
    pub fn new(config: &FindReturnConfig) -> Self {
        let mut action_vocab = Vocabulary::new();
        let mut obs_vocab = Vocabulary::new();

        let symbols = FindReturnSymbols {
            obs_tile_empty: obs_vocab.add(TILE_EMPTY),
            obs_tile_destructible_wall: obs_vocab.add(TILE_DESTRUCTIBLE_WALL),
            obs_tile_wall: obs_vocab.add(TILE_WALL),
            obs_tile_flag: obs_vocab.add(TILE_FLAG),
            obs_tile_decor: TILE_DECOR.map(|symbol| obs_vocab.add(symbol)),
            obs_agent_generic: obs_vocab.add(AGENT_GENERIC),
            action_move_up: action_vocab.add(MOVE_UP),
            action_move_right: action_vocab.add(MOVE_RIGHT),
            action_move_down: action_vocab.add(MOVE_DOWN),
            action_move_left: action_vocab.add(MOVE_LEFT),
        };

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
                base_map: Array2::zeros((width as usize, height as usize)),
                free_positions: Vec::new(),
                map: Array2::zeros((width as usize, height as usize)),
                rngs: SmallRng::seed_from_u64(0),
                time: 0,
            },

            pad_width,
            pad_height,
            width,
            height,

            obs_spec,
            action_spec,

            action_vocab,
            obs_vocab,

            symbols,
        }
    }

    fn calculate_free_positions(&mut self) {
        self.state.free_positions.clear();
        for x in self.pad_width..self.width - self.pad_width {
            for y in self.pad_height..self.height - self.pad_height {
                let position = Position::new(x, y);
                let tile = self.state.map[position.idx()];

                if !self.symbols.blocked(tile) {
                    self.state.free_positions.push(position);
                }
            }
        }

        self.state.free_positions.shuffle(&mut self.state.rngs);
    }

    fn encode_observations(&self, timestep: &mut TimeStepMut) {
        let view_width = self.config.view_width;
        let view_height = self.config.view_height;

        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            // wall padding keeps the view window inside the map
            let x0 = agent.position.x - view_width / 2;
            let y0 = agent.position.y - view_height / 2;

            let window = self.state.map.slice(s![
                x0 as usize..(x0 + view_width) as usize,
                y0 as usize..(y0 + view_height) as usize,
            ]);
            timestep
                .obs
                .slice_mut(s![agent_id, .., .., 0])
                .assign(&window);
        }

        timestep.time.fill(self.state.time);
        timestep.terminated.fill(false);
        timestep.task_ids.fill(0);
        // all move actions are always valid; true marks a legal action, same
        // convention as the python side (mapox.timestep.TimeStep.action_mask)
        timestep.action_mask.fill(true);
    }
}

impl Environment for FindReturn {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.state.rngs = SmallRng::seed_from_u64(seed);

        self.state.time = 0;

        let dim = (self.width as usize, self.height as usize);
        if self.state.base_map.dim() != dim {
            self.state.map = Array2::zeros(dim);
            self.state.base_map = Array2::zeros(dim);
        }

        self.state.base_map.fill(self.symbols.obs_tile_wall);
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
                    self.symbols.obs_tile_destructible_wall
                } else {
                    self.symbols.obs_tile_empty
                }
            },
        );

        sprinkle_decor(
            interior,
            self.symbols.obs_tile_empty,
            &self.symbols.obs_tile_decor,
            &mut self.state.rngs,
        );

        // Base map finished
        self.state.map.assign(&self.state.base_map);

        self.state.agents.clear();
        self.calculate_free_positions();

        // Place the flag
        for _ in 0..self.config.num_flags {
            let flag_position = self.state.free_positions.pop().unwrap();
            self.state.base_map[flag_position.idx()] = self.symbols.obs_tile_flag;
            self.state.map[flag_position.idx()] = self.symbols.obs_tile_flag;
        }

        // Place the agents
        for _ in 0..self.num_agents() {
            let position = self.state.free_positions.pop().unwrap();
            self.state.agents.push(FindReturnAgent {
                position,
                ..Default::default()
            });
            self.state.map[position.idx()] = self.symbols.obs_agent_generic;
        }

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.encode_observations(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        let mut agent_respawn_ids: Vec<usize> = Vec::new();
        self.state.agent_order.shuffle(&mut self.state.rngs);

        for &agent_id in &self.state.agent_order {
            let (agent, map, base_map, symbols) = (
                &mut self.state.agents[agent_id],
                &mut self.state.map,
                &mut self.state.base_map,
                &self.symbols,
            );

            timestep.last_action[agent_id] = actions[agent_id];
            timestep.reward[agent_id] = 0.0;

            if agent.timeout > 0 {
                agent.timeout -= 1;
                continue;
            }

            let dir = symbols.direction(actions[agent_id]);
            let target = agent.position + dir;
            let target_tile = map[target.idx()];

            if !symbols.blocked(target_tile) {
                // unpaint the agent because it's moving
                map[agent.position.idx()] = base_map[agent.position.idx()];
                agent.position = target;
            } else if target_tile == symbols.obs_tile_destructible_wall {
                // dig action
                map[target.idx()] = symbols.obs_tile_empty;
                base_map[target.idx()] = symbols.obs_tile_empty;
                agent.timeout = self.config.digging_timeout;
                continue;
            }

            let found_flag = base_map[agent.position.idx()] == symbols.obs_tile_flag;
            if found_flag {
                agent_respawn_ids.push(agent_id);
                timestep.reward[agent_id] = self.config.treasure_reward;
            } else {
                // paint the agent back only if it's not found the flag
                map[agent.position.idx()] = symbols.obs_agent_generic;
            }
        }

        if !agent_respawn_ids.is_empty() {
            self.calculate_free_positions();

            let free_positions = &mut self.state.free_positions;
            for &agent_id in &agent_respawn_ids {
                let agent = &mut self.state.agents[agent_id];
                agent.position = free_positions.pop().unwrap();
                self.state.map[agent.position.idx()] = self.symbols.obs_agent_generic;
            }
        }

        self.state.time += 1;
        self.encode_observations(timestep);
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
            tile_width: self.width as usize,
            tile_height: self.height as usize,
            view_width: self.config.view_width as usize,
            view_height: self.config.view_height as usize,
        }
    }

    fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
        let tilemap = &mut grid_render_state.tilemap;
        if tilemap.dim() != self.state.map.dim() {
            *tilemap = Array2::zeros(self.state.map.dim());
        }
        tilemap.assign(&self.state.map);

        grid_render_state.agent_positions.clear();
        for agent in &self.state.agents {
            tilemap[agent.position.idx()] = self.symbols.obs_agent_generic;
            grid_render_state.agent_positions.push(agent.position);
        }
    }
}
