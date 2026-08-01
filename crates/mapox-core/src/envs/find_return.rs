use ndarray::{Array2, s};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::common::Position,
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_GENERIC, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, TILE_DESTRUCTIBLE_WALL,
        TILE_EMPTY, TILE_FLAG, TILE_WALL,
    },
    timestep::TimeStepMut,
    vocab::Vocabulary,
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

    pub mapgen_threshold: f64,
    pub digging_timeout: i32,
    pub treasure_reward: f64,
}

impl Default for FindReturnConfig {
    fn default() -> Self {
        Self {
            num_agents: 1,
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
pub struct FindReturnAgent {
    pub position: Position,
    pub found_reward: bool,
}

#[derive(Debug, Default, Clone)]
pub struct FindReturnState {
    pub agents: Vec<FindReturnAgent>,
    pub time: i32,
    pub map: Array2<i8>,
}

#[derive(Debug, Default, Clone)]
pub struct FindReturn {
    pub config: FindReturnConfig,
    pub state: FindReturnState,

    pad_width: i32,
    pad_height: i32,

    // full map size including wall padding on both sides
    width: i32,
    height: i32,

    obs_spec: ObservationSpec,
    action_spec: ActionSpec,

    obs_vocab: Vocabulary,
    action_vocab: Vocabulary,

    obs_tile_empty: usize,
    obs_tile_destructible_wall: usize,
    obs_tile_wall: usize,
    obs_agent_generic: usize,
    action_move_up: usize,
    action_move_right: usize,
    action_move_down: usize,
    action_move_left: usize,
}

impl FindReturn {
    pub fn new(config: &FindReturnConfig) -> Self {
        let mut action_vocab = Vocabulary::new();
        let mut obs_vocab = Vocabulary::new();

        let obs_tile_empty = obs_vocab.add(TILE_EMPTY);
        let obs_tile_destructible_wall = obs_vocab.add(TILE_DESTRUCTIBLE_WALL);
        let obs_tile_wall = obs_vocab.add(TILE_WALL);
        obs_vocab.add(TILE_FLAG);
        let obs_agent_generic = obs_vocab.add(AGENT_GENERIC);

        let action_move_up = action_vocab.add(MOVE_UP);
        let action_move_right = action_vocab.add(MOVE_RIGHT);
        let action_move_down = action_vocab.add(MOVE_DOWN);
        let action_move_left = action_vocab.add(MOVE_LEFT);

        let pad_width = config.view_width / 2;
        let pad_height = config.view_height / 2;

        let width = config.width + 2 * pad_width;
        let height = config.height + 2 * pad_height;

        let obs_spec = ObservationSpec::new(config.view_width, config.view_height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        Self {
            config: config.clone(),
            state: FindReturnState::default(),

            pad_width,
            pad_height,
            width,
            height,

            obs_spec,
            action_spec,

            action_vocab,
            obs_vocab,

            action_move_up,
            action_move_right,
            action_move_down,
            action_move_left,
            obs_tile_empty,
            obs_tile_destructible_wall,
            obs_tile_wall,
            obs_agent_generic,
        }
    }

    fn idx(&self, x: i32, y: i32) -> usize {
        (x * self.height + y) as usize
    }

    fn direction(&self, action: i32) -> (i32, i32) {
        let action = action as usize;
        if action == self.action_move_up {
            (0, 1)
        } else if action == self.action_move_right {
            (1, 0)
        } else if action == self.action_move_down {
            (0, -1)
        } else if action == self.action_move_left {
            (-1, 0)
        } else {
            (0, 0)
        }
    }

    fn blocked(&self, tile: i8) -> bool {
        let tile = tile as usize;
        tile == self.obs_tile_wall || tile == self.obs_tile_destructible_wall
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

            for other in &self.state.agents {
                let view_x = other.position.x - x0;
                let view_y = other.position.y - y0;
                if (0..view_width).contains(&view_x) && (0..view_height).contains(&view_y) {
                    timestep.obs[[agent_id, view_x as usize, view_y as usize, 0]] =
                        self.obs_agent_generic as i8;
                }
            }
        }

        timestep.time.fill(self.state.time);
        timestep.terminated.fill(0);
        timestep.task_ids.fill(0);
        // all move actions are always valid
        timestep.action_mask.fill(1);
    }
}

impl Environment for FindReturn {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        let mut rng = StdRng::seed_from_u64(seed);

        self.state.time = 0;

        let dim = (self.width as usize, self.height as usize);
        if self.state.map.dim() != dim {
            self.state.map = Array2::zeros(dim);
        }
        // wall border, empty interior
        self.state.map.fill(self.obs_tile_wall as i8);
        self.state
            .map
            .slice_mut(s![
                self.pad_width as usize..(self.width - self.pad_width) as usize,
                self.pad_height as usize..(self.height - self.pad_height) as usize,
            ])
            .fill(self.obs_tile_empty as i8);

        self.state.agents.clear();
        for _ in 0..self.config.num_agents {
            self.state.agents.push(FindReturnAgent {
                position: Position {
                    x: rng.random_range(self.pad_width..self.width - self.pad_width),
                    y: rng.random_range(self.pad_height..self.height - self.pad_height),
                },
                found_reward: false,
            });
        }

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.encode_observations(timestep);
    }

    fn step(&mut self, actions: &[i32], timestep: &mut TimeStepMut) {
        for agent_id in 0..self.state.agents.len() {
            let (dx, dy) = self.direction(actions[agent_id]);
            let agent = &self.state.agents[agent_id];
            let target_x = agent.position.x + dx;
            let target_y = agent.position.y + dy;

            if !self.blocked(self.state.map[[target_x as usize, target_y as usize]]) {
                self.state.agents[agent_id].position = Position {
                    x: target_x,
                    y: target_y,
                };
            }

            timestep.reward[agent_id] = 0.0;
            timestep.last_action[agent_id] = actions[agent_id];
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
        grid_render_state.tilemap.clear();
        grid_render_state
            .tilemap
            .extend(self.state.map.iter().map(|&tile| tile as u8));
        grid_render_state.agent_positions.clear();

        for agent in &self.state.agents {
            grid_render_state.tilemap[self.idx(agent.position.x, agent.position.y)] =
                self.obs_agent_generic as u8;
            grid_render_state.agent_positions.push(agent.position);
        }
    }
}
