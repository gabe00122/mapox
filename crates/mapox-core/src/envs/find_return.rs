use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{
    env::Environment,
    envs::common::Position,
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_GENERIC, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, TILE_DESTRUCTIBLE_WALL,
        TILE_EMPTY, TILE_FLAG, TILE_WALL,
    },
    timestep::{OBS_CHANNELS, TimeStepMut},
    vocab::Vocabulary,
};

#[derive(Debug, Clone)]
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

    fn blocked(&self, tile: u8) -> bool {
        let tile = tile as usize;
        tile == self.obs_tile_wall || tile == self.obs_tile_destructible_wall
    }

    fn obs_base(&self, agent_id: usize, view_x: i32, view_y: i32) -> usize {
        ((agent_id * self.config.view_width as usize + view_x as usize)
            * self.config.view_height as usize
            + view_y as usize)
            * OBS_CHANNELS
    }

    fn encode_observations(&self, state: &FindReturnState, timestep: &mut TimeStepMut) {
        let view_width = self.config.view_width as usize;
        let view_height = self.config.view_height as usize;

        for (agent_id, agent) in state.agents.iter().enumerate() {
            let x0 = agent.position.x as usize - view_width / 2;
            let y0 = agent.position.y as usize - view_height / 2;

            for view_x in 0..view_width {
                for view_y in 0..view_height {
                    let map_x = x0 + view_x;
                    let map_y = y0 + view_y;

                    let tile = state.map[[map_x, map_y]];
                    timestep.obs[[agent_id, x0, y0, 0]] = tile as i8;
                }
            }

            for other in &state.agents {
                let view_x = other.position.x - x0;
                let view_y = other.position.y - y0;
                if (0..view_width).contains(&view_x) && (0..view_height).contains(&view_y) {
                    timestep.obs[self.obs_base(agent_id, view_x, view_y)] =
                        self.obs_agent_generic as i8;
                }
            }

            timestep.time[agent_id] = state.time;
            timestep.terminated[agent_id] = 0;
            timestep.task_ids[agent_id] = 0;
        }

        // all move actions are always valid
        timestep.action_mask.fill(1);
    }
}

impl Environment for FindReturn {
    type EnvState = FindReturnState;

    fn init_state(&self) -> FindReturnState {
        FindReturnState::default()
    }

    fn reset(&self, state: &mut Self::EnvState, seed: u64, timestep: &mut TimeStepMut) {
        let mut rng = StdRng::seed_from_u64(seed);

        state.time = 0;
        state.map.clear();
        state.map.resize(
            (self.width * self.height) as usize,
            self.obs_tile_empty as u8,
        );

        for x in 0..self.width {
            for y in 0..self.height {
                let interior = x >= self.pad_width
                    && x < self.width - self.pad_width
                    && y >= self.pad_height
                    && y < self.height - self.pad_height;
                if !interior {
                    state.map[self.idx(x, y)] = self.obs_tile_wall as u8;
                }
            }
        }

        state.agents.clear();
        for _ in 0..self.config.num_agents {
            state.agents.push(FindReturnAgent {
                position: Position {
                    x: rng.random_range(self.pad_width..self.width - self.pad_width),
                    y: rng.random_range(self.pad_height..self.height - self.pad_height),
                },
                found_reward: false,
            });
        }

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.encode_observations(state, timestep);
    }

    fn step(&self, state: &mut Self::EnvState, actions: &[i32], timestep: &mut TimeStepMut) {
        for agent_id in 0..state.agents.len() {
            let (dx, dy) = self.direction(actions[agent_id]);
            let agent = &state.agents[agent_id];
            let target_x = agent.position.x + dx;
            let target_y = agent.position.y + dy;

            if !self.blocked(state.map[self.idx(target_x, target_y)]) {
                state.agents[agent_id].position = Position {
                    x: target_x,
                    y: target_y,
                };
            }

            timestep.reward[agent_id] = 0.0;
            timestep.last_action[agent_id] = actions[agent_id];
        }

        state.time += 1;
        self.encode_observations(state, timestep);
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

    fn render_state_into(&self, state: &Self::EnvState, grid_render_state: &mut GridRenderState) {
        grid_render_state.tilemap.clear();
        grid_render_state.tilemap.extend_from_slice(&state.map);
        grid_render_state.agent_positions.clear();

        for agent in &state.agents {
            grid_render_state.tilemap[self.idx(agent.position.x, agent.position.y)] =
                self.obs_agent_generic as u8;
            grid_render_state.agent_positions.push(agent.position);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_timestep_buffers(
        env: &FindReturn,
    ) -> (
        Vec<i8>,
        Vec<i32>,
        Vec<u8>,
        Vec<i32>,
        Vec<f32>,
        Vec<u8>,
        Vec<i32>,
    ) {
        let n = env.num_agents();
        let obs_spec = env.observation_spec();
        let obs_len = n * obs_spec.width as usize * obs_spec.height as usize * OBS_CHANNELS;
        (
            vec![0; obs_len],
            vec![0; n],
            vec![0; n],
            vec![0; n],
            vec![0.0; n],
            vec![0; n * env.action_spec().num_actions],
            vec![0; n],
        )
    }

    #[test]
    fn agent_moves_and_walls_block() {
        let env = FindReturn::new(&FindReturnConfig::default());
        let mut state = env.init_state();

        let (mut obs, mut time, mut terminated, mut last_action, mut reward, mut mask, mut tasks) =
            make_timestep_buffers(&env);
        let mut timestep = TimeStepMut {
            obs: &mut obs,
            time: &mut time,
            terminated: &mut terminated,
            last_action: &mut last_action,
            reward: &mut reward,
            action_mask: &mut mask,
            task_ids: &mut tasks,
        };

        env.reset(&mut state, 0, &mut timestep);
        let start = state.agents[0].position;

        // moving up shifts y by +1 on an empty map
        env.step(&mut state, &[env.action_move_up as i32], &mut timestep);
        assert_eq!(state.agents[0].position.y, start.y + 1);
        assert_eq!(state.agents[0].position.x, start.x);
        assert_eq!(state.time, 1);

        // walking left into the wall border eventually stops the agent
        for _ in 0..env.width {
            env.step(&mut state, &[env.action_move_left as i32], &mut timestep);
        }
        assert_eq!(state.agents[0].position.x, env.pad_width);

        // the center of the agent's view is itself
        let view_width = env.config.view_width as usize;
        let view_height = env.config.view_height as usize;
        let center = ((view_width / 2) * view_height + view_height / 2) * OBS_CHANNELS;
        assert_eq!(timestep.obs[center], env.obs_agent_generic as i8);
    }
}
