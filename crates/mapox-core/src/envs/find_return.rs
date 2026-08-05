use ndarray::{Array2, s};
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::common::Position,
    map_gen::{choose_positions, fractal_noise, sprinkle_decor},
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

    pub mapgen_threshold: f64,
    pub digging_timeout: i32,
    pub treasure_reward: f64,
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
pub struct FindReturnAgent {
    pub position: Position,
    pub found_reward: bool,
}

#[derive(Debug, Default, Clone)]
pub struct FindReturnState {
    pub agents: Vec<FindReturnAgent>,
    pub time: i32,
    pub map: Array2<VocabId>,
}

#[derive(Debug, Default, Clone)]
pub struct FindReturn {
    pub config: FindReturnConfig,
    pub state: FindReturnState,

    /// Map with agents stamped on top; observation windows slice this so
    /// encoding stays O(agents × view area) instead of O(agents²).
    stamped_map: Array2<VocabId>,

    pad_width: i32,
    pad_height: i32,

    // full map size including wall padding on both sides
    width: i32,
    height: i32,

    obs_spec: ObservationSpec,
    action_spec: ActionSpec,

    obs_vocab: Vocabulary,
    action_vocab: Vocabulary,

    obs_tile_empty: VocabId,
    obs_tile_destructible_wall: VocabId,
    obs_tile_wall: VocabId,
    obs_tile_decor: [VocabId; 4],
    obs_agent_generic: VocabId,
    action_move_up: VocabId,
    action_move_right: VocabId,
    action_move_down: VocabId,
    action_move_left: VocabId,
}

impl FindReturn {
    pub fn new(config: &FindReturnConfig) -> Self {
        let mut action_vocab = Vocabulary::new();
        let mut obs_vocab = Vocabulary::new();

        let obs_tile_empty = obs_vocab.add(TILE_EMPTY);
        let obs_tile_destructible_wall = obs_vocab.add(TILE_DESTRUCTIBLE_WALL);
        let obs_tile_wall = obs_vocab.add(TILE_WALL);
        obs_vocab.add(TILE_FLAG);
        let obs_tile_decor = TILE_DECOR.map(|symbol| obs_vocab.add(symbol));
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

            stamped_map: Array2::default((0, 0)),

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
            obs_tile_decor,
            obs_agent_generic,
        }
    }

    fn direction(&self, action: i32) -> (i32, i32) {
        if action == i32::from(self.action_move_up) {
            (0, 1)
        } else if action == i32::from(self.action_move_right) {
            (1, 0)
        } else if action == i32::from(self.action_move_down) {
            (0, -1)
        } else if action == i32::from(self.action_move_left) {
            (-1, 0)
        } else {
            (0, 0)
        }
    }

    fn blocked(&self, tile: VocabId) -> bool {
        tile == self.obs_tile_wall || tile == self.obs_tile_destructible_wall
    }

    fn encode_observations(&mut self, timestep: &mut TimeStepMut) {
        let view_width = self.config.view_width;
        let view_height = self.config.view_height;

        if self.stamped_map.dim() != self.state.map.dim() {
            self.stamped_map = Array2::zeros(self.state.map.dim());
        }
        self.stamped_map.assign(&self.state.map);
        for agent in &self.state.agents {
            self.stamped_map[[agent.position.x as usize, agent.position.y as usize]] =
                self.obs_agent_generic;
        }

        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            // wall padding keeps the view window inside the map
            let x0 = agent.position.x - view_width / 2;
            let y0 = agent.position.y - view_height / 2;

            let window = self.stamped_map.slice(s![
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
        let mut rng = StdRng::seed_from_u64(seed);

        self.state.time = 0;

        let dim = (self.width as usize, self.height as usize);
        if self.state.map.dim() != dim {
            self.state.map = Array2::zeros(dim);
        }
        // wall border; the interior is carved out of fractal noise like the
        // jax `_generate_map` — destructible wall above the threshold, decor
        // sprinkled over what stays empty
        self.state.map.fill(self.obs_tile_wall);
        let mut interior = self.state.map.slice_mut(s![
            self.pad_width as usize..(self.width - self.pad_width) as usize,
            self.pad_height as usize..(self.height - self.pad_height) as usize,
        ]);
        let noise = fractal_noise(
            self.config.width as usize,
            self.config.height as usize,
            &mut rng,
        );
        for (tile, &value) in interior.iter_mut().zip(&noise) {
            *tile = if value as f64 > self.config.mapgen_threshold {
                self.obs_tile_destructible_wall
            } else {
                self.obs_tile_empty
            };
        }
        sprinkle_decor(
            interior,
            self.obs_tile_empty,
            &self.obs_tile_decor,
            &mut rng,
        );

        self.state.agents.clear();
        for position in choose_positions(
            &self.state.map,
            self.obs_tile_empty,
            self.config.num_agents,
            &mut rng,
        ) {
            self.state.agents.push(FindReturnAgent {
                position,
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
        let tilemap = &mut grid_render_state.tilemap;
        if tilemap.dim() != self.state.map.dim() {
            *tilemap = Array2::zeros(self.state.map.dim());
        }
        tilemap.assign(&self.state.map);

        grid_render_state.agent_positions.clear();
        for agent in &self.state.agents {
            tilemap[[agent.position.x as usize, agent.position.y as usize]] =
                self.obs_agent_generic;
            grid_render_state.agent_positions.push(agent.position);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array4};

    /// The render tilemap and the env's own map are both `[x, y]`-indexed
    /// `u8`; a transposed `assign` or a stale buffer would show up here rather
    /// than as scrambled art in the demo.
    #[test]
    fn render_state_matches_the_map_with_agents_stamped() {
        let config = FindReturnConfig {
            num_agents: 4,
            ..Default::default()
        };
        let mut env = FindReturn::new(&config);

        let obs_spec = env.observation_spec();
        let mut obs = Array4::zeros((
            config.num_agents,
            obs_spec.width as usize,
            obs_spec.height as usize,
            crate::timestep::OBS_CHANNELS,
        ));
        let mut time = Array1::zeros(config.num_agents);
        let mut terminated = Array1::default(config.num_agents);
        let mut last_action = Array1::zeros(config.num_agents);
        let mut reward = Array1::zeros(config.num_agents);
        let mut action_mask = Array2::default((config.num_agents, env.action_spec().num_actions));
        let mut task_ids = Array1::zeros(config.num_agents);
        env.reset(
            0,
            &mut TimeStepMut {
                obs: obs.view_mut(),
                time: time.view_mut(),
                terminated: terminated.view_mut(),
                last_action: last_action.view_mut(),
                reward: reward.view_mut(),
                action_mask: action_mask.view_mut(),
                task_ids: task_ids.view_mut(),
            },
        );

        // true marks a legal action (python convention); FindReturn's moves
        // are always legal, so an all-false mask here means the flag flipped
        assert!(action_mask.iter().all(|&legal| legal));

        env.step(
            &vec![0; config.num_agents],
            &mut TimeStepMut {
                obs: obs.view_mut(),
                time: time.view_mut(),
                terminated: terminated.view_mut(),
                last_action: last_action.view_mut(),
                reward: reward.view_mut(),
                action_mask: action_mask.view_mut(),
                task_ids: task_ids.view_mut(),
            },
        );
        assert!(action_mask.iter().all(|&legal| legal));

        let mut render_state = GridRenderState::default();
        env.render_state_into(&mut render_state);

        assert_eq!(render_state.tilemap.dim(), env.state.map.dim());
        assert_eq!(render_state.agent_positions.len(), config.num_agents);

        // the padded border is wall, and every agent cell carries the agent id
        assert_eq!(render_state.tilemap[[0, 0]], env.obs_tile_wall);
        for position in &render_state.agent_positions {
            assert_eq!(
                render_state.tilemap[[position.x as usize, position.y as usize]],
                env.obs_agent_generic
            );
        }

        // cells with no agent on them still mirror the map
        let occupied: Vec<_> = render_state
            .agent_positions
            .iter()
            .map(|p| (p.x as usize, p.y as usize))
            .collect();
        for ((x, y), &tile) in render_state.tilemap.indexed_iter() {
            if !occupied.contains(&(x, y)) {
                assert_eq!(tile, env.state.map[[x, y]], "mismatch at ({x}, {y})");
            }
        }
    }
}
