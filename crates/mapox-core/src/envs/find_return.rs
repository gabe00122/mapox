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
        AGENT_GENERIC, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, TILE_DECOR_1, TILE_DECOR_2,
        TILE_DECOR_3, TILE_DECOR_4, TILE_DESTRUCTIBLE_WALL, TILE_EMPTY, TILE_FLAG, TILE_MASK,
        TILE_UI, TILE_WALL,
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
    pub digging_timeout: u32,
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
    time: usize,

    base_map: Array2<FindReturnObs>, // the bottom layer of the map without agents
    map: Array2<FindReturnObs>,      // the base map plus the agents
    free_positions: Vec<Position>,   // these are used to calculate spawn positions
}

vocab_enum!(FindReturnObs {
    UI => TILE_UI,
    Mask => TILE_MASK,
    TileEmpty => TILE_EMPTY,
    TileDestructibleWall => TILE_DESTRUCTIBLE_WALL,
    TileWall => TILE_WALL,
    TileFlag => TILE_FLAG,
    TileDecor1 => TILE_DECOR_1,
    TileDecor2 => TILE_DECOR_2,
    TileDecor3 => TILE_DECOR_3,
    TileDecor4 => TILE_DECOR_4,
    AgentGeneric => AGENT_GENERIC,
});

impl FindReturnObs {
    // const TABLE: &[FindReturnObs] = &[FindReturnObs::TileDecor1, FindReturnObs::TileDecor1];

    fn blocked(self) -> bool {
        use FindReturnObs::*;
        matches!(self, TileWall | TileDestructibleWall | AgentGeneric)
    }

    fn opaque(self) -> bool {
        use FindReturnObs::*;
        matches!(self, TileWall | TileDestructibleWall)
    }
}

vocab_enum!(FindReturnAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
});

impl FindReturnAction {
    fn direction(self) -> Position {
        use FindReturnAction::*;
        match self {
            MoveUp => Position::new(0, 1),
            MoveRight => Position::new(1, 0),
            MoveDown => Position::new(0, -1),
            MoveLeft => Position::new(-1, 0),
        }
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
            ui.fill(FindReturnObs::Mask as VocabId);
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length - 1);
        timestep.task_ids.fill(0);
        timestep.action_mask.fill(true);
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

        // Place the flag
        for _ in 0..self.config.num_flags {
            let flag_position = self.state.free_positions.pop().unwrap();
            self.state.base_map[flag_position.idx()] = FindReturnObs::TileFlag;
            self.state.map[flag_position.idx()] = FindReturnObs::TileFlag;
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
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        let mut agent_respawn_ids: Vec<usize> = Vec::new();
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

            let dir = FindReturnAction::from_id(actions[agent_id]).direction();
            let target = agent.position + dir;
            let target_tile = map[target.idx()];

            if !target_tile.blocked() {
                // unpaint the agent because it's moving
                map[agent.position.idx()] = base_map[agent.position.idx()];
                agent.position = target;
            } else if target_tile == FindReturnObs::TileDestructibleWall {
                // dig action
                map[target.idx()] = FindReturnObs::TileEmpty;
                base_map[target.idx()] = FindReturnObs::TileEmpty;
                agent.timeout = self.config.digging_timeout;
                continue;
            }

            let found_flag = base_map[agent.position.idx()] == FindReturnObs::TileFlag;
            if found_flag {
                agent_respawn_ids.push(agent_id);
                timestep.reward[agent_id] = self.config.treasure_reward;
            } else {
                // paint the agent back only if it's not found the flag
                map[agent.position.idx()] = FindReturnObs::AgentGeneric;
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
        tilemap.zip_mut_with(&self.state.map, |dst, &tile| *dst = tile.into());

        grid_render_state.agent_positions.clear();
        for agent in &self.state.agents {
            tilemap[agent.position.idx()] = FindReturnObs::AgentGeneric.into();
            grid_render_state.agent_positions.push(agent.position);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timestep::TimeStepBuffers;
    use FindReturnObs::*;

    /// An env whose interior is bare floor, for poking walls into by hand.
    fn empty_env() -> FindReturn {
        let mut env = FindReturn::new(
            &FindReturnConfig {
                num_agents: 1,
                width: 21,
                height: 21,
                ..Default::default()
            },
            512,
        );

        env.state.base_map.fill(FindReturnObs::TileWall);
        env.state
            .base_map
            .slice_mut(s![
                env.pad_width as usize..(env.width - env.pad_width) as usize,
                env.pad_height as usize..(env.height - env.pad_height) as usize,
            ])
            .fill(FindReturnObs::TileEmpty);
        env.state.map.assign(&env.state.base_map);

        env
    }

    fn center(env: &FindReturn) -> Position {
        Position::new(env.width / 2, env.height / 2)
    }

    /// Drops the one agent at the centre of the map, then encodes the
    /// observation it would be handed. The geometry of the sweep itself is
    /// [`fov`]'s to test; what matters here is that the env feeds it the right
    /// map, mask and opacity rule.
    fn observe(env: &mut FindReturn) -> Array2<VocabId> {
        let position = center(env);
        env.state.agents.push(FindReturnAgent {
            position,
            ..Default::default()
        });
        env.state.map[position.idx()] = FindReturnObs::AgentGeneric;

        let mut buffers = TimeStepBuffers::new(env);
        env.encode_observations(&mut buffers.view_mut());
        buffers
            .obs
            .slice(s![0, .., .., 0])
            .into_owned()
            .into_dimensionality()
            .expect("the observation window is 2d")
    }

    /// View-window coordinates of a map offset from the agent.
    fn cell(env: &FindReturn, dx: i32, dy: i32) -> [usize; 2] {
        [
            (env.config.view_width / 2 + dx) as usize,
            (env.config.view_height / 2 + dy) as usize,
        ]
    }

    /// The vocab id a tile turns into once it has been encoded into a view.
    fn id(tile: FindReturnObs) -> VocabId {
        tile.into()
    }

    #[test]
    fn tiles_behind_a_wall_arrive_masked() {
        let mut env = empty_env();
        let wall = center(&env) + Position::new(0, 1);
        env.state.map[wall.idx()] = TileWall;

        let view = observe(&mut env);

        assert_eq!(view[cell(&env, 0, 0)], id(AgentGeneric));
        assert_eq!(view[cell(&env, 0, 1)], id(TileWall));
        assert_eq!(view[cell(&env, 0, 2)], id(Mask));
        // ... while an open room reaches the agent whole
        assert_eq!(view[cell(&env, 0, -2)], id(TileEmpty));
    }

    /// Diggable walls block sight the same as solid ones, so a corridor the
    /// agent dug out is the only thing it can see down.
    #[test]
    fn destructible_walls_are_opaque() {
        let mut env = empty_env();
        let wall = center(&env) + Position::new(2, 0);
        env.state.map[wall.idx()] = TileDestructibleWall;

        let view = observe(&mut env);

        assert_eq!(view[cell(&env, 2, 0)], id(TileDestructibleWall));
        assert_eq!(view[cell(&env, 3, 0)], id(Mask));
    }

    /// Agents stop each other moving but not seeing: standing in a queue, every
    /// agent still watches the same corridor.
    #[test]
    fn agents_do_not_block_sight() {
        let mut env = empty_env();
        let other = center(&env) + Position::new(2, 0);
        env.state.map[other.idx()] = AgentGeneric;

        let view = observe(&mut env);

        assert_eq!(view[cell(&env, 2, 0)], id(AgentGeneric));
        assert_eq!(view[cell(&env, 3, 0)], id(TileEmpty));
    }
}
