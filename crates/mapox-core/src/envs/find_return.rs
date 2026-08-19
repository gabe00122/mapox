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
        AGENT_GENERIC, DIG_ACTION, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, PLACE_PIPE,
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
    Dig => DIG_ACTION
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

    /// Flips every flag from locked to claimable. Idempotent, and safe to
    /// call with an agent standing on a flag: that agent owns the top layer
    /// until it steps off, when the base map paints the new tile back in.
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

            let mut target = agent.position + agent.dir;
            match action {
                FindReturnAction::MoveUp
                | FindReturnAction::MoveRight
                | FindReturnAction::MoveDown
                | FindReturnAction::MoveLeft => {
                    // move along a pip
                    while (agent.dir.y == 0 && map[target.idx()] == FindReturnObs::PipeHorizontal)
                        | (agent.dir.x == 0 && map[target.idx()] == FindReturnObs::PipeVirtical)
                    {
                        target = target + agent.dir;
                    }

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
        empty_env_with(FindReturnConfig {
            num_agents: 1,
            width: 21,
            height: 21,
            ..Default::default()
        })
    }

    /// The same bare floor, but sized so that [`FindReturn::step`] can encode
    /// an observation into it: that path splits the window at row 15 for the UI
    /// band, so a view shorter than that slices past the end of the buffer.
    fn walkable_env() -> FindReturn {
        empty_env_with(FindReturnConfig {
            num_agents: 1,
            width: 21,
            height: 21,
            view_height: 21,
            ..Default::default()
        })
    }

    /// A walkable env whose flag stays locked for `preparation_steps` steps.
    fn flag_env(preparation_steps: usize) -> FindReturn {
        empty_env_with(FindReturnConfig {
            num_agents: 1,
            width: 21,
            height: 21,
            view_height: 21,
            preparation_steps,
            ..Default::default()
        })
    }

    fn place_flag(env: &mut FindReturn, position: Position) {
        env.state.base_map[position.idx()] = TileFlag;
        env.state.map[position.idx()] = TileFlag;
        env.state.flag_positions.push(position);
    }

    fn empty_env_with(config: FindReturnConfig) -> FindReturn {
        let mut env = FindReturn::new(&config, 512);

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

    fn spawn_agent(env: &mut FindReturn, position: Position) {
        env.state.agents.push(FindReturnAgent {
            position,
            ..Default::default()
        });
        env.state.map[position.idx()] = FindReturnObs::AgentGeneric;
    }

    /// Drops the one agent at the centre of the map, then encodes the
    /// observation it would be handed. The geometry of the sweep itself is
    /// [`fov`]'s to test; what matters here is that the env feeds it the right
    /// map, mask and opacity rule.
    fn observe(env: &mut FindReturn) -> Array2<VocabId> {
        let position = center(env);
        spawn_agent(env, position);

        let mut buffers = TimeStepBuffers::new(env);
        env.encode_observations(&mut buffers.view_mut());
        buffers
            .obs
            .slice(s![0, .., .., 0])
            .into_owned()
            .into_dimensionality()
            .expect("the observation window is 2d")
    }

    /// The map an agent standing at `position` can see, encoded the way
    /// [`FindReturn::encode_observations`] does it but without the UI band
    /// packed alongside.
    fn look(env: &FindReturn, position: Position) -> Array2<VocabId> {
        let mut view = Array2::zeros((
            env.config.view_width as usize,
            env.config.view_height as usize,
        ));
        fov::encode_visible(
            &env.state.map,
            position,
            &mut view.view_mut(),
            FindReturnObs::Mask,
            |tile| tile.opaque(),
        );
        view
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

    /// Water stops feet but not eyes, so an agent on the shore watches the far
    /// bank the way it would an open room.
    #[test]
    fn water_does_not_block_sight() {
        let mut env = empty_env();
        let position = center(&env);
        let pond = position + Position::new(2, 0);
        env.state.map[pond.idx()] = TileWater;

        let view = look(&env, position);

        assert_eq!(view[cell(&env, 2, 0)], id(TileWater));
        assert_eq!(view[cell(&env, 3, 0)], id(TileEmpty));
    }

    /// An agent that walks into water stays put, the way it would against a
    /// wall, and the pond is still there afterwards.
    #[test]
    fn water_blocks_movement() {
        let mut env = walkable_env();
        let start = center(&env);
        let pond = start + Position::new(1, 0);
        env.state.base_map[pond.idx()] = TileWater;
        env.state.map[pond.idx()] = TileWater;
        spawn_agent(&mut env, start);

        let mut buffers = TimeStepBuffers::new(&env);
        env.step(
            &[FindReturnAction::MoveRight.into()],
            &mut buffers.view_mut(),
        );

        assert_eq!(env.state.agents[0].position.idx(), start.idx());
        assert_eq!(env.state.map[pond.idx()], TileWater);
    }

    /// Unlike a destructible wall, water cannot be dug away: a lake is a
    /// permanent divide, and the agent does not even lose a turn to the attempt.
    #[test]
    fn water_cannot_be_dug_out() {
        let mut env = walkable_env();
        let start = center(&env);
        let pond = start + Position::new(1, 0);
        env.state.base_map[pond.idx()] = TileWater;
        env.state.map[pond.idx()] = TileWater;
        spawn_agent(&mut env, start);

        let mut buffers = TimeStepBuffers::new(&env);
        let mut timestep = buffers.view_mut();
        // face the pond, then swing at it
        env.step(&[FindReturnAction::MoveRight.into()], &mut timestep);
        env.step(&[FindReturnAction::Dig.into()], &mut timestep);

        assert_eq!(env.state.map[pond.idx()], TileWater);
        assert_eq!(env.state.agents[0].timeout, 0);
    }

    /// During the preparation phase the flag is scenery: an agent can walk over
    /// it, but it scores nothing and stays where it is.
    #[test]
    fn a_locked_flag_pays_nothing() {
        let mut env = flag_env(4);
        let start = center(&env);
        let flag = start + Position::new(1, 0);
        place_flag(&mut env, flag);
        spawn_agent(&mut env, start);

        let mut buffers = TimeStepBuffers::new(&env);
        env.step(
            &[FindReturnAction::MoveRight.into()],
            &mut buffers.view_mut(),
        );

        assert_eq!(buffers.reward[0], 0.0);
        assert_eq!(env.state.agents[0].position.idx(), flag.idx());
        assert_eq!(env.state.base_map[flag.idx()], TileFlag);
    }

    /// Once the preparation phase is over the flag swaps to its unlocked tile
    /// and the next agent to reach it is paid and sent back out onto the map.
    #[test]
    fn the_flag_unlocks_after_the_preparation_phase() {
        let mut env = flag_env(1);
        let start = center(&env);
        let flag = start + Position::new(2, 0);
        place_flag(&mut env, flag);
        spawn_agent(&mut env, start);

        let mut buffers = TimeStepBuffers::new(&env);
        let mut timestep = buffers.view_mut();
        // one step of preparation, then the step that walks onto the flag
        env.step(&[FindReturnAction::MoveRight.into()], &mut timestep);
        assert_eq!(env.state.base_map[flag.idx()], TileFlag);
        env.step(&[FindReturnAction::MoveRight.into()], &mut timestep);

        assert_eq!(env.state.base_map[flag.idx()], TileFlagUnlocked);
        assert_eq!(buffers.reward[0], env.config.treasure_reward);
        // ... and the agent is respawned somewhere else rather than parked on it
        assert_ne!(env.state.agents[0].position.idx(), flag.idx());
    }

    /// A zero-length preparation phase means the flag is live from reset, so an
    /// agent that stumbles onto it on the first step is paid.
    #[test]
    fn a_zero_step_preparation_phase_unlocks_at_reset() {
        let mut env = flag_env(0);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(0, &mut buffers.view_mut());

        for &position in &env.state.flag_positions {
            assert_eq!(env.state.base_map[position.idx()], TileFlagUnlocked);
        }
    }
}
