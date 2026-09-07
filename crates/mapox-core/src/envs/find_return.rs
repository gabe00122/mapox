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
    fn move_blocked(self) -> bool {
        use FindReturnObs::*;
        matches!(
            self,
            TileWall | TileDestructibleWall | TileWater | AgentGeneric
        )
    }

    fn spawnable(self) -> bool {
        use FindReturnObs::*;
        matches!(
            self,
            TileEmpty | TileDecor1 | TileDecor2 | TileDecor3 | TileDecor4
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

        let ui_height = 2;
        let view_height = config.view_height + ui_height;

        let width = config.width + 2 * pad_width;
        let height = config.height + 2 * pad_height;

        let obs_spec = ObservationSpec::new(config.view_width, view_height, obs_vocab.len());
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

                if tile.spawnable() {
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
        let fov_height = self.config.view_height as usize;

        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            // wall padding keeps the view window inside the map
            let mut view = timestep.obs.slice_mut(s![agent_id, .., ..fov_height, 0]);
            fov::encode_visible(
                &self.state.map,
                agent.position,
                &mut view,
                FindReturnObs::Mask,
                |tile| tile.opaque(),
            );

            let mut ui = timestep.obs.slice_mut(s![agent_id, .., fov_height.., 0]);
            ui.fill(FindReturnObs::UI as VocabId);
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length);
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
                        !self.state.map[target.idx()].move_blocked()
                    }
                    FindReturnAction::Dig => {
                        let target = agent.position + agent.dir;
                        self.state.map[target.idx()].destructible()
                    }
                    FindReturnAction::PlacePipe => {
                        let target = agent.position + agent.dir;
                        self.state.map[target.idx()].spawnable()
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
                    if !map[target.idx()].move_blocked() {
                        // unpaint the agent because it's moving
                        map[agent.position.idx()] = base_map[agent.position.idx()];
                        agent.position = target;

                        // a locked flag is scenery: only the unlocked tile pays
                        let found_flag =
                            base_map[agent.position.idx()] == FindReturnObs::TileFlagUnlocked;
                        if found_flag {
                            agent_respawn_ids.push(agent_id);
                            timestep.reward[agent_id] = self.config.treasure_reward;
                        } else {
                            map[agent.position.idx()] = FindReturnObs::AgentGeneric;
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

                    if target_tile.spawnable() {
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
        for agent in &self.state.agents {
            let local_pos = agent.position - Position::new(self.pad_width, self.pad_height);
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
    use FindReturnAction::*;
    use FindReturnObs::*;
    use rand::RngExt;
    use std::collections::HashSet;

    /// One agent on bare floor, no flags, for poking walls, pipes, and
    /// flags into by hand.
    fn empty_env() -> FindReturn {
        empty_env_with(FindReturnConfig {
            num_agents: 1,
            num_flags: 0,
            width: 21,
            height: 21,
            ..Default::default()
        })
    }

    fn empty_env_with(config: FindReturnConfig) -> FindReturn {
        let mut env = FindReturn::new(&config, 512);

        env.state.base_map.fill(TileWall);
        env.state
            .base_map
            .slice_mut(s![
                env.pad_width as usize..(env.width - env.pad_width) as usize,
                env.pad_height as usize..(env.height - env.pad_height) as usize,
            ])
            .fill(TileEmpty);
        env.state.map.assign(&env.state.base_map);

        env
    }

    fn center(env: &FindReturn) -> Position {
        Position::new(env.width / 2, env.height / 2)
    }

    /// Puts every agent on the map, `positions` in agent-id order.
    fn spawn_agents(env: &mut FindReturn, positions: &[Position]) {
        for &position in positions {
            env.state.agents.push(FindReturnAgent {
                position,
                ..Default::default()
            });
            env.state.map[position.idx()] = AgentGeneric;
        }
    }

    fn place_tile(env: &mut FindReturn, position: Position, tile: FindReturnObs) {
        env.state.base_map[position.idx()] = tile;
        env.state.map[position.idx()] = tile;
    }

    /// Places a flag the way `reset` does: on both layers, and registered
    /// in `flag_positions`, which is the list the unlock step walks.
    fn place_flag(env: &mut FindReturn, position: Position, tile: FindReturnObs) {
        place_tile(env, position, tile);
        env.state.flag_positions.push(position);
    }

    /// Steps every agent with the action named for it, and hands back the
    /// buffers so the caller can read rewards out.
    fn step(env: &mut FindReturn, buffers: &mut TimeStepBuffers, actions: &[FindReturnAction]) {
        let actions: Vec<VocabId> = actions.iter().map(|&a| a.into()).collect();
        env.step(&actions, &mut buffers.view_mut());
    }

    /// The view-window cell a map offset from the agent lands in. The agent
    /// sits at the centre of the fov, which is the window minus its UI band.
    fn cell(env: &FindReturn, dx: i32, dy: i32) -> [usize; 2] {
        [
            (env.config.view_width / 2 + dx) as usize,
            (env.config.view_height / 2 + dy) as usize,
        ]
    }

    fn id(tile: FindReturnObs) -> VocabId {
        tile.into()
    }

    /// The frozen mask: nothing but the noop is legal.
    fn masked_to_noop(buffers: &TimeStepBuffers, agent: usize) {
        let mask = buffers.action_mask.row(agent);
        assert_eq!(mask.iter().filter(|&&legal| legal).count(), 1);
        assert!(mask[Noop as usize]);
    }

    /// The flag stock on the bottom layer, split by lock state.
    fn flag_tally(env: &FindReturn) -> (usize, usize) {
        let mut locked = 0;
        let mut unlocked = 0;
        for &tile in env.state.base_map.iter() {
            match tile {
                TileFlag => locked += 1,
                TileFlagUnlocked => unlocked += 1,
                _ => {}
            }
        }
        (locked, unlocked)
    }

    /// Reset has to leave the map consistent: every agent painted on the top
    /// layer, the full stock of flags on the bottom one, and the wall
    /// padding that keeps every window and slide in bounds still in place.
    #[test]
    fn reset_places_every_agent_and_flag() {
        let config = FindReturnConfig {
            num_agents: 3,
            num_flags: 2,
            width: 24,
            height: 24,
            preparation_steps: 8,
            ..Default::default()
        };
        let mut env = FindReturn::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(7, &mut buffers.view_mut());

        assert_eq!(env.state.agents.len(), 3);
        let positions: HashSet<_> = env.state.agents.iter().map(|a| a.position.idx()).collect();
        assert_eq!(positions.len(), 3, "agents share a tile");
        for agent in &env.state.agents {
            assert_eq!(env.state.map[agent.position.idx()], AgentGeneric);
            let p = agent.position;
            assert!(p.x >= env.pad_width && p.x < env.width - env.pad_width);
            assert!(p.y >= env.pad_height && p.y < env.height - env.pad_height);
        }

        assert_eq!(flag_tally(&env), (2, 0));
        assert_eq!(env.state.base_map[[0, 0]], TileWall);
        assert_eq!(
            env.state.base_map[[env.width as usize - 1, env.height as usize - 1]],
            TileWall
        );

        // ... and with no preparation phase the flags start open instead.
        let open = FindReturnConfig {
            preparation_steps: 0,
            ..config
        };
        let mut env = FindReturn::new(&open, 512);
        env.reset(7, &mut buffers.view_mut());
        assert_eq!(flag_tally(&env), (0, 2));
    }

    /// Flags sit locked while the preparation phase runs, and open on the
    /// step after it has: `preparation_steps` full steps, then the next step
    /// starts with the stock open.
    #[test]
    fn flags_unlock_after_the_preparation_steps() {
        let mut env = empty_env_with(FindReturnConfig {
            num_agents: 1,
            num_flags: 0,
            width: 21,
            height: 21,
            preparation_steps: 2,
            ..Default::default()
        });
        let start = center(&env);
        let flag = start + Position::new(1, 0);
        place_flag(&mut env, flag, TileFlag);
        spawn_agents(&mut env, &[start]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.base_map[flag.idx()], TileFlag);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.base_map[flag.idx()], TileFlag);
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.base_map[flag.idx()], TileFlagUnlocked);
    }

    /// A locked flag is scenery: the agent may stand on it, but it pays
    /// nothing and stays locked until the preparation phase is over.
    #[test]
    fn walking_over_a_locked_flag_pays_nothing() {
        let mut env = empty_env_with(FindReturnConfig {
            num_agents: 1,
            num_flags: 0,
            width: 21,
            height: 21,
            preparation_steps: 1,
            ..Default::default()
        });
        let start = center(&env);
        let flag = start + Position::new(1, 0);
        place_flag(&mut env, flag, TileFlag);
        spawn_agents(&mut env, &[start]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[MoveRight]);

        assert_eq!(buffers.reward[0], 0.0);
        assert_eq!(env.state.base_map[flag.idx()], TileFlag);
        assert_eq!(env.state.agents[0].position.idx(), flag.idx());
        assert_eq!(env.state.map[flag.idx()], AgentGeneric);
    }

    /// The unlocked flag is the only tile that pays: walking on it rewards
    /// the agent and sends it back to free ground, and the flag stands
    /// there open for the next visit.
    #[test]
    fn an_unlocked_flag_pays_and_respawns_the_agent() {
        let mut env = empty_env_with(FindReturnConfig {
            num_agents: 1,
            num_flags: 0,
            width: 21,
            height: 21,
            preparation_steps: 0,
            ..Default::default()
        });
        let start = center(&env);
        let flag = start + Position::new(1, 0);
        place_flag(&mut env, flag, TileFlagUnlocked);
        spawn_agents(&mut env, &[start]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[MoveRight]);

        assert_eq!(buffers.reward[0], env.config.treasure_reward);
        assert_eq!(buffers.last_action[0], VocabId::from(MoveRight));
        assert_ne!(env.state.agents[0].position.idx(), flag.idx());
        assert!(!env.state.base_map[env.state.agents[0].position.idx()].move_blocked());
        assert_eq!(
            env.state.map[env.state.agents[0].position.idx()],
            AgentGeneric
        );
        assert_eq!(env.state.base_map[flag.idx()], TileFlagUnlocked);
    }

    /// Digging clears the wall in front of the agent for good, and costs
    /// `digging_timeout` frozen steps: through them the noop is the only
    /// legal action and moves are ignored, and the first free step moves
    /// into the hole it dug.
    #[test]
    fn digging_clears_a_wall_and_freezes_the_agent() {
        let mut env = empty_env_with(FindReturnConfig {
            num_agents: 1,
            num_flags: 0,
            width: 21,
            height: 21,
            digging_timeout: 2,
            ..Default::default()
        });
        let start = center(&env);
        let wall = start + Position::new(1, 0);
        place_tile(&mut env, wall, TileDestructibleWall);
        spawn_agents(&mut env, &[start]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[MoveRight]); // face the wall, do not pass
        assert_eq!(env.state.agents[0].position.idx(), start.idx());

        step(&mut env, &mut buffers, &[Dig]);
        assert_eq!(env.state.base_map[wall.idx()], TileEmpty);
        assert_eq!(env.state.map[wall.idx()], TileEmpty);
        assert_eq!(env.state.agents[0].timeout, env.config.digging_timeout);
        masked_to_noop(&buffers, 0);

        for _ in 0..env.config.digging_timeout {
            step(&mut env, &mut buffers, &[MoveRight]); // frozen: ignored
            assert_eq!(env.state.agents[0].position.idx(), start.idx());
        }
        assert_eq!(env.state.agents[0].timeout, 0);

        step(&mut env, &mut buffers, &[MoveRight]);
        assert_eq!(env.state.agents[0].position.idx(), wall.idx());
    }

    /// A dig with nothing destructible in front is a free step: no freeze,
    /// no movement, and the mask keeps saying digging is illegal.
    #[test]
    fn digging_needs_something_to_dig() {
        let mut env = empty_env();
        let start = center(&env);
        spawn_agents(&mut env, &[start]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[MoveRight]);
        step(&mut env, &mut buffers, &[Dig]);

        assert_eq!(
            env.state.agents[0].position.idx(),
            (start + Position::new(1, 0)).idx()
        );
        assert_eq!(env.state.agents[0].timeout, 0);
        assert_eq!(
            env.state.map[(start + Position::new(2, 0)).idx()],
            TileEmpty
        );
        let mask = buffers.action_mask.row(0);
        assert!(!mask[Dig as usize]);
        assert!(mask[MoveRight as usize]);
    }

    /// A horizontal run of pipes is a slideway: one move carries the agent
    /// the whole length of it, to the first tile that is not pipe.
    #[test]
    fn a_move_slides_over_pipes() {
        let mut env = empty_env();
        let start = center(&env);
        for dx in 1..=3 {
            place_tile(&mut env, start + Position::new(dx, 0), PipeHorizontal);
        }
        spawn_agents(&mut env, &[start]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[MoveRight]);

        assert_eq!(
            env.state.agents[0].position.idx(),
            (start + Position::new(4, 0)).idx()
        );
    }

    /// A slide runs until the pipe run ends, and a run that ends on a wall
    /// is a trap: the whole move is cancelled rather than stopping short.
    #[test]
    fn a_slide_stops_at_a_wall() {
        let mut env = empty_env();
        let start = center(&env);
        for dx in 1..=2 {
            place_tile(&mut env, start + Position::new(dx, 0), PipeHorizontal);
        }
        place_tile(&mut env, start + Position::new(3, 0), TileWall);
        spawn_agents(&mut env, &[start]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[MoveRight]);

        assert_eq!(env.state.agents[0].position.idx(), start.idx());
        assert_eq!(env.state.map[start.idx()], AgentGeneric);
    }

    /// Sliding only follows a pipe's run: a vertical pipe in the path of a
    /// horizontal move is plain floor, one tile deep.
    #[test]
    fn a_crosswise_pipe_is_just_a_floor_tile() {
        let mut env = empty_env();
        let start = center(&env);
        let pipe = start + Position::new(1, 0);
        place_tile(&mut env, pipe, PipeVirtical);
        spawn_agents(&mut env, &[start]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[MoveRight]);

        assert_eq!(env.state.agents[0].position.idx(), pipe.idx());
        assert_eq!(env.state.map[pipe.idx()], AgentGeneric);
    }

    /// A pipe is laid one tile in front of the facing, perpendicular to it:
    /// facing up or down lays a horizontal pipe, facing left or right a
    /// vertical one, and a blocked tile takes nothing at all.
    #[test]
    fn a_pipe_is_laid_perpendicular_to_the_facing() {
        let mut env = empty_env();
        let start = center(&env);
        spawn_agents(&mut env, &[start]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[MoveUp]);
        step(&mut env, &mut buffers, &[PlacePipe]);
        let above = start + Position::new(0, 2);
        assert_eq!(env.state.map[above.idx()], PipeHorizontal);
        assert_eq!(env.state.base_map[above.idx()], PipeHorizontal);

        step(&mut env, &mut buffers, &[MoveRight]);
        step(&mut env, &mut buffers, &[PlacePipe]);
        let ahead = start + Position::new(2, 1);
        assert_eq!(env.state.map[ahead.idx()], PipeVirtical);
        assert_eq!(env.state.base_map[ahead.idx()], PipeVirtical);

        // a wall in front takes nothing
        let wall = start + Position::new(3, 1);
        place_tile(&mut env, wall, TileWall);
        step(&mut env, &mut buffers, &[MoveRight]); // onto the pipe it just laid
        assert_eq!(env.state.agents[0].position.idx(), ahead.idx());
        step(&mut env, &mut buffers, &[PlacePipe]);
        assert_eq!(env.state.map[wall.idx()], TileWall);
        assert_eq!(env.state.base_map[wall.idx()], TileWall);
    }

    /// Agents are solid: two of them never share a cell, whichever order the
    /// step happens to shuffle them into.
    #[test]
    fn agents_block_each_other() {
        let mut env = empty_env_with(FindReturnConfig {
            num_agents: 2,
            num_flags: 0,
            width: 21,
            height: 21,
            ..Default::default()
        });
        let start = center(&env);
        let other = start + Position::new(1, 0);
        spawn_agents(&mut env, &[start, other]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[MoveRight, MoveLeft]);

        assert_eq!(env.state.agents[0].position.idx(), start.idx());
        assert_eq!(env.state.agents[1].position.idx(), other.idx());
    }

    /// Water stops feet but not eyes: the agent cannot cross the lake, yet
    /// the lake and the ground beyond it both show in its window.
    #[test]
    fn water_blocks_movement_but_not_sight() {
        let mut env = empty_env();
        let start = center(&env);
        let pond = start + Position::new(1, 0);
        place_tile(&mut env, pond, TileWater);
        spawn_agents(&mut env, &[start]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[MoveRight]);

        assert_eq!(env.state.agents[0].position.idx(), start.idx());
        let view = buffers.obs.slice(s![0, .., .., 0]);
        assert_eq!(view[cell(&env, 1, 0)], id(TileWater));
        assert_eq!(view[cell(&env, 2, 0)], id(TileEmpty));
    }

    /// The noop is exactly that: the agent keeps its square and its facing,
    /// and a fresh agent's zero facing keeps its dig aimed at itself, where
    /// there is nothing to dig.
    #[test]
    fn a_noop_changes_nothing() {
        let mut env = empty_env_with(FindReturnConfig {
            num_agents: 1,
            num_flags: 0,
            width: 21,
            height: 21,
            preparation_steps: 0,
            ..Default::default()
        });
        let start = center(&env);
        let flag = start + Position::new(1, 0);
        place_flag(&mut env, flag, TileFlagUnlocked);
        spawn_agents(&mut env, &[start]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Noop]);
        step(&mut env, &mut buffers, &[Dig]);

        assert_eq!(env.state.agents[0].position.idx(), start.idx());
        assert_eq!(buffers.reward[0], 0.0);
        assert_eq!(env.state.agents[0].timeout, 0);
        assert_eq!(env.state.base_map[flag.idx()], TileFlagUnlocked);
        assert_eq!(env.state.map[start.idx()], AgentGeneric);
    }

    /// The UI band is a placeholder, but it has to be a consistent one: the
    /// top rows of every agent's window carry the UI tile and nothing else,
    /// and the field of view stops short of them rather than running under.
    #[test]
    fn the_ui_band_caps_every_window() {
        let mut env = empty_env_with(FindReturnConfig {
            num_agents: 2,
            num_flags: 0,
            width: 21,
            height: 21,
            ..Default::default()
        });
        let start = center(&env);
        spawn_agents(&mut env, &[start, start + Position::new(0, 5)]);

        let mut buffers = TimeStepBuffers::new(&env);
        env.encode_observations(&mut buffers.view_mut());

        let fov_height = env.config.view_height as usize;
        assert_eq!(buffers.obs.dim().1, env.config.view_width as usize);
        assert_eq!(buffers.obs.dim().2, fov_height + 2);

        for agent_id in 0..env.num_agents() {
            let window = buffers.obs.slice(s![agent_id, .., .., 0]);
            for y in 0..fov_height + 2 {
                let row_is_ui =
                    (0..env.config.view_width as usize).all(|x| window[[x, y]] == id(UI));
                let row_has_ui =
                    (0..env.config.view_width as usize).any(|x| window[[x, y]] == id(UI));

                if y >= fov_height {
                    assert!(row_is_ui, "row {y} of agent {agent_id} is not all UI");
                } else {
                    assert!(!row_has_ui, "UI leaked into fov row {y}");
                }
            }
        }
    }

    /// A full episode on generated maps, driven by random actions: the env
    /// has to survive whatever the noise hands it, nobody may share a tile,
    /// and the flag stock can only shrink, never grow: laying a pipe on a
    /// flag erases it, and nothing adds flags back.
    #[test]
    fn a_random_rollout_keeps_the_world_consistent() {
        let config = FindReturnConfig {
            preparation_steps: 0,
            ..Default::default()
        };
        let mut env = FindReturn::new(&config, 256);
        let mut buffers = TimeStepBuffers::new(&env);
        let mut rng = SmallRng::seed_from_u64(11);

        for seed in 0..4 {
            env.reset(seed, &mut buffers.view_mut());

            for _ in 0..256 {
                let actions: Vec<VocabId> = (0..env.num_agents())
                    .map(|_| rng.random_range(0..env.action_spec.num_actions) as VocabId)
                    .collect();
                env.step(&actions, &mut buffers.view_mut());
            }

            let positions: HashSet<_> = env.state.agents.iter().map(|a| a.position.idx()).collect();
            assert_eq!(positions.len(), env.num_agents(), "on seed {seed}");
            for agent in &env.state.agents {
                assert_eq!(env.state.map[agent.position.idx()], AgentGeneric);
                let p = agent.position;
                assert!(p.x >= env.pad_width && p.x < env.width - env.pad_width);
                assert!(p.y >= env.pad_height && p.y < env.height - env.pad_height);
            }

            let (locked, unlocked) = flag_tally(&env);
            assert!(
                locked + unlocked <= config.num_flags,
                "flag stock grew on seed {seed}"
            );
        }
    }
}
