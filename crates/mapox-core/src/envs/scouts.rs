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
        AGENT_HARVESTER, AGENT_SCOUT, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, NOOP,
        TILE_DECOR_1, TILE_DECOR_2, TILE_DECOR_3, TILE_DECOR_4, TILE_EMPTY, TILE_FLAG,
        TILE_FLAG_UNLOCKED, TILE_MASK, TILE_UI, TILE_WALL, TILE_WATER,
    },
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
    vocab_enum,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScoutsConfig {
    pub num_scouts: usize,
    pub num_harvesters: usize,
    pub num_treasures: usize,

    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,
    pub ui_height: i32,

    pub mapgen_threshold: f32,
    pub water_threshold: f32,

    pub harvesters_move_every: u32,

    pub scout_reward: f32,
    pub harvester_reward: f32,
}

impl Default for ScoutsConfig {
    fn default() -> Self {
        Self {
            num_scouts: 4,
            num_harvesters: 4,
            num_treasures: 12,
            width: 40,
            height: 40,
            view_width: 11,
            view_height: 13,
            ui_height: 2,
            mapgen_threshold: 0.3,
            water_threshold: -0.45,
            harvesters_move_every: 6,
            scout_reward: 1.0,
            harvester_reward: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Scout,
    Harvester,
}

impl Role {
    fn tile(self) -> ScoutsObs {
        match self {
            Role::Scout => ScoutsObs::AgentScout,
            Role::Harvester => ScoutsObs::AgentHarvester,
        }
    }
}

#[derive(Debug, Clone)]
struct ScoutsAgent {
    role: Role,
    position: Position,
    timeout: u32,
}

#[derive(Debug, Clone)]
struct ScoutsState {
    rngs: SmallRng,
    agents: Vec<ScoutsAgent>,
    agent_order: Vec<usize>, // agent turn order
    time: usize,

    base_map: Array2<ScoutsObs>, // the bottom layer of the map without agents
    map: Array2<ScoutsObs>,      // the base map plus the agents
    free_positions: Vec<Position>, // these are used to calculate spawn positions
}

vocab_enum!(ScoutsObs {
    UI => TILE_UI,
    Mask => TILE_MASK,
    TileEmpty => TILE_EMPTY,
    TileWall => TILE_WALL,
    TileWater => TILE_WATER,
    TileFlag => TILE_FLAG,
    TileFlagUnlocked => TILE_FLAG_UNLOCKED,
    TileDecor1 => TILE_DECOR_1,
    TileDecor2 => TILE_DECOR_2,
    TileDecor3 => TILE_DECOR_3,
    TileDecor4 => TILE_DECOR_4,
    AgentScout => AGENT_SCOUT,
    AgentHarvester => AGENT_HARVESTER,
});

impl ScoutsObs {
    fn blocked(self) -> bool {
        use ScoutsObs::*;
        matches!(self, TileWall | TileWater)
    }

    fn opaque(self) -> bool {
        matches!(self, ScoutsObs::TileWall)
    }

    fn spawnable(self) -> bool {
        use ScoutsObs::*;
        !self.blocked() && !matches!(self, TileFlag | TileFlagUnlocked)
    }
}

vocab_enum!(ScoutsAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
    Noop => NOOP,
});

impl ScoutsAction {
    fn direction(self) -> Position {
        use ScoutsAction::*;
        match self {
            MoveUp => Position::new(0, 1),
            MoveRight => Position::new(1, 0),
            MoveDown => Position::new(0, -1),
            MoveLeft => Position::new(-1, 0),
            Noop => Position::new(0, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Scouts {
    pub config: ScoutsConfig,
    state: ScoutsState,

    length: usize,

    pad_width: i32,
    pad_height: i32,

    fov_height: i32,

    width: i32,
    height: i32,

    obs_spec: ObservationSpec,
    action_spec: ActionSpec,

    obs_vocab: Vocabulary,
    action_vocab: Vocabulary,
}

impl Scouts {
    pub fn new(config: &ScoutsConfig, length: usize) -> Self {
        let action_vocab = ScoutsAction::vocab();
        let obs_vocab = ScoutsObs::vocab();

        let fov_height = config.view_height - config.ui_height;
        assert!(
            fov_height > 0,
            "ui_height {} leaves no room in a {}-row window",
            config.ui_height,
            config.view_height
        );

        let pad_width = config.view_width / 2;
        let pad_height = fov_height / 2;

        let width = config.width + 2 * pad_width;
        let height = config.height + 2 * pad_height;

        let obs_spec = ObservationSpec::new(config.view_width, config.view_height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        let num_agents = config.num_scouts + config.num_harvesters;

        Self {
            config: config.clone(),
            state: ScoutsState {
                agents: Vec::with_capacity(num_agents),
                agent_order: (0..num_agents).collect(),
                base_map: Array2::from_elem(
                    (width as usize, height as usize),
                    ScoutsObs::TileEmpty,
                ),
                map: Array2::from_elem((width as usize, height as usize), ScoutsObs::TileEmpty),
                free_positions: Vec::new(),
                rngs: SmallRng::seed_from_u64(0),
                time: 0,
            },
            length,

            pad_width,
            pad_height,
            fov_height,
            width,
            height,

            obs_spec,
            action_spec,

            action_vocab,
            obs_vocab,
        }
    }

    fn role(&self, agent_id: usize) -> Role {
        if agent_id < self.config.num_scouts {
            Role::Scout
        } else {
            Role::Harvester
        }
    }

    fn calculate_free_positions(&mut self) {
        self.state.free_positions.clear();
        for x in self.pad_width..self.width - self.pad_width {
            for y in self.pad_height..self.height - self.pad_height {
                let position = Position::new(x, y);

                if self.state.base_map[position.idx()].spawnable() {
                    self.state.free_positions.push(position);
                }
            }
        }

        self.state.free_positions.shuffle(&mut self.state.rngs);
    }

    fn place_treasures(&mut self, count: usize) {
        for _ in 0..count {
            let Some(position) = self.state.free_positions.pop() else {
                return;
            };
            self.state.base_map[position.idx()] = ScoutsObs::TileFlag;
            self.state.map[position.idx()] = ScoutsObs::TileFlag;
        }
    }

    /// Repaint the top layer of the map: the base plus every agent. A
    /// scout can share a cell with a harvester, so harvesters are painted
    /// first and a scout riding one stays visible on top.
    fn repaint(&mut self) {
        self.state.map.assign(&self.state.base_map);
        for role in [Role::Harvester, Role::Scout] {
            for agent in self.state.agents.iter().filter(|a| a.role == role) {
                self.state.map[agent.position.idx()] = role.tile();
            }
        }
    }

    fn claim(&self, role: Role, tile: ScoutsObs) -> Option<(f32, ScoutsObs)> {
        match (role, tile) {
            (Role::Harvester, ScoutsObs::TileFlag) => {
                Some((self.config.harvester_reward, ScoutsObs::TileFlagUnlocked))
            }
            (Role::Scout, ScoutsObs::TileFlagUnlocked) => {
                Some((self.config.scout_reward, ScoutsObs::TileEmpty))
            }
            _ => None,
        }
    }

    fn encode_observations(&self, timestep: &mut TimeStepMut) {
        let fov_height = self.fov_height as usize;

        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            let mut view = timestep.obs.slice_mut(s![agent_id, .., ..fov_height, 0]);
            fov::encode_visible(
                &self.state.map,
                agent.position,
                &mut view,
                ScoutsObs::Mask,
                |tile| tile.opaque(),
            );

            let mut ui = timestep.obs.slice_mut(s![agent_id, .., fov_height.., 0]);
            ui.fill(ScoutsObs::UI as VocabId);
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length);
        timestep.task_ids.fill(0);
    }

    fn encode_action_mask(&self, timestep: &mut TimeStepMut) {
        timestep.action_mask.fill(true);

        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            if agent.timeout > 0 {
                let mut mask = timestep.action_mask.row_mut(agent_id);
                mask.fill(false);
                mask[ScoutsAction::Noop as usize] = true;
            }
        }
    }
}

impl Environment for Scouts {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.state.rngs = SmallRng::seed_from_u64(seed);

        self.state.time = 0;

        let dim = (self.width as usize, self.height as usize);
        if self.state.base_map.dim() != dim {
            self.state.map = Array2::from_elem(dim, ScoutsObs::TileEmpty);
            self.state.base_map = Array2::from_elem(dim, ScoutsObs::TileEmpty);
        }

        self.state.base_map.fill(ScoutsObs::TileWall);
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
                    ScoutsObs::TileWall
                } else if sample < self.config.water_threshold {
                    ScoutsObs::TileWater
                } else {
                    ScoutsObs::TileEmpty
                }
            },
        );

        sprinkle_decor(
            interior,
            ScoutsObs::TileEmpty,
            &[
                ScoutsObs::TileDecor1,
                ScoutsObs::TileDecor2,
                ScoutsObs::TileDecor3,
                ScoutsObs::TileDecor4,
            ],
            &mut self.state.rngs,
        );

        self.state.map.assign(&self.state.base_map);

        self.state.agents.clear();
        self.calculate_free_positions();

        self.place_treasures(self.config.num_treasures);

        // Harvesters take free cells first, then every scout spawns on top
        // of a harvester; a surplus scout cycles onto a shared one.
        let mut ridden: Vec<Position> = Vec::with_capacity(self.config.num_harvesters);
        for _ in 0..self.config.num_harvesters {
            ridden.push(self.state.free_positions.pop().unwrap());
        }
        ridden.shuffle(&mut self.state.rngs);

        for agent_id in 0..self.num_agents() {
            let role = self.role(agent_id);
            let position = match role {
                Role::Scout if !ridden.is_empty() => ridden[agent_id % ridden.len()],
                Role::Scout => self.state.free_positions.pop().unwrap(),
                Role::Harvester => ridden[agent_id - self.config.num_scouts],
            };
            self.state.agents.push(ScoutsAgent {
                role,
                position,
                timeout: 0,
            });
        }
        self.repaint();

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.state.agent_order.shuffle(&mut self.state.rngs);

        for &agent_id in &self.state.agent_order {
            timestep.last_action[agent_id] = actions[agent_id];
            timestep.reward[agent_id] = 0.0;

            let agent = &mut self.state.agents[agent_id];
            if agent.timeout > 0 {
                agent.timeout -= 1;
                continue;
            }

            if agent.role == Role::Harvester {
                agent.timeout = self.config.harvesters_move_every.saturating_sub(1);
            }

            let (role, from) = (agent.role, agent.position);
            let target = from + ScoutsAction::from_id(actions[agent_id]).direction();

            // Only terrain stops an agent: agents walk over each other,
            // whichever role they are.
            if self.state.base_map[target.idx()].blocked() {
                continue;
            }

            if let Some((reward, claimed)) = self.claim(role, self.state.base_map[target.idx()]) {
                self.state.base_map[target.idx()] = claimed;
                timestep.reward[agent_id] = reward;
            }

            self.state.agents[agent_id].position = target;
        }

        self.repaint();
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
        self.config.num_scouts + self.config.num_harvesters
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
            ui_height: self.config.ui_height as usize,
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
    use ScoutsObs::*;
    use rand::RngExt;

    /// One scout and one harvester on bare floor, for poking treasure and
    /// terrain into by hand.
    fn empty_env() -> Scouts {
        empty_env_with(ScoutsConfig {
            num_scouts: 1,
            num_harvesters: 1,
            num_treasures: 0,
            width: 21,
            height: 21,
            ..Default::default()
        })
    }

    fn empty_env_with(config: ScoutsConfig) -> Scouts {
        let mut env = Scouts::new(&config, 512);

        env.state.base_map.fill(ScoutsObs::TileWall);
        env.state
            .base_map
            .slice_mut(s![
                env.pad_width as usize..(env.width - env.pad_width) as usize,
                env.pad_height as usize..(env.height - env.pad_height) as usize,
            ])
            .fill(ScoutsObs::TileEmpty);
        env.state.map.assign(&env.state.base_map);

        env
    }

    fn center(env: &Scouts) -> Position {
        Position::new(env.width / 2, env.height / 2)
    }

    /// Puts every agent on the map, `positions` in agent-id order, so the
    /// scouts come first.
    fn spawn_agents(env: &mut Scouts, positions: &[Position]) {
        for (agent_id, &position) in positions.iter().enumerate() {
            let role = env.role(agent_id);
            env.state.agents.push(ScoutsAgent {
                role,
                position,
                timeout: 0,
            });
            env.state.map[position.idx()] = role.tile();
        }
    }

    fn place_treasure(env: &mut Scouts, position: Position, tile: ScoutsObs) {
        env.state.base_map[position.idx()] = tile;
        env.state.map[position.idx()] = tile;
    }

    /// Steps every agent with the action named for it, and hands back the
    /// buffers so the caller can read rewards out.
    fn step(env: &mut Scouts, buffers: &mut TimeStepBuffers, actions: &[ScoutsAction]) {
        let actions: Vec<VocabId> = actions.iter().map(|&a| a.into()).collect();
        env.step(&actions, &mut buffers.view_mut());
    }

    /// The view-window cell a map offset from the agent lands in. The agent
    /// sits at the centre of the fov, which is the window minus its UI band.
    fn cell(env: &Scouts, dx: i32, dy: i32) -> [usize; 2] {
        [
            (env.config.view_width / 2 + dx) as usize,
            (env.fov_height / 2 + dy) as usize,
        ]
    }

    fn id(tile: ScoutsObs) -> VocabId {
        tile.into()
    }

    /// A locked treasure pays the harvester that breaks it open and nothing
    /// more: it is left standing, open, for a scout to come and spend.
    #[test]
    fn a_harvester_unlocks_a_treasure_without_spending_it() {
        let mut env = empty_env();
        let harvester = center(&env);
        let treasure = harvester + Position::new(1, 0);
        place_treasure(&mut env, treasure, TileFlag);
        spawn_agents(&mut env, &[harvester + Position::new(0, 5), harvester]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(
            &mut env,
            &mut buffers,
            &[ScoutsAction::MoveUp, ScoutsAction::MoveRight],
        );

        assert_eq!(buffers.reward[1], env.config.harvester_reward);
        assert_eq!(env.state.base_map[treasure.idx()], TileFlagUnlocked);
        assert_eq!(env.state.agents[1].position.idx(), treasure.idx());
    }

    /// The other half of the trade: a scout is paid only for a treasure a
    /// harvester already opened, and that spends it for good.
    #[test]
    fn a_scout_spends_an_unlocked_treasure() {
        let mut env = empty_env();
        let scout = center(&env);
        let treasure = scout + Position::new(1, 0);
        place_treasure(&mut env, treasure, TileFlagUnlocked);
        spawn_agents(&mut env, &[scout, scout + Position::new(0, 5)]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(
            &mut env,
            &mut buffers,
            &[ScoutsAction::MoveRight, ScoutsAction::MoveUp],
        );

        assert_eq!(buffers.reward[0], env.config.scout_reward);
        assert_eq!(env.state.base_map[treasure.idx()], TileEmpty);
    }

    /// Neither role can do the other's job: a scout walking over a locked
    /// treasure leaves it locked, and a harvester revisiting one it already
    /// opened is not paid twice.
    #[test]
    fn each_role_only_claims_its_own_half() {
        let mut env = empty_env();
        let start = center(&env);
        let locked = start + Position::new(1, 0);
        let unlocked = start + Position::new(0, 6);
        place_treasure(&mut env, locked, TileFlag);
        place_treasure(&mut env, unlocked, TileFlagUnlocked);
        // the scout stands by the locked treasure, the harvester under the open one
        spawn_agents(&mut env, &[start, start + Position::new(0, 5)]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(
            &mut env,
            &mut buffers,
            &[ScoutsAction::MoveRight, ScoutsAction::MoveUp],
        );

        assert_eq!(buffers.reward[0], 0.0);
        assert_eq!(buffers.reward[1], 0.0);
        assert_eq!(env.state.base_map[locked.idx()], TileFlag);
        assert_eq!(env.state.base_map[unlocked.idx()], TileFlagUnlocked);
    }

    /// A harvester acts on one step in `harvesters_move_every` and stands
    /// still through the rest, while the scout beside it moves every step.
    #[test]
    fn harvesters_act_on_one_step_in_n() {
        let mut env = empty_env_with(ScoutsConfig {
            num_scouts: 1,
            num_harvesters: 1,
            num_treasures: 0,
            width: 21,
            height: 21,
            harvesters_move_every: 3,
            ..Default::default()
        });
        let start = center(&env);
        spawn_agents(&mut env, &[start, start + Position::new(0, 5)]);

        let mut buffers = TimeStepBuffers::new(&env);
        for _ in 0..6 {
            step(
                &mut env,
                &mut buffers,
                &[ScoutsAction::MoveRight, ScoutsAction::MoveRight],
            );
        }

        assert_eq!(env.state.agents[0].position.x, start.x + 6);
        assert_eq!(env.state.agents[1].position.x, start.x + 2);
    }

    /// A spent treasure is gone: the map holds a fixed stock, and an episode
    /// that works through it runs dry rather than topping itself back up.
    #[test]
    fn a_spent_treasure_does_not_come_back() {
        let mut env = empty_env();
        let scout = center(&env);
        let treasure = scout + Position::new(1, 0);
        place_treasure(&mut env, treasure, TileFlagUnlocked);
        spawn_agents(&mut env, &[scout, scout + Position::new(0, 5)]);

        let mut buffers = TimeStepBuffers::new(&env);
        for _ in 0..4 {
            step(
                &mut env,
                &mut buffers,
                &[ScoutsAction::MoveRight, ScoutsAction::MoveUp],
            );
        }

        let treasures = env
            .state
            .base_map
            .iter()
            .filter(|&&tile| matches!(tile, TileFlag | TileFlagUnlocked))
            .count();
        assert_eq!(treasures, 0);
    }

    /// The noop is exactly that: the agent keeps its square, and the treasure
    /// it is standing next to keeps its lock.
    #[test]
    fn a_noop_changes_nothing() {
        let mut env = empty_env();
        let scout = center(&env);
        let treasure = scout + Position::new(1, 0);
        place_treasure(&mut env, treasure, TileFlagUnlocked);
        spawn_agents(&mut env, &[scout, scout + Position::new(0, 5)]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(
            &mut env,
            &mut buffers,
            &[ScoutsAction::Noop, ScoutsAction::Noop],
        );

        assert_eq!(env.state.agents[0].position.idx(), scout.idx());
        assert_eq!(buffers.reward[0], 0.0);
        assert_eq!(env.state.base_map[treasure.idx()], TileFlagUnlocked);
    }

    /// A resting harvester is frozen, and the mask says so: through the steps
    /// between its moves the noop is the only action it may send, while the
    /// scout beside it keeps the full set.
    #[test]
    fn a_resting_harvester_is_masked_to_the_noop() {
        let mut env = empty_env_with(ScoutsConfig {
            num_scouts: 1,
            num_harvesters: 1,
            num_treasures: 0,
            width: 21,
            height: 21,
            harvesters_move_every: 3,
            ..Default::default()
        });
        let start = center(&env);
        spawn_agents(&mut env, &[start, start + Position::new(0, 5)]);

        let mut buffers = TimeStepBuffers::new(&env);
        // the move that spends the harvester's turn
        step(
            &mut env,
            &mut buffers,
            &[ScoutsAction::MoveRight, ScoutsAction::MoveRight],
        );

        for _ in 0..env.config.harvesters_move_every - 1 {
            let mask = buffers.action_mask.row(1);
            assert_eq!(
                mask.iter().filter(|&&legal| legal).count(),
                1,
                "a resting harvester has one action"
            );
            assert!(mask[ScoutsAction::Noop as usize]);
            assert!(buffers.action_mask.row(0).iter().all(|&legal| legal));

            step(
                &mut env,
                &mut buffers,
                &[ScoutsAction::MoveRight, ScoutsAction::Noop],
            );
        }

        // ... and it is free again on the step it may move
        assert!(buffers.action_mask.row(1).iter().all(|&legal| legal));
    }

    /// Water stops feet but not eyes, and unlike find_return's terrain there is
    /// no digging here, so a lake is a permanent divide.
    #[test]
    fn water_blocks_movement_but_not_sight() {
        let mut env = empty_env();
        let start = center(&env);
        let pond = start + Position::new(1, 0);
        place_treasure(&mut env, pond, TileWater);
        spawn_agents(&mut env, &[start, start + Position::new(0, 5)]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(
            &mut env,
            &mut buffers,
            &[ScoutsAction::MoveRight, ScoutsAction::MoveUp],
        );

        assert_eq!(env.state.agents[0].position.idx(), start.idx());

        let view = buffers.obs.slice(s![0, .., .., 0]);
        assert_eq!(view[cell(&env, 1, 0)], id(TileWater));
        assert_eq!(view[cell(&env, 2, 0)], id(TileEmpty));
    }

    /// The two roles read differently in an observation, which is the whole
    /// point of splitting them: a scout can tell a harvester from its own kind.
    #[test]
    fn the_two_roles_are_distinct_tiles() {
        let mut env = empty_env();
        let scout = center(&env);
        spawn_agents(&mut env, &[scout, scout + Position::new(2, 0)]);

        let mut buffers = TimeStepBuffers::new(&env);
        env.encode_observations(&mut buffers.view_mut());

        let view = buffers.obs.slice(s![0, .., .., 0]);
        assert_eq!(view[cell(&env, 0, 0)], id(AgentScout));
        assert_eq!(view[cell(&env, 2, 0)], id(AgentHarvester));
    }

    /// A scout walks over a harvester and the cell they share reads as the
    /// scout: no agent blocks another's movement.
    #[test]
    fn scouts_and_harvesters_share_cells() {
        let mut env = empty_env();
        let scout = center(&env);
        let harvester = scout + Position::new(1, 0);
        spawn_agents(&mut env, &[scout, harvester]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(
            &mut env,
            &mut buffers,
            &[ScoutsAction::MoveRight, ScoutsAction::Noop],
        );

        assert_eq!(
            env.state.agents[0].position.idx(),
            env.state.agents[1].position.idx()
        );
        assert_eq!(env.state.map[harvester.idx()], AgentScout);
    }

    /// The same is true of an agent's own kind: two scouts stack in one
    /// cell rather than refusing it.
    #[test]
    fn agents_of_one_role_walk_over_each_other_too() {
        let mut env = empty_env_with(ScoutsConfig {
            num_scouts: 2,
            num_harvesters: 0,
            num_treasures: 0,
            width: 21,
            height: 21,
            ..Default::default()
        });
        let scout = center(&env);
        let peer = scout + Position::new(1, 0);
        spawn_agents(&mut env, &[scout, peer]);

        let mut buffers = TimeStepBuffers::new(&env);
        step(
            &mut env,
            &mut buffers,
            &[ScoutsAction::MoveRight, ScoutsAction::Noop],
        );

        assert_eq!(
            env.state.agents[0].position.idx(),
            env.state.agents[1].position.idx()
        );
    }

    /// The UI band is a placeholder, but it has to be a consistent one: the
    /// top rows of every agent's window carry the UI tile and nothing else,
    /// and the field of view stops short of them rather than running under.
    #[test]
    fn the_ui_band_caps_every_window() {
        let mut env = empty_env();
        let start = center(&env);
        spawn_agents(&mut env, &[start, start + Position::new(0, 5)]);

        let mut buffers = TimeStepBuffers::new(&env);
        env.encode_observations(&mut buffers.view_mut());

        assert_eq!(
            env.fov_height,
            env.config.view_height - env.config.ui_height
        );

        for agent_id in 0..env.num_agents() {
            let window = buffers.obs.slice(s![agent_id, .., .., 0]);
            for y in 0..env.config.view_height as usize {
                let row_is_ui =
                    (0..env.config.view_width as usize).all(|x| window[[x, y]] == id(UI));
                let row_has_ui =
                    (0..env.config.view_width as usize).any(|x| window[[x, y]] == id(UI));

                if y >= env.fov_height as usize {
                    assert!(row_is_ui, "row {y} of agent {agent_id} is not all UI");
                } else {
                    assert!(!row_has_ui, "UI leaked into fov row {y}");
                }
            }
        }
    }

    /// A full episode on generated maps, driven by random actions: the env has
    /// to survive whatever the noise hands it, and the treasure stock can only
    /// ever be worked down.
    #[test]
    fn a_random_rollout_never_gains_treasure() {
        let config = ScoutsConfig::default();
        let mut env = Scouts::new(&config, 256);
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

            let treasures = env
                .state
                .base_map
                .iter()
                .filter(|&&tile| matches!(tile, TileFlag | TileFlagUnlocked))
                .count();
            assert!(treasures <= config.num_treasures, "on seed {seed}");
        }
    }

    /// Reset has to leave the map consistent: every agent painted on the top
    /// layer, and the full stock of treasures locked on the bottom one.
    #[test]
    fn reset_places_every_agent_and_treasure() {
        let config = ScoutsConfig {
            num_scouts: 2,
            num_harvesters: 3,
            num_treasures: 7,
            width: 24,
            height: 24,
            ..Default::default()
        };
        let mut env = Scouts::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(7, &mut buffers.view_mut());

        assert_eq!(env.state.agents.len(), 5);
        for (agent_id, agent) in env.state.agents.iter().enumerate() {
            assert_eq!(agent.role, env.role(agent_id));
            let painted = env.state.map[agent.position.idx()];
            if agent.role == Role::Scout {
                // a scout is always the tile you see on its cell
                assert_eq!(painted, Role::Scout.tile());
            } else {
                // ... and it is the only thing that can hide a harvester
                assert!(painted == Role::Scout.tile() || painted == Role::Harvester.tile());
            }
        }

        // every scout started life on a harvester's back
        for scout in &env.state.agents[..config.num_scouts] {
            assert!(
                env.state.agents[config.num_scouts..]
                    .iter()
                    .any(|h| h.position.idx() == scout.position.idx()),
                "a scout spawned off the harvesters"
            );
        }

        let locked = env
            .state
            .base_map
            .iter()
            .filter(|&&tile| tile == TileFlag)
            .count();
        assert_eq!(locked, 7);
    }
}
