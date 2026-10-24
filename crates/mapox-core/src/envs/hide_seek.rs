//! Hide and seek, after OpenAI's: two teams on a generated map, the hiders
//! trying to stay out of the seekers' sight.
//!
//! An episode opens with a preparation phase in which the seekers are frozen
//! and blind while the hiders scatter and wall themselves in. Each hider has
//! a fixed stock of walls it can raise on the tile in front of it, and each
//! seeker a fixed stock of walls it can dig out; both counts sit in the top
//! row of the UI band, with the preparation countdown in the row beneath.
//!
//! Sight is the shadowcast every observation is drawn with, so a hider is
//! spotted exactly when it shows in a seeker's observation. Every seeker
//! with a hider in sight at the end of a step is paid for it, and the hider
//! is penalised, flagged terminated, and respawns as a new agent somewhere
//! no seeker can see, with a fresh stock of walls.

use ndarray::{Array2, s};
use rand::{RngExt, SeedableRng, rngs::SmallRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::common::{
        Direction, Position, UI_HEIGHT,
        fov::{self, ViewTile},
        map_gen::{fractal_noise, sprinkle_decor},
        ui::write_number,
        vocab_enum::VocabEnum,
    },
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_HIDER_DOWN, AGENT_HIDER_LEFT, AGENT_HIDER_RIGHT, AGENT_HIDER_UP, AGENT_SEEKER_DOWN,
        AGENT_SEEKER_LEFT, AGENT_SEEKER_RIGHT, AGENT_SEEKER_UP, BUILD_WALL, DIG_ACTION, MOVE_DOWN,
        MOVE_LEFT, MOVE_RIGHT, MOVE_UP, NOOP, TILE_DECOR_1, TILE_DECOR_2, TILE_DECOR_3,
        TILE_DECOR_4, TILE_DESTRUCTIBLE_WALL, TILE_EMPTY, TILE_MASK, TILE_UI, TILE_WALL,
        TILE_WATER, UI_DIGITS,
    },
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
    vocab_enum,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HideSeekConfig {
    pub num_hiders: usize,
    pub num_seekers: usize,

    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,

    pub mapgen_threshold: f32,
    pub water_threshold: f32,

    /// Steps at the start of an episode the seekers stand frozen and blind
    /// while the hiders get away.
    pub preparation_steps: usize,
    /// Walls each hider may build in a life; a respawned hider starts on a
    /// fresh stock.
    pub hider_walls: u32,
    /// Walls each seeker may dig out in an episode.
    pub seeker_digs: u32,

    /// Paid to a seeker for each hider it has in sight.
    pub spot_reward: f32,
    /// Paid to a hider on the step it is spotted, its last as that agent.
    pub spotted_reward: f32,
}

impl Default for HideSeekConfig {
    fn default() -> Self {
        Self {
            num_hiders: 2,
            num_seekers: 2,
            width: 40,
            height: 40,
            view_width: 15,
            view_height: 15,
            mapgen_threshold: 0.07,
            water_threshold: -0.45,
            preparation_steps: 64,
            hider_walls: 8,
            seeker_digs: 8,
            spot_reward: 1.0,
            spotted_reward: -1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Team {
    Hider,
    Seeker,
}

impl Team {
    fn tile(self, facing: Direction) -> HideSeekObs {
        use HideSeekObs::*;
        let by_facing = match self {
            Team::Hider => [
                AgentHiderUp,
                AgentHiderRight,
                AgentHiderDown,
                AgentHiderLeft,
            ],
            Team::Seeker => [
                AgentSeekerUp,
                AgentSeekerRight,
                AgentSeekerDown,
                AgentSeekerLeft,
            ],
        };
        by_facing[facing as usize]
    }
}

#[derive(Debug, Clone)]
struct HideSeekAgent {
    team: Team,
    position: Position,
    facing: Direction, // the way it last tried to move, which is where it builds or digs
    budget: u32,       // walls a hider has left to build, or a seeker to dig
}

impl HideSeekAgent {
    fn front(&self) -> Position {
        self.position + self.facing.to_pos()
    }
}

#[derive(Debug, Clone)]
struct HideSeekState {
    rngs: SmallRng,
    agents: Vec<HideSeekAgent>,
    agent_order: Vec<usize>, // agent turn order
    time: usize,

    base_map: Array2<HideSeekObs>, // the bottom layer of the map without agents
    map: Array2<HideSeekObs>,      // the base map plus the agents
    seen: Array2<bool>,            // what any seeker saw at the end of the last step
    spotted: Vec<usize>,           // hiders spotted on the last step, by agent id
    free_positions: Vec<Position>, // scratch space for picking spawn cells
}

vocab_enum!(HideSeekObs {
    UI => TILE_UI,
    Mask => TILE_MASK,
    TileEmpty => TILE_EMPTY,
    TileDestructibleWall => TILE_DESTRUCTIBLE_WALL,
    TileWall => TILE_WALL,
    TileWater => TILE_WATER,
    TileDecor1 => TILE_DECOR_1,
    TileDecor2 => TILE_DECOR_2,
    TileDecor3 => TILE_DECOR_3,
    TileDecor4 => TILE_DECOR_4,
    AgentHiderUp => AGENT_HIDER_UP,
    AgentHiderRight => AGENT_HIDER_RIGHT,
    AgentHiderDown => AGENT_HIDER_DOWN,
    AgentHiderLeft => AGENT_HIDER_LEFT,
    AgentSeekerUp => AGENT_SEEKER_UP,
    AgentSeekerRight => AGENT_SEEKER_RIGHT,
    AgentSeekerDown => AGENT_SEEKER_DOWN,
    AgentSeekerLeft => AGENT_SEEKER_LEFT,
    Digit0 => UI_DIGITS[0],
    Digit1 => UI_DIGITS[1],
    Digit2 => UI_DIGITS[2],
    Digit3 => UI_DIGITS[3],
    Digit4 => UI_DIGITS[4],
    Digit5 => UI_DIGITS[5],
    Digit6 => UI_DIGITS[6],
    Digit7 => UI_DIGITS[7],
    Digit8 => UI_DIGITS[8],
    Digit9 => UI_DIGITS[9],
});

const DIGIT_TILES: [HideSeekObs; 10] = [
    HideSeekObs::Digit0,
    HideSeekObs::Digit1,
    HideSeekObs::Digit2,
    HideSeekObs::Digit3,
    HideSeekObs::Digit4,
    HideSeekObs::Digit5,
    HideSeekObs::Digit6,
    HideSeekObs::Digit7,
    HideSeekObs::Digit8,
    HideSeekObs::Digit9,
];

impl HideSeekObs {
    /// Bare floor: where an agent can spawn and a hider can build.
    fn open(self) -> bool {
        use HideSeekObs::*;
        matches!(
            self,
            TileEmpty | TileDecor1 | TileDecor2 | TileDecor3 | TileDecor4
        )
    }

    /// Agents stand on floor alone, one to a cell.
    fn move_blocked(self) -> bool {
        !self.open()
    }

    fn destructible(self) -> bool {
        matches!(self, HideSeekObs::TileDestructibleWall)
    }

    fn is_hider(self) -> bool {
        use HideSeekObs::*;
        matches!(
            self,
            AgentHiderUp | AgentHiderRight | AgentHiderDown | AgentHiderLeft
        )
    }
}

impl fov::ViewTile for HideSeekObs {
    const MASK: Self = HideSeekObs::Mask;

    /// Water is the one blocking tile an agent can see straight over.
    fn opaque(self) -> bool {
        use HideSeekObs::*;
        matches!(self, TileWall | TileDestructibleWall)
    }
}

vocab_enum!(HideSeekAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
    Build => BUILD_WALL,
    Dig => DIG_ACTION,
    Noop => NOOP,
});

impl HideSeekAction {
    fn facing(self) -> Option<Direction> {
        use HideSeekAction::*;
        match self {
            MoveUp => Some(Direction::Up),
            MoveRight => Some(Direction::Right),
            MoveDown => Some(Direction::Down),
            MoveLeft => Some(Direction::Left),
            Build | Dig | Noop => None,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct HideSeekMetrics {
    hider_reward: f64,
    seeker_reward: f64,
    hiders_spotted: f64,
    walls_built: f64,
    walls_dug: f64,
}

#[derive(Debug, Clone)]
pub struct HideSeek {
    pub config: HideSeekConfig,
    state: HideSeekState,
    metrics: HideSeekMetrics,

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

impl HideSeek {
    pub fn new(config: &HideSeekConfig, length: usize) -> Self {
        let action_vocab = HideSeekAction::vocab();
        let obs_vocab = HideSeekObs::vocab();

        let pad_width = config.view_width / 2;
        let pad_height = config.view_height / 2;

        let view_height = config.view_height + UI_HEIGHT as i32;

        let width = config.width + 2 * pad_width;
        let height = config.height + 2 * pad_height;
        let dim = (width as usize, height as usize);

        let obs_spec = ObservationSpec::new(config.view_width, view_height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        let num_agents = config.num_hiders + config.num_seekers;

        Self {
            config: config.clone(),
            state: HideSeekState {
                rngs: SmallRng::seed_from_u64(0),
                agents: Vec::with_capacity(num_agents),
                agent_order: (0..num_agents).collect(),
                time: 0,
                base_map: Array2::from_elem(dim, HideSeekObs::TileEmpty),
                map: Array2::from_elem(dim, HideSeekObs::TileEmpty),
                seen: Array2::from_elem(dim, false),
                spotted: Vec::with_capacity(config.num_hiders),
                free_positions: Vec::new(),
            },
            metrics: HideSeekMetrics::default(),
            length,

            pad_width,
            pad_height,
            width,
            height,

            obs_spec,
            action_spec,

            obs_vocab,
            action_vocab,
        }
    }

    fn team(&self, agent_id: usize) -> Team {
        if agent_id < self.config.num_hiders {
            Team::Hider
        } else {
            Team::Seeker
        }
    }

    fn new_agent(&self, agent_id: usize, position: Position) -> HideSeekAgent {
        let team = self.team(agent_id);
        HideSeekAgent {
            team,
            position,
            facing: Direction::Up,
            budget: match team {
                Team::Hider => self.config.hider_walls,
                Team::Seeker => self.config.seeker_digs,
            },
        }
    }

    /// Whether the seekers are still frozen and blind.
    fn preparing(&self) -> bool {
        self.state.time < self.config.preparation_steps
    }

    /// Fills `free_positions` with every open cell of the interior that
    /// `keep` accepts.
    fn collect_free_positions(&mut self, keep: impl Fn(Position) -> bool) {
        let state = &mut self.state;
        state.free_positions.clear();
        for x in self.pad_width..self.width - self.pad_width {
            for y in self.pad_height..self.height - self.pad_height {
                let position = Position::new(x, y);
                if state.map[position.idx()].open() && keep(position) {
                    state.free_positions.push(position);
                }
            }
        }
    }

    /// Carries out one agent's action. A move always turns the agent, and
    /// steps it forward if the cell there is open floor.
    fn act(&mut self, agent_id: usize, action: HideSeekAction) {
        let HideSeekState {
            agents,
            base_map,
            map,
            ..
        } = &mut self.state;
        let agent = &mut agents[agent_id];

        if let Some(facing) = action.facing() {
            agent.facing = facing;
            let target = agent.front();
            if !map[target.idx()].move_blocked() {
                map[agent.position.idx()] = base_map[agent.position.idx()];
                agent.position = target;
            }
            map[agent.position.idx()] = agent.team.tile(agent.facing);
            return;
        }

        let front = agent.front();
        let built = match action {
            HideSeekAction::Build if agent.team == Team::Hider && map[front.idx()].open() => {
                Some(HideSeekObs::TileDestructibleWall)
            }
            HideSeekAction::Dig
                if agent.team == Team::Seeker && map[front.idx()].destructible() =>
            {
                Some(HideSeekObs::TileEmpty)
            }
            _ => None,
        };

        if let Some(tile) = built
            && agent.budget > 0
        {
            base_map[front.idx()] = tile;
            map[front.idx()] = tile;
            agent.budget -= 1;
            match agent.team {
                Team::Hider => self.metrics.walls_built += 1.0,
                Team::Seeker => self.metrics.walls_dug += 1.0,
            }
        }
    }

    /// Sweeps every seeker's view for hiders, pays out for each one in
    /// sight, and sends the spotted hiders off to respawn out of sight.
    fn spot(&mut self, timestep: &mut TimeStepMut) {
        let center = Position::new(self.config.view_width / 2, self.config.view_height / 2);
        let fov_dim = (
            self.config.view_width as usize,
            self.config.view_height as usize,
        );
        let num_hiders = self.config.num_hiders;

        let state = &mut self.state;
        state.seen.fill(false);
        state.spotted.clear();

        let mut in_sight = Vec::new();
        for seeker_id in num_hiders..state.agents.len() {
            // the sweep runs over the seeker's view window, as observe does
            let origin = state.agents[seeker_id].position - center;
            let (map, seen) = (&state.map, &mut state.seen);
            in_sight.clear();
            fov::shadowcast(
                center,
                fov_dim,
                |cell| !map[(origin + cell).idx()].opaque(),
                |cell| {
                    let position = origin + cell;
                    seen[position.idx()] = true;
                    if map[position.idx()].is_hider() {
                        in_sight.push(position);
                    }
                },
            );

            for &position in &in_sight {
                let hider_id = state.agents[..num_hiders]
                    .iter()
                    .position(|hider| hider.position == position)
                    .expect("a hider tile has a hider standing on it");

                timestep.reward[seeker_id] += self.config.spot_reward;
                self.metrics.seeker_reward += f64::from(self.config.spot_reward);
                if !state.spotted.contains(&hider_id) {
                    state.spotted.push(hider_id);
                }
            }
        }

        // lift every spotted hider off the map before placing any of them,
        // so the cells they leave are free to respawn into
        for &hider_id in &state.spotted {
            let position = state.agents[hider_id].position;
            state.map[position.idx()] = state.base_map[position.idx()];

            timestep.reward[hider_id] += self.config.spotted_reward;
            self.metrics.hider_reward += f64::from(self.config.spotted_reward);
            self.metrics.hiders_spotted += 1.0;
        }

        for i in 0..self.state.spotted.len() {
            let hider_id = self.state.spotted[i];
            self.respawn(hider_id);
        }
    }

    /// Puts `agent_id` back on the map as a new agent, on a random open
    /// cell out of every seeker's sight if there is one, and on any open
    /// cell if not.
    fn respawn(&mut self, agent_id: usize) {
        let seen = std::mem::take(&mut self.state.seen);
        self.collect_free_positions(|position| !seen[position.idx()]);
        self.state.seen = seen;
        if self.state.free_positions.is_empty() {
            self.collect_free_positions(|_| true);
        }

        let free = &self.state.free_positions;
        let position = free[self.state.rngs.random_range(0..free.len())];
        let agent = self.new_agent(agent_id, position);
        self.state.map[position.idx()] = agent.team.tile(agent.facing);
        self.state.agents[agent_id] = agent;
    }

    fn encode_observations(&self, timestep: &mut TimeStepMut) {
        let fov_height = self.config.view_height as usize;
        let blind = self.preparing();
        let countdown = self
            .config
            .preparation_steps
            .saturating_sub(self.state.time) as u32;

        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            // wall padding keeps the view window inside the map
            let mut view = timestep.obs.slice_mut(s![agent_id, .., ..fov_height, 0]);
            if blind && agent.team == Team::Seeker {
                view.fill(HideSeekObs::Mask.into());
            } else {
                fov::observe(&self.state.map, agent.position, &mut view);
            }

            // The top row of the band is the agent's stock of walls to build
            // or dig, the one under it the steps left until the seekers move.
            let mut ui = timestep.obs.slice_mut(s![agent_id, .., fov_height.., 0]);
            ui.fill(HideSeekObs::UI.into());
            write_number(
                ui.slice_mut(s![.., UI_HEIGHT - 1]),
                agent.budget,
                &DIGIT_TILES,
            );
            if countdown > 0 {
                write_number(ui.slice_mut(s![.., 0]), countdown, &DIGIT_TILES);
            }
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length);
        timestep.task_ids.fill(0);
    }

    /// A frozen seeker may only wait. Otherwise moves and the noop are always
    /// legal (a move into a wall still turns the agent), a hider may build
    /// onto open floor in front of it and a seeker dig out a destructible
    /// wall there, each while its stock lasts.
    fn encode_action_mask(&self, timestep: &mut TimeStepMut) {
        let frozen = self.preparing();
        timestep.action_mask.fill(true);

        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            let mut mask = timestep.action_mask.row_mut(agent_id);
            if frozen && agent.team == Team::Seeker {
                mask.fill(false);
                mask[HideSeekAction::Noop as usize] = true;
                continue;
            }

            let front = self.state.map[agent.front().idx()];
            let stocked = agent.budget > 0;
            mask[HideSeekAction::Build as usize] =
                stocked && agent.team == Team::Hider && front.open();
            mask[HideSeekAction::Dig as usize] =
                stocked && agent.team == Team::Seeker && front.destructible();
        }
    }
}

impl Environment for HideSeek {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.state.rngs = SmallRng::seed_from_u64(seed);

        self.state.time = 0;
        self.state.spotted.clear();

        let dim = (self.width as usize, self.height as usize);
        if self.state.base_map.dim() != dim {
            self.state.map = Array2::from_elem(dim, HideSeekObs::TileEmpty);
            self.state.base_map = Array2::from_elem(dim, HideSeekObs::TileEmpty);
            self.state.seen = Array2::from_elem(dim, false);
        }
        self.state.seen.fill(false);

        self.state.base_map.fill(HideSeekObs::TileWall);
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
                    HideSeekObs::TileDestructibleWall
                } else if sample < self.config.water_threshold {
                    HideSeekObs::TileWater
                } else {
                    HideSeekObs::TileEmpty
                }
            },
        );

        sprinkle_decor(
            interior,
            HideSeekObs::TileEmpty,
            &[
                HideSeekObs::TileDecor1,
                HideSeekObs::TileDecor2,
                HideSeekObs::TileDecor3,
                HideSeekObs::TileDecor4,
            ],
            &mut self.state.rngs,
        );

        self.state.map.assign(&self.state.base_map);

        // both teams scatter over the open floor, one agent to a cell
        self.collect_free_positions(|_| true);
        self.state.free_positions.shuffle(&mut self.state.rngs);
        self.state.agents.clear();
        for agent_id in 0..self.num_agents() {
            let position = self
                .state
                .free_positions
                .pop()
                .expect("the map has an open cell for every agent");
            let agent = self.new_agent(agent_id, position);
            self.state.map[position.idx()] = agent.team.tile(agent.facing);
            self.state.agents.push(agent);
        }

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        timestep.reward.fill(0.0);
        let frozen = self.preparing();
        self.state.agent_order.shuffle(&mut self.state.rngs);

        for turn in 0..self.state.agent_order.len() {
            let agent_id = self.state.agent_order[turn];
            timestep.last_action[agent_id] = actions[agent_id];

            if frozen && self.team(agent_id) == Team::Seeker {
                continue;
            }
            self.act(agent_id, HideSeekAction::from_id(actions[agent_id]));
        }

        self.state.time += 1;
        // the seekers open their eyes on the step preparation ends, and
        // whoever is in sight then is caught then
        if self.preparing() {
            self.state.spotted.clear();
        } else {
            self.spot(timestep);
        }

        self.encode_observations(timestep);
        for &hider_id in &self.state.spotted {
            timestep.terminated[hider_id] = true;
        }
        self.encode_action_mask(timestep);
    }

    fn consume_metrics(&mut self) -> serde_json::Value {
        let metrics = std::mem::take(&mut self.metrics);
        let agents = self.num_agents().max(1) as f64;
        let hiders = self.config.num_hiders.max(1) as f64;
        let seekers = self.config.num_seekers.max(1) as f64;
        serde_json::json!({
            "reward": (metrics.hider_reward + metrics.seeker_reward) / agents,
            "hider_reward": metrics.hider_reward / hiders,
            "seeker_reward": metrics.seeker_reward / seekers,
            "hiders_spotted": metrics.hiders_spotted / hiders,
            "walls_built": metrics.walls_built / hiders,
            "walls_dug": metrics.walls_dug / seekers,
        })
    }

    fn observation_spec(&self) -> ObservationSpec {
        self.obs_spec
    }

    fn action_spec(&self) -> ActionSpec {
        self.action_spec
    }

    fn num_agents(&self) -> usize {
        self.config.num_hiders + self.config.num_seekers
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
            view_height: self.config.view_height as usize + UI_HEIGHT,
            ui_height: UI_HEIGHT,
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
    use HideSeekAction::*;
    use HideSeekObs::*;
    use std::collections::HashSet;

    /// One hider and one seeker on bare floor with no preparation phase, for
    /// poking walls into by hand.
    fn empty_env() -> HideSeek {
        empty_env_with(HideSeekConfig {
            num_hiders: 1,
            num_seekers: 1,
            width: 21,
            height: 21,
            preparation_steps: 0,
            ..Default::default()
        })
    }

    fn empty_env_with(config: HideSeekConfig) -> HideSeek {
        let mut env = HideSeek::new(&config, 512);

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

    fn center(env: &HideSeek) -> Position {
        Position::new(env.width / 2, env.height / 2)
    }

    /// Puts every agent on the map, `positions` in agent-id order, so the
    /// hiders come first.
    fn spawn_agents(env: &mut HideSeek, positions: &[Position]) {
        for (agent_id, &position) in positions.iter().enumerate() {
            let agent = env.new_agent(agent_id, position);
            env.state.map[position.idx()] = agent.team.tile(agent.facing);
            env.state.agents.push(agent);
        }
    }

    fn place_tile(env: &mut HideSeek, position: Position, tile: HideSeekObs) {
        env.state.base_map[position.idx()] = tile;
        env.state.map[position.idx()] = tile;
    }

    fn step(env: &mut HideSeek, buffers: &mut TimeStepBuffers, actions: &[HideSeekAction]) {
        let actions: Vec<VocabId> = actions.iter().map(|&a| a.into()).collect();
        env.step(&actions, &mut buffers.view_mut());
    }

    /// Encodes the current state without stepping it.
    fn observe(env: &HideSeek) -> TimeStepBuffers {
        let mut buffers = TimeStepBuffers::new(env);
        env.encode_observations(&mut buffers.view_mut());
        env.encode_action_mask(&mut buffers.view_mut());
        buffers
    }

    /// The UI band row `row` of `agent_id`'s window, `0` the lower one.
    fn ui_row(
        env: &HideSeek,
        buffers: &TimeStepBuffers,
        agent_id: usize,
        row: usize,
    ) -> Vec<VocabId> {
        let y = env.config.view_height as usize + row;
        buffers.obs.slice(s![agent_id, .., y, 0]).to_vec()
    }

    /// `value` right-aligned in a band row, the rest of it plain UI.
    fn number_row(env: &HideSeek, value: &str) -> Vec<VocabId> {
        let width = env.config.view_width as usize;
        let mut row = vec![VocabId::from(UI); width - value.len()];
        row.extend(
            value
                .bytes()
                .map(|digit| VocabId::from(DIGIT_TILES[usize::from(digit - b'0')])),
        );
        row
    }

    fn blank_row(env: &HideSeek) -> Vec<VocabId> {
        vec![UI.into(); env.config.view_width as usize]
    }

    /// The mask of `agent_id` as the set of actions it allows.
    fn legal(buffers: &TimeStepBuffers, agent_id: usize) -> Vec<HideSeekAction> {
        HideSeekAction::TABLE
            .iter()
            .copied()
            .filter(|&action| buffers.action_mask[[agent_id, action as usize]])
            .collect()
    }

    /// Through the preparation phase a seeker sees nothing, may only wait,
    /// and stays put whatever it sends; the hider beside it moves freely
    /// and is not caught however plainly it stands in view.
    #[test]
    fn seekers_are_frozen_and_blind_while_hiders_prepare() {
        let mut env = empty_env_with(HideSeekConfig {
            num_hiders: 1,
            num_seekers: 1,
            width: 21,
            height: 21,
            preparation_steps: 3,
            ..Default::default()
        });
        let start = center(&env);
        let seeker = start + Position::new(0, 3);
        spawn_agents(&mut env, &[start, seeker]);

        let mut buffers = observe(&env);
        for _ in 0..env.config.preparation_steps - 1 {
            assert_eq!(legal(&buffers, 1), vec![Noop]);
            let view = buffers
                .obs
                .slice(s![1, .., ..env.config.view_height as usize, 0]);
            assert!(view.iter().all(|&tile| tile == VocabId::from(Mask)));

            step(&mut env, &mut buffers, &[MoveRight, MoveRight]);
            assert_eq!(env.state.agents[1].position, seeker);
            assert_eq!(buffers.reward[0], 0.0);
            assert!(!buffers.terminated[0]);
        }
        assert_eq!(env.state.agents[0].position, start + Position::new(2, 0));

        // the last step of preparation opens the seeker's eyes on the hider
        step(&mut env, &mut buffers, &[Noop, MoveRight]);
        assert_eq!(env.state.agents[1].position, seeker);
        assert_eq!(buffers.reward[1], env.config.spot_reward);
        assert!(buffers.terminated[0]);
        assert!(legal(&buffers, 1).contains(&MoveRight));
    }

    /// A hider raises a wall on the tile it faces, one off its stock, and
    /// the band counts the stock down; spent, the build is masked out and
    /// does nothing.
    #[test]
    fn a_hider_builds_walls_until_its_stock_runs_out() {
        let mut env = empty_env_with(HideSeekConfig {
            num_hiders: 1,
            num_seekers: 1,
            width: 21,
            height: 21,
            hider_walls: 2,
            preparation_steps: 100,
            ..Default::default()
        });
        let start = center(&env);
        spawn_agents(&mut env, &[start, start + Position::new(0, 9)]);

        let mut buffers = observe(&env);
        assert_eq!(ui_row(&env, &buffers, 0, 1), number_row(&env, "2"));

        step(&mut env, &mut buffers, &[Build, Noop]);
        assert_eq!(
            env.state.map[(start + Position::new(0, 1)).idx()],
            TileDestructibleWall
        );
        assert_eq!(ui_row(&env, &buffers, 0, 1), number_row(&env, "1"));
        // the wall in front leaves nothing to build on
        assert!(!legal(&buffers, 0).contains(&Build));

        step(&mut env, &mut buffers, &[MoveRight, Noop]);
        step(&mut env, &mut buffers, &[Build, Noop]);
        let second = start + Position::new(2, 0);
        assert_eq!(env.state.base_map[second.idx()], TileDestructibleWall);
        assert_eq!(ui_row(&env, &buffers, 0, 1), number_row(&env, "0"));

        step(&mut env, &mut buffers, &[MoveUp, Noop]);
        assert!(!legal(&buffers, 0).contains(&Build));
        step(&mut env, &mut buffers, &[Build, Noop]);
        let ahead = env.state.agents[0].front();
        assert_eq!(env.state.base_map[ahead.idx()], TileEmpty);
    }

    /// A hider builds only onto open floor: not over water, another agent,
    /// or a wall already standing.
    #[test]
    fn hiders_build_only_on_open_floor() {
        let mut env = empty_env_with(HideSeekConfig {
            num_hiders: 1,
            num_seekers: 1,
            width: 21,
            height: 21,
            preparation_steps: 100,
            ..Default::default()
        });
        let start = center(&env);
        place_tile(&mut env, start + Position::new(1, 0), TileWater);
        spawn_agents(&mut env, &[start, start + Position::new(0, 1)]);
        let mut buffers = observe(&env);

        // facing the seeker
        assert!(!legal(&buffers, 0).contains(&Build));
        step(&mut env, &mut buffers, &[Build, Noop]);
        assert_eq!(env.state.agents[0].budget, env.config.hider_walls);

        // facing water
        step(&mut env, &mut buffers, &[MoveRight, Noop]);
        assert!(!legal(&buffers, 0).contains(&Build));
        step(&mut env, &mut buffers, &[Build, Noop]);
        assert_eq!(
            env.state.base_map[(start + Position::new(1, 0)).idx()],
            TileWater
        );
        assert_eq!(env.state.agents[0].budget, env.config.hider_walls);
    }

    /// A seeker digs out the wall it faces, generated or built, one off its
    /// stock; spent, it can dig no more. The border is not diggable.
    #[test]
    fn a_seeker_digs_walls_until_its_stock_runs_out() {
        let mut env = empty_env_with(HideSeekConfig {
            num_hiders: 1,
            num_seekers: 1,
            width: 21,
            height: 21,
            seeker_digs: 1,
            preparation_steps: 0,
            ..Default::default()
        });
        let start = center(&env);
        let wall = start + Position::new(0, 1);
        place_tile(&mut env, wall, TileDestructibleWall);
        place_tile(&mut env, wall + Position::new(0, 1), TileDestructibleWall);
        // the hider waits in a far corner, out of the seeker's sight
        let hider = Position::new(env.pad_width, env.pad_height);
        spawn_agents(&mut env, &[hider, start]);

        let mut buffers = observe(&env);
        assert_eq!(ui_row(&env, &buffers, 1, 1), number_row(&env, "1"));
        assert!(legal(&buffers, 1).contains(&Dig));

        step(&mut env, &mut buffers, &[Noop, Dig]);
        assert_eq!(env.state.map[wall.idx()], TileEmpty);
        assert_eq!(ui_row(&env, &buffers, 1, 1), number_row(&env, "0"));

        step(&mut env, &mut buffers, &[Noop, MoveUp]);
        assert_eq!(env.state.agents[1].position, wall);
        assert!(!legal(&buffers, 1).contains(&Dig));
        step(&mut env, &mut buffers, &[Noop, Dig]);
        assert_eq!(
            env.state.map[(wall + Position::new(0, 1)).idx()],
            TileDestructibleWall
        );
    }

    #[test]
    fn seekers_cannot_dig_the_border() {
        let mut env = empty_env();
        let edge = Position::new(env.pad_width, env.height / 2);
        spawn_agents(&mut env, &[edge + Position::new(12, 9), edge]);
        let mut buffers = observe(&env);

        step(&mut env, &mut buffers, &[Noop, MoveLeft]);
        assert!(!legal(&buffers, 1).contains(&Dig));
        step(&mut env, &mut buffers, &[Noop, Dig]);
        assert_eq!(
            env.state.base_map[(edge + Position::new(-1, 0)).idx()],
            TileWall
        );
        assert_eq!(env.state.agents[1].budget, env.config.seeker_digs);
    }

    /// Each team has only its own half of the walls: a hider cannot dig and a
    /// seeker cannot build.
    #[test]
    fn neither_team_has_the_others_action() {
        let mut env = empty_env();
        let start = center(&env);
        place_tile(&mut env, start + Position::new(0, 1), TileDestructibleWall);
        // the hider faces a wall and the seeker, out of its sight, open floor
        spawn_agents(&mut env, &[start, start + Position::new(9, 0)]);
        let mut buffers = observe(&env);

        assert!(!legal(&buffers, 0).contains(&Dig));
        assert!(!legal(&buffers, 1).contains(&Build));
        step(&mut env, &mut buffers, &[Dig, Build]);
        assert_eq!(
            env.state.base_map[(start + Position::new(0, 1)).idx()],
            TileDestructibleWall
        );
        assert_eq!(
            env.state.base_map[(start + Position::new(9, 1)).idx()],
            TileEmpty
        );
    }

    /// A hider in a seeker's sight is spotted: the seeker is paid, the hider
    /// penalised and terminated, and it comes back as a new agent with a
    /// full stock somewhere the seeker cannot see.
    #[test]
    fn a_spotted_hider_respawns_out_of_sight() {
        let mut env = empty_env_with(HideSeekConfig {
            num_hiders: 1,
            num_seekers: 1,
            width: 30,
            height: 30,
            hider_walls: 3,
            preparation_steps: 0,
            ..Default::default()
        });
        let start = center(&env);
        spawn_agents(&mut env, &[start, start + Position::new(4, 0)]);
        env.state.agents[0].budget = 1;
        let mut buffers = observe(&env);

        step(&mut env, &mut buffers, &[Noop, Noop]);
        assert_eq!(buffers.reward[0], env.config.spotted_reward);
        assert_eq!(buffers.reward[1], env.config.spot_reward);
        assert!(buffers.terminated[0]);
        assert!(!buffers.terminated[1]);

        let hider = &env.state.agents[0];
        assert_eq!(hider.budget, env.config.hider_walls);
        assert_eq!(
            env.state.map[hider.position.idx()],
            Team::Hider.tile(hider.facing)
        );
        assert!(env.state.map[start.idx()].open(), "the old body is gone");
        let seeker = env.state.agents[1].position;
        let (dx, dy) = (hider.position.x - seeker.x, hider.position.y - seeker.y);
        let half = (env.config.view_width / 2, env.config.view_height / 2);
        assert!(dx.abs() > half.0 || dy.abs() > half.1, "respawned in sight");

        // and the new agent is not in the seeker's observation
        let view = buffers
            .obs
            .slice(s![1, .., ..env.config.view_height as usize, 0]);
        assert!(
            !view
                .iter()
                .any(|&tile| HideSeekObs::from_id(tile).is_hider())
        );
    }

    /// A wall between them hides a hider from a seeker that would otherwise
    /// see it; water does not.
    #[test]
    fn walls_block_sight_and_water_does_not() {
        for (cover, hidden) in [(TileDestructibleWall, true), (TileWater, false)] {
            let mut env = empty_env();
            let start = center(&env);
            for dy in -1..=1 {
                place_tile(&mut env, start + Position::new(1, dy), cover);
            }
            spawn_agents(&mut env, &[start, start + Position::new(3, 0)]);
            let mut buffers = observe(&env);

            step(&mut env, &mut buffers, &[Noop, Noop]);
            assert_eq!(buffers.terminated[0], !hidden, "behind {cover:?}");
            let paid = if hidden { 0.0 } else { env.config.spot_reward };
            assert_eq!(buffers.reward[1], paid, "behind {cover:?}");
        }
    }

    /// Every seeker with the hider in view is paid for it; the hider pays
    /// once however many see it.
    #[test]
    fn every_seeker_in_sight_is_paid_and_the_hider_once() {
        let mut env = empty_env_with(HideSeekConfig {
            num_hiders: 1,
            num_seekers: 2,
            width: 30,
            height: 30,
            preparation_steps: 0,
            ..Default::default()
        });
        let start = center(&env);
        spawn_agents(
            &mut env,
            &[
                start,
                start + Position::new(2, 0),
                start + Position::new(-2, 0),
            ],
        );
        let mut buffers = observe(&env);

        step(&mut env, &mut buffers, &[Noop, Noop, Noop]);
        assert_eq!(buffers.reward[0], env.config.spotted_reward);
        assert_eq!(buffers.reward[1], env.config.spot_reward);
        assert_eq!(buffers.reward[2], env.config.spot_reward);

        let metrics = env.consume_metrics();
        assert_eq!(metrics["hiders_spotted"], 1.0);
        assert_eq!(metrics["seeker_reward"], env.config.spot_reward as f64);
    }

    /// Agents stand one to a cell, and a move into another agent only turns
    /// the mover; its sprite shows which way it now faces.
    #[test]
    fn agents_block_each_other_and_turn_in_place() {
        let mut env = empty_env_with(HideSeekConfig {
            num_hiders: 2,
            num_seekers: 0,
            width: 21,
            height: 21,
            ..Default::default()
        });
        let start = center(&env);
        spawn_agents(&mut env, &[start, start + Position::new(1, 0)]);
        let mut buffers = observe(&env);

        step(&mut env, &mut buffers, &[MoveRight, Noop]);
        assert_eq!(env.state.agents[0].position, start);
        assert_eq!(env.state.map[start.idx()], AgentHiderRight);
        assert_eq!(
            env.state.map[(start + Position::new(1, 0)).idx()],
            AgentHiderUp
        );
    }

    /// The lower band row counts down the steps until the seekers move, and
    /// is blank from then on.
    #[test]
    fn the_band_counts_down_preparation() {
        let mut env = empty_env_with(HideSeekConfig {
            num_hiders: 1,
            num_seekers: 1,
            width: 21,
            height: 21,
            preparation_steps: 12,
            ..Default::default()
        });
        let start = center(&env);
        spawn_agents(&mut env, &[start, start + Position::new(0, 9)]);
        let mut buffers = observe(&env);

        assert_eq!(ui_row(&env, &buffers, 0, 0), number_row(&env, "12"));
        assert_eq!(ui_row(&env, &buffers, 1, 0), number_row(&env, "12"));
        for _ in 0..3 {
            step(&mut env, &mut buffers, &[Noop, Noop]);
        }
        assert_eq!(ui_row(&env, &buffers, 0, 0), number_row(&env, "9"));
        for _ in 0..9 {
            step(&mut env, &mut buffers, &[Noop, Noop]);
        }
        assert_eq!(ui_row(&env, &buffers, 0, 0), blank_row(&env));
    }

    #[test]
    fn metrics_before_steps_have_float_schema_even_without_agents() {
        for (num_hiders, num_seekers) in [(0, 0), (2, 2)] {
            let mut env = HideSeek::new(
                &HideSeekConfig {
                    num_hiders,
                    num_seekers,
                    ..Default::default()
                },
                512,
            );
            let metrics = env.consume_metrics();
            let values = metrics.as_object().unwrap();
            assert_eq!(values.len(), 6);
            assert!(values.values().all(|value| value.as_f64() == Some(0.0)));
        }
    }

    /// Reset scatters both teams over distinct open cells, every one painted
    /// as its team, with a full stock on the band.
    #[test]
    fn reset_places_every_agent_on_its_own_open_cell() {
        let config = HideSeekConfig {
            num_hiders: 3,
            num_seekers: 2,
            width: 24,
            height: 24,
            ..Default::default()
        };
        let mut env = HideSeek::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(7, &mut buffers.view_mut());

        assert_eq!(env.state.agents.len(), 5);
        let cells: HashSet<_> = env.state.agents.iter().map(|a| a.position).collect();
        assert_eq!(cells.len(), 5);
        for (agent_id, agent) in env.state.agents.iter().enumerate() {
            assert_eq!(agent.team, env.team(agent_id));
            assert!(env.state.base_map[agent.position.idx()].open());
            assert_eq!(
                env.state.map[agent.position.idx()],
                agent.team.tile(agent.facing)
            );
        }
        assert_eq!(ui_row(&env, &buffers, 0, 1), number_row(&env, "8"));
        assert_eq!(ui_row(&env, &buffers, 4, 1), number_row(&env, "8"));
    }

    /// A full episode on generated maps, driven by random actions: the env has
    /// to survive whatever the noise hands it, and agents never end up on a
    /// wall or on each other.
    #[test]
    fn a_random_rollout_keeps_agents_on_open_floor() {
        let config = HideSeekConfig {
            num_hiders: 4,
            num_seekers: 4,
            preparation_steps: 32,
            ..Default::default()
        };
        let mut env = HideSeek::new(&config, 256);
        let mut buffers = TimeStepBuffers::new(&env);
        let mut rng = SmallRng::seed_from_u64(11);

        for seed in 0..4 {
            env.reset(seed, &mut buffers.view_mut());

            for _ in 0..256 {
                let actions: Vec<VocabId> = (0..env.num_agents())
                    .map(|agent_id| {
                        let legal: Vec<_> = (0..env.action_spec.num_actions)
                            .filter(|&a| buffers.action_mask[[agent_id, a]])
                            .collect();
                        legal[rng.random_range(0..legal.len())] as VocabId
                    })
                    .collect();
                env.step(&actions, &mut buffers.view_mut());

                let cells: HashSet<_> = env.state.agents.iter().map(|a| a.position).collect();
                assert_eq!(cells.len(), env.num_agents(), "on seed {seed}");
                for agent in &env.state.agents {
                    assert!(env.state.base_map[agent.position.idx()].open());
                    assert_eq!(
                        env.state.map[agent.position.idx()],
                        agent.team.tile(agent.facing)
                    );
                }
            }
            assert!(
                buffers.terminated.iter().all(|&t| t),
                "the episode ends at its length"
            );
        }
        let metrics = env.consume_metrics();
        assert!(
            metrics["hiders_spotted"].as_f64().unwrap() > 0.0,
            "random seekers never found anyone"
        );
    }
}
