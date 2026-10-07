//! The survival env: a game built from systems that overlap on one shared
//! world and play by the same rules.
//!
//! The world is three layers on one grid, in [`SurvivalState`]: `base_map`
//! is the ground (terrain, and items lying on it), `map` is the ground with
//! the creatures drawn over it, and `lit` is the ground fires and torches
//! light. Systems
//! change them only through the shared rules here, [`Survival::set_ground`]
//! for the ground and [`Survival::place_creature`] and
//! [`Survival::remove_creature`] for creatures, and they judge a cell by the
//! tile predicates in [`tiles`] (`walkable`, `is_floor`, `opaque`, ...)
//! rather than by matching tiles. So any creature hides the item it stands
//! on, any system's ground change shows under a creature once it steps off,
//! and a new tile or creature means the same to every system.
//!
//! Each system has its own module:
//! - [`actions`]: what agents can do, and doing it
//! - [`survivor`]: the agents' stats, life and death
//! - [`items`]: items, recipes and jobs
//! - [`fire`]: fires and torches burning down, and the light they cast
//! - [`plants`]: bushes fruiting again (carrots and tall grass don't regrow)
//! - [`clock`]: day, dusk and night
//! - [`season`]: summer, and winter at the end
//! - [`spiders`]: nests and the spiders out of them
//! - [`terrain`]: map generation
//! - [`obs`]: what each agent sees
//!
//! A step runs them in a fixed order: the agents act, one at a time in a
//! shuffled order; then the world takes its turn ([`Survival::tick_world`]);
//! then hunger, temperature and health tick, each agent is rewarded by
//! them, and the dead respawn.

mod actions;
mod clock;
mod config;
mod fire;
mod items;
mod metrics;
mod obs;
mod plants;
mod season;
mod spiders;
mod survivor;
mod terrain;
#[cfg(test)]
mod tests;
mod tiles;

use std::collections::VecDeque;

use ndarray::{Array2, s};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

pub use config::{MAX_STAT, SurvivalConfig};

use crate::{
    env::Environment,
    envs::common::{Position, UI_HEIGHT, map_gen::spread_out, vocab_enum::VocabEnum},
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
};
use actions::SurvivalAction;
use fire::Fire;
use metrics::SurvivalMetrics;
use obs::UI_WIDTH;
use spiders::Spider;
use survivor::Survivor;
use tiles::SurvivalObs;

#[derive(Debug, Clone)]
struct SurvivalState {
    rng: SmallRng,
    agents: Vec<Survivor>,
    agent_order: Vec<usize>, // agent turn order
    time: usize,

    base_map: Array2<SurvivalObs>, // terrain and ground items, without agents
    map: Array2<SurvivalObs>,      // the base map plus the agents
    fires: Vec<Fire>,
    /// Torches lying on the ground, and how far each had burnt when it was
    /// set down.
    laid_torches: Vec<(Position, u32)>,
    /// Picked bushes and the step each fruits again, in that order.
    regrowing: VecDeque<(Position, usize)>,
    /// Ground a fire or a held torch lights: spiders keep off it, and agents
    /// see it at night as far as by day. Brought up to date every world turn.
    lit: Array2<bool>,
    /// Where the spider nests are, and the spiders out of them.
    eggs: Vec<Position>,
    spiders: Vec<Spider>,
    free_positions: Vec<Position>, // these are used to calculate spawn positions
}

/// A first pass at a survival game: agents keep themselves fed off berry
/// bushes and carrots, and craft from what lies around.
///
/// Each agent has health, hunger and temperature, 0 to [`MAX_STAT`]. Hunger
/// drains with time; while it is high health grows back, and at zero health
/// drains instead. Temperature rises on ground a fire lights, in any season,
/// and in winter drops everywhere else; at zero it drains health too. An
/// agent whose health runs out drops what it carries where it stood, is
/// flagged terminated, and respawns with fresh stats elsewhere.
///
/// Everything an agent does to the world it does to the tile in front of it,
/// and its tile shows which way that is. Moving turns the agent to face the
/// move even when the way is blocked, so turning in place is a blocked move.
/// Items on the ground don't block, nor do bushes and tall grass: an agent
/// walks over them, hiding the one it stands on, and has to step off and face
/// it to pick it up.
/// The inventory is two slots, hands and backpack. Grab takes the item in
/// front (or the berries off a bush, or a buried carrot out of the ground)
/// into empty hands, or into the backpack if the hands are full; swap trades
/// hands and backpack. Put puts the hand's item on what is in front: on open
/// ground it is set down, except a campfire, which is lit there; a raw berry
/// or carrot put to a fire cooks at once, back into hand; and wood, a stick
/// or grass put on a fire burning low stokes it, wood by `wood_fuel` steps,
/// a stick by `stick_fuel` and grass by `grass_fuel`.
/// Use uses the hand's item, or bare hands: food is eaten, a berry or a
/// carrot, raw or, feeding more, cooked; the axe sets to felling a tree or
/// clearing a bush in front, bare hands to harvesting tall grass.
/// Felling is a [`Job`](items::Job), slow work: the agent can only wait until
/// it is done, `chop_steps` in all, and then a log lies where the tree stood.
/// Clearing a bush, ripe, bare or dead, is another (`clear_bush_steps`), and
/// leaves a stick; harvesting tall grass is a third, done by using empty
/// hands on it: after `harvest_steps` a bundle of grass lies where it grew.
/// Carrots are buried at reset and never grow back. Combine turns the
/// hand and backpack items into a new one per [`RECIPES`](items::RECIPES):
/// stick and stone make an axe, wood and grass a campfire, stick and grass a
/// torch. A torch in hand lights the ground `torch_light_radius` around its
/// holder, as a fire does but without warming it, and burns down a step for
/// every step it is held, gone after `torch_burn_steps`.
///
/// Days and nights alternate, starting at dawn. By night an agent sees only
/// `night_vision_radius` around itself, except for ground a fire lights,
/// which it sees as far as by day. Night closes in gradually: over the last
/// `dusk_length` steps of the day, sight shrinks step by step from the
/// whole view down to the night's, so the dark is a warning before it is a
/// danger. Held torches light the dark too. A fire burns `fire_burn_steps`, the last
/// `fire_low_steps` of them low, when fuel stokes it back up.
/// Spider nests hatch a giant spider each at nightfall, and the spiders hunt
/// until dawn: each bites an agent next to it, or walks toward the nearest it
/// can track, or wanders. At dawn they walk back to their nests and burrow in;
/// a nest hatches again only once its spider is home. Spiders fear fire. They
/// never set foot on lit ground, can't bite or track an agent standing on it,
/// and one caught in a fire's light walks straight out of it, so a fire is
/// the safe place to spend the night.
///
/// The map is biomes. An elevation field floods the low ground and raises
/// dirt banks on the heights, diggable walls once a tool can dig them; the
/// map's edge is solid wall. A moisture field splits the land between into
/// forest (trees and sticks), meadow (berry bushes, tall grass and buried
/// carrots) and scrub (stones), so where to look for a thing is something to
/// learn. Pockets of open ground
/// too small to matter are filled in and the rest joined up by paths, so
/// every agent can reach every other, and agents start out of each other's
/// sight where the map allows. Spider nests go in the forest.
///
/// The episode ends in winter, its last `winter_length` steps. When it sets
/// in, every bush and all the tall grass die and the water freezes into ice
/// that can be walked on, for the rest of the episode; berries stop growing,
/// and the cold drains temperature away from the fires. Buried carrots keep.
///
/// Every step an agent ends alive earns it `alive_reward`, less penalties
/// for missing health, for hunger below `hunger_threshold` and for
/// temperature below `temperature_threshold`, each growing as its stat
/// falls. The metrics count reward, deaths and per-life achievements.
///
/// The UI band shows health, hunger and temperature as numbers, what is in
/// hands and backpack, whether it is day or night, and whether it is winter
/// (see
/// [`HEALTH_COL`](obs::HEALTH_COL) and friends for the layout).
#[derive(Debug, Clone)]
pub struct Survival {
    pub config: SurvivalConfig,
    state: SurvivalState,
    metrics: SurvivalMetrics,

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

impl Survival {
    pub fn new(config: &SurvivalConfig, length: usize) -> Self {
        assert!(
            config.view_width as usize >= UI_WIDTH,
            "the UI band needs a view at least {UI_WIDTH} wide"
        );
        assert!(
            config.view_height >= 3,
            "the map padding has to reach the tile in front of an agent"
        );
        assert!(
            config.start_hunger <= MAX_STAT
                && config.start_health <= MAX_STAT
                && config.start_temperature <= MAX_STAT,
            "starting stats run past {MAX_STAT}"
        );
        assert!(
            config.start_health > 0,
            "agents would spawn dead at zero health"
        );
        assert!(
            config.hunger_interval > 0 && config.regen_interval > 0 && config.chill_interval > 0,
            "stat intervals must be at least one step"
        );
        let share = |fraction: f64| (0.0..=1.0).contains(&fraction);
        assert!(
            share(config.water_fraction)
                && share(config.rock_fraction)
                && share(config.water_fraction + config.rock_fraction)
                && share(config.forest_fraction)
                && share(config.scrub_fraction)
                && share(config.forest_fraction + config.scrub_fraction),
            "terrain fractions are shares, and add up to at most the whole"
        );
        assert!(config.day_length > 0, "every cycle starts with a day");
        assert!(
            config.dusk_length <= config.day_length,
            "dusk is the end of the day, so no longer than it"
        );

        let action_vocab = SurvivalAction::vocab();
        let obs_vocab = SurvivalObs::vocab();

        let pad_width = config.view_width / 2;
        let pad_height = config.view_height / 2;

        let view_height = config.view_height + UI_HEIGHT as i32;

        let width = config.width + 2 * pad_width;
        let height = config.height + 2 * pad_height;
        let dim = (width as usize, height as usize);

        let obs_spec = ObservationSpec::new(config.view_width, view_height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        Self {
            config: config.clone(),
            state: SurvivalState {
                rng: SmallRng::seed_from_u64(0),
                agents: Vec::with_capacity(config.num_agents),
                agent_order: (0..config.num_agents).collect(),
                time: 0,
                base_map: Array2::from_elem(dim, SurvivalObs::TileEmpty),
                map: Array2::from_elem(dim, SurvivalObs::TileEmpty),
                fires: Vec::new(),
                laid_torches: Vec::new(),
                regrowing: VecDeque::new(),
                lit: Array2::from_elem(dim, false),
                eggs: Vec::new(),
                spiders: Vec::new(),
                free_positions: Vec::new(),
            },
            metrics: SurvivalMetrics::default(),
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

    fn calculate_free_positions(&mut self) {
        self.state.free_positions.clear();
        for x in self.pad_width..self.width - self.pad_width {
            for y in self.pad_height..self.height - self.pad_height {
                let position = Position::new(x, y);
                if self.state.map[position.idx()].is_floor() {
                    self.state.free_positions.push(position);
                }
            }
        }

        self.state.free_positions.shuffle(&mut self.state.rng);
    }

    /// Changes what lies on a cell. A creature standing there stays drawn
    /// over it, as over any item it stands on.
    fn set_ground(&mut self, position: Position, tile: SurvivalObs) {
        self.state.base_map[position.idx()] = tile;
        if !self.state.map[position.idx()].is_creature() {
            self.state.map[position.idx()] = tile;
        }
    }

    /// Draws a creature on a cell, over whatever ground is there.
    fn place_creature(&mut self, position: Position, tile: SurvivalObs) {
        debug_assert!(tile.is_creature());
        self.state.map[position.idx()] = tile;
    }

    /// Takes whatever creature is on a cell off it, showing the ground
    /// under it again.
    fn remove_creature(&mut self, position: Position) {
        self.state.map[position.idx()] = self.state.base_map[position.idx()];
    }

    /// The world's own turn, after the agents': winter sets in if it is
    /// due, fires burn down and picked bushes fruit again, the light catches
    /// up with the fires lit and gone out this step and the torches carried,
    /// held torches burn down a step for the light they gave, and the
    /// spiders move by it.
    fn tick_world(&mut self) {
        self.tick_season(self.state.time + 1);
        self.burn_fires();
        self.regrow_bushes();
        self.light_up();
        self.burn_torches();
        self.tick_spiders();
    }
}

impl Environment for Survival {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.state.rng = SmallRng::seed_from_u64(seed);
        self.state.time = 0;
        // the turn order is shuffled in place every step, so start it afresh
        // for the episode to follow from the seed alone
        self.state.agent_order.sort_unstable();

        self.state.base_map.fill(SurvivalObs::TileWall);
        self.generate_map();

        // Base map finished
        self.state.map.assign(&self.state.base_map);
        self.state.fires.clear();
        self.state.laid_torches.clear();
        self.state.regrowing.clear();
        self.state.lit.fill(false);
        self.state.spiders.clear();
        self.tick_season(0);

        self.state.agents.clear();
        self.calculate_free_positions();
        // out of each other's sight, where the map has room for it
        let apart = self.pad_width.max(self.pad_height) + 1;
        let spawns = spread_out(&self.state.free_positions, self.num_agents(), apart);
        assert_eq!(
            spawns.len(),
            self.num_agents(),
            "no open ground left to spawn on"
        );
        for (agent_id, position) in spawns.into_iter().enumerate() {
            self.spawn(agent_id, position);
        }

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.state.agent_order.shuffle(&mut self.state.rng);
        for i in 0..self.state.agent_order.len() {
            let agent_id = self.state.agent_order[i];
            timestep.last_action[agent_id] = actions[agent_id];
            self.act(agent_id, SurvivalAction::from_id(actions[agent_id]));
        }

        self.tick_world();

        let mut dead = Vec::new();
        for agent_id in 0..self.num_agents() {
            if self.tick_stats(agent_id) {
                dead.push(agent_id);
            }
            // judged on the stats it ends the step with, before a respawn
            let reward = self.state.agents[agent_id].reward(&self.config);
            timestep.reward[agent_id] = reward;
            self.metrics.reward += f64::from(reward);
        }
        for &agent_id in &dead {
            self.kill(agent_id);
        }
        if !dead.is_empty() {
            self.calculate_free_positions();
            for &agent_id in &dead {
                let position = self
                    .state
                    .free_positions
                    .pop()
                    .expect("no open ground left to respawn on");
                self.spawn(agent_id, position);
            }
        }

        self.state.time += 1;
        self.encode_observations(timestep);
        for &agent_id in &dead {
            timestep.terminated[agent_id] = true;
        }
        self.encode_action_mask(timestep);
    }

    fn consume_metrics(&mut self) -> serde_json::Value {
        std::mem::take(&mut self.metrics).per_agent(self.num_agents())
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
