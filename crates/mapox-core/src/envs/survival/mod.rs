pub mod action;
pub mod config;
pub mod obs;

mod fire;
mod items;
mod lightning;
mod needs;
mod plants;
mod portal;
mod prototypes;
mod refine;
mod survivor;
mod terrain;
mod time;
mod world;

use hecs::{Entity, World};
use ndarray::{Array2, s};
use rand::{SeedableRng, rngs::SmallRng};

use crate::{
    env::Environment,
    envs::{
        common::{Position, UI_HEIGHT, vocab_enum::VocabEnum},
        survival::{
            action::SurvivalAction,
            config::SurvivalConfig,
            items::ItemType,
            obs::{SurvivalObs, UI_WIDTH},
            world::{Agent, EntityCell, Slot},
        },
    },
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
};

struct SurvivalState {
    rngs: SmallRng,
    world: World,
    agents: Vec<Entity>,     // agent_id -> entity, stale while the agent is dead
    agent_order: Vec<usize>, // agent turn order
    time: usize,
    portal: Position, // where agents spawn and respawn

    tiles: Array2<SurvivalObs>,        // terrain layer
    spatial_index: Array2<EntityCell>, // fast special entity lookup index
    lighting: Array2<bool>,            // true for a lit tile and false for a dark tile
    render_map: Array2<SurvivalObs>,   // the render target for the agent views,

    // day night cycle
    dusk: Option<i32>, // when dusk is in play this represents the agent vision radius
}

#[derive(Debug, Clone, Default)]
struct SurvivalMetrics {
    // summed over the episode, where every life ends by the last step
    reward: f64,
    turns: f64,
    lives: f64,
}

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

        let action_vocab = SurvivalAction::vocab();
        let obs_vocab = SurvivalObs::vocab();

        let pad_width = config.view_width / 2;
        let pad_height = config.view_height / 2;

        let view_height = config.view_height + UI_HEIGHT as i32;

        let width = config.width + 2 * pad_width;
        let height = config.height + 2 * pad_height;

        let obs_spec = ObservationSpec::new(config.view_width, view_height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        let map_dim = (width as usize, height as usize);

        Self {
            config: config.clone(),
            state: SurvivalState {
                agents: Vec::with_capacity(config.num_agents),
                agent_order: (0..config.num_agents).collect(),
                world: World::new(),
                tiles: Array2::from_elem(map_dim, SurvivalObs::TileEmpty),
                spatial_index: Array2::from_elem(map_dim, [None; 2]),
                lighting: Array2::from_elem(map_dim, true),
                render_map: Array2::from_elem(map_dim, SurvivalObs::TileEmpty),
                rngs: SmallRng::seed_from_u64(0),
                time: 0,
                portal: Position::new(0, 0),

                dusk: None,
            },
            metrics: SurvivalMetrics::default(),
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

    fn prepare_render(&mut self) {
        let (world, render_map, tiles) = (
            &mut self.state.world,
            &mut self.state.render_map,
            &self.state.tiles,
        );
        render_map.assign(tiles);

        for (position, &slot, &tile) in world.query_mut::<(&Position, &Slot, &SurvivalObs)>() {
            if slot == Slot::Lower {
                render_map[position.idx()] = tile;
            }
        }

        for (position, &slot, &tile) in world.query_mut::<(&Position, &Slot, &SurvivalObs)>() {
            if slot == Slot::Upper || slot == Slot::Both {
                render_map[position.idx()] = tile;
            }
        }

        // render lights, this should be a separate function
        self.prepare_lights();
    }

    /// Adds this step to the totals and counts the lives that end here, by
    /// death or by the episode running out.
    fn record_lives(&mut self, timestep: &TimeStepMut) {
        for agent in self.state.world.query_mut::<&Agent>() {
            let i = agent.agent_index;
            self.metrics.reward += f64::from(timestep.reward[i]);
            self.metrics.turns += 1.0;
            if timestep.terminated[i] {
                self.metrics.lives += 1.0;
            }
        }
    }

    /// Masks out the item actions that would do nothing. Moves always turn
    /// the agent, so they and the noop stay legal and every row has one.
    fn encode_action_mask(&mut self, timestep: &mut TimeStepMut) {
        let state = &mut self.state;
        for agent_id in 0..state.agents.len() {
            let id = state.agents[agent_id];
            let mut mask = timestep.action_mask.row_mut(agent_id);

            mask.fill(false);
            for &action in SurvivalAction::TABLE {
                mask[action as usize] = match action {
                    SurvivalAction::MoveUp
                    | SurvivalAction::MoveRight
                    | SurvivalAction::MoveDown
                    | SurvivalAction::MoveLeft
                    | SurvivalAction::Noop => true,
                    SurvivalAction::ItemTake => state.plan_take(id).is_some(),
                    SurvivalAction::ItemPut => state.plan_put(id).is_some(),
                    SurvivalAction::ItemSwap => state.can_swap(id),
                    SurvivalAction::ItemCombine => state.plan_craft(id).is_some(),
                    SurvivalAction::ItemUse => state.plan_use(id).is_some(),
                    SurvivalAction::ItemRefine => state.plan_refine(id).is_some(),
                };
            }
        }
    }
}

impl Environment for Survival {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.state.rngs = SmallRng::seed_from_u64(seed);

        self.state.time = 0;

        let dim = (self.width as usize, self.height as usize);
        if self.state.tiles.dim() != dim {
            self.state.tiles = Array2::from_elem(dim, SurvivalObs::TileEmpty);
            self.state.spatial_index = Array2::from_elem(dim, [None; 2]);
        }

        for i in 0..self.num_agents() {
            // reset this so the seed is the source of truth for agent order
            self.state.agent_order[i] = i;
        }

        self.state.spatial_index.fill([None; 2]);
        self.state.world.clear();
        // every slot starts dead, so the respawn brings them all in
        self.state.agents.clear();
        self.state
            .agents
            .resize(self.num_agents(), Entity::DANGLING);

        // wall padding around the generated interior
        self.state.tiles.fill(SurvivalObs::TileWall);
        self.generate_map();
        self.place_portal();
        self.state.tick_respawn();

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);

        self.state.tick_direction_tiles();
        self.prepare_render();
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.state.reset_health_changed();
        self.state.reset_hunger_changed();

        self.state.tick_survivors(actions, timestep);
        self.state.tick_starvation();
        self.state.tick_hunger();

        // buildings
        self.state.tick_damage_to_above();
        self.state.tick_fueled();
        self.tick_reactivate_after();

        //update obs
        self.state.tick_direction_tiles();
        self.state.tick_active_obs();
        //

        self.tick_time(); // day night cycle
        self.prepare_render();
        self.state.calculae_survival_reward();
        self.encode_observations(timestep);
        self.record_lives(timestep);
        self.state.tick_death();
        self.state.tick_respawn();
        // the next action goes to the respawned entity, so mask against it
        self.encode_action_mask(timestep);
    }

    fn consume_metrics(&mut self) -> serde_json::Value {
        let metrics = std::mem::take(&mut self.metrics);
        let lives = metrics.lives.max(1.0);
        serde_json::json!({
            "reward": metrics.reward / lives,
            "turns_survived": metrics.turns / lives,
        })
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
        let interior = self.state.render_map.slice(s![
            self.pad_width as usize..(self.width - self.pad_width) as usize,
            self.pad_height as usize..(self.height - self.pad_height) as usize,
        ]);
        tilemap.zip_mut_with(&interior, |dst, &tile| *dst = tile.into());

        let pad = Position::new(self.pad_width, self.pad_height);
        grid_render_state.agent_positions.clear();
        grid_render_state
            .agent_positions
            .extend(self.state.agents.iter().map(|&id| {
                self.state
                    .world
                    .get::<&Position>(id)
                    .map(|p| *p - pad)
                    .unwrap_or(Position::new(-1, -1))
            }));
    }

    fn num_tasks(&self) -> usize {
        1
    }
}
