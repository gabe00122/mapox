pub mod action;
pub mod config;
pub mod obs;

mod fire;
mod items;
mod lightning;
mod needs;
mod prototypes;
mod refine;
mod survivor;
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
            prototypes::Prototype,
            world::{EntityCell, Slot},
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
    agents: Vec<Entity>,     // agent_id -> entity
    agent_order: Vec<usize>, // agent turn order
    time: usize,

    tiles: Array2<SurvivalObs>,        // terrain layer
    spatial_index: Array2<EntityCell>, // fast special entity lookup index
    lighting: Array2<bool>,            // true for a lit tile and false for a dark tile
    render_map: Array2<SurvivalObs>,   // the render target for the agent views
}

pub struct Survival {
    pub config: SurvivalConfig,
    state: SurvivalState,

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
            if slot == Slot::Upper {
                render_map[position.idx()] = tile;
            }
        }

        // render lights, this should be a seperate function
        self.prepare_lights();
    }

    fn encode_action_mask(&self, timestep: &mut TimeStepMut) {
        for agent_id in 0..self.num_agents() {
            let mut mask = timestep.action_mask.row_mut(agent_id);

            mask.fill(false);
            for &action in SurvivalAction::TABLE {
                mask[action as usize] = true;
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

        // an empty room: wall padding around bare floor
        self.state.tiles.fill(SurvivalObs::TileWall);
        self.state
            .tiles
            .slice_mut(s![
                self.pad_width as usize..(self.width - self.pad_width) as usize,
                self.pad_height as usize..(self.height - self.pad_height) as usize,
            ])
            .fill(SurvivalObs::TileEmpty);

        for i in 0..self.num_agents() {
            // reset this so the seed is the source of truth for agent order
            self.state.agent_order[i] = i;
        }

        self.state.spatial_index.fill([None; 2]);
        self.state.world.clear();
        self.state.agents.clear();

        for i in 0..self.num_agents() {
            let id = self.state.spawn_prototype(
                Prototype::Survivor { agent_index: i },
                Position::new(10, 20 + i as i32),
            );
            self.state.agents.push(id);
        }

        // swap items for testing
        self.state
            .spawn_prototype(Prototype::Fire, Position::new(18, 20));

        let items = [
            ItemType::Wood,
            ItemType::Rock,
            ItemType::CutGrass,
            ItemType::Stick,
            ItemType::StoneAxe,
        ];

        for (i, item) in items.into_iter().enumerate() {
            self.state
                .spawn_prototype(Prototype::Item(item), Position::new(20, 20 + i as i32));
        }
        //

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);

        self.state.tick_direction_tiles();
        self.prepare_render();
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.state.tick_survivors(actions, timestep);
        self.state.tick_starvation();
        self.state.tick_hunger();
        self.state.tick_fire();
        self.state.tick_death();

        self.state.tick_direction_tiles();

        self.state.time += 1;
        self.prepare_render();
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn consume_metrics(&mut self) -> serde_json::Value {
        serde_json::json!({})
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
