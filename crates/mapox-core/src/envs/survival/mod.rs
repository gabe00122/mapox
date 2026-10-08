pub mod action;
pub mod obs;

use ndarray::{Array2, s};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use slotmap::{SlotMap, new_key_type};

use crate::{
    env::Environment,
    envs::{
        common::{
            Position, UI_HEIGHT,
            fov::{self, ViewTile, window},
            vocab_enum::VocabEnum,
        },
        survival::{action::SurvivalAction, obs::SurvivalObs},
    },
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SurvivalConfig {
    pub num_agents: usize,

    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,
}

impl Default for SurvivalConfig {
    fn default() -> Self {
        Self {
            num_agents: 8,
            width: 40,
            height: 40,
            view_width: 15,
            view_height: 15,
        }
    }
}

new_key_type! {
    /// Generational key: a removed entity's id stops resolving instead of aliasing a reused slot.
    struct EntityId;
}

/// Each tile holds up to one entity per slot, so e.g. an agent can stand on top of something.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    Lower = 0,
    Upper = 1,
}

type ObjectCell = [Option<EntityId>; 2];

/// Anything that sits on the objects layer. Only agents for now.
#[derive(Debug, Clone)]
struct Entity {
    position: Position,
    slot: Slot,
    tile: SurvivalObs, // what the entity draws as
}

#[derive(Debug, Clone)]
struct SurvivalState {
    rngs: SmallRng,
    entities: SlotMap<EntityId, Entity>, // owns every entity; the layers refer into it by id
    agents: Vec<EntityId>,               // agent_id -> entity
    agent_order: Vec<usize>,             // agent turn order
    time: usize,

    tiles: Array2<SurvivalObs>,      // terrain layer
    objects: Array2<ObjectCell>,     // entity layer on top of the terrain
    lighting: Array2<bool>,          // true for a lit tile and false for a dark tile
    render_map: Array2<SurvivalObs>, // the render target for the agent views
}

impl SurvivalState {
    fn agent(&self, agent_id: usize) -> &Entity {
        &self.entities[self.agents[agent_id]]
    }

    fn spawn(&mut self, entity: Entity) -> EntityId {
        let (position, slot) = (entity.position, entity.slot);
        let id = self.entities.insert(entity);
        self.objects[position.idx()][slot as usize] = Some(id);
        id
    }

    /// Move an entity, keeping the objects layer in sync.
    fn move_entity(&mut self, id: EntityId, target: Position) {
        let entity = &mut self.entities[id];
        let slot = entity.slot as usize;
        self.objects[entity.position.idx()][slot] = None;
        entity.position = target;
        self.objects[target.idx()][slot] = Some(id);
    }
}

#[derive(Debug, Clone)]
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
                entities: SlotMap::with_capacity_and_key(config.num_agents),
                tiles: Array2::from_elem(map_dim, SurvivalObs::TileEmpty),
                objects: Array2::from_elem(map_dim, [None; 2]),
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
        let state = &mut self.state;
        state.render_map.assign(&state.tiles);
        for (dst, cell) in state.render_map.iter_mut().zip(&state.objects) {
            if let Some(id) = cell[Slot::Upper as usize].or(cell[Slot::Lower as usize]) {
                *dst = state.entities[id].tile;
            }
        }

        // render lights, this should be a seperate function
        state.lighting.fill(false);
        let (lighting, render_map) = (&mut state.lighting, &state.render_map);
        let light = Position::new(10, 10);
        fov::shadowcast(light, lighting.dim(), |cell| {
            if !fov::within(cell - light, 5) {
                return true; // past the light's reach, and so is all behind it
            }
            lighting[cell.idx()] = true;
            render_map[cell.idx()].opaque()
        });
    }

    fn encode_observations(&self, timestep: &mut TimeStepMut) {
        let fov_height = self.config.view_height as usize;
        for agent_id in 0..self.num_agents() {
            let agent = self.state.agent(agent_id);
            // wall padding keeps the view window inside the map
            let mut view = timestep.obs.slice_mut(s![agent_id, .., ..fov_height, 0]);
            fov::observe(&self.state.render_map, agent.position, &mut view);

            let lightning_window = window(&self.state.lighting, agent.position, view.dim());
            view.zip_mut_with(&lightning_window, |target, light| {
                if !light {
                    *target = SurvivalObs::Mask.into();
                }
            });

            let mut ui = timestep.obs.slice_mut(s![agent_id, .., fov_height.., 0]);
            ui.fill(SurvivalObs::UI as VocabId);
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length);
        timestep.task_ids.fill(0);
    }

    fn encode_action_mask(&self, timestep: &mut TimeStepMut) {
        for agent_id in 0..self.num_agents() {
            let agent = self.state.agent(agent_id);
            let mut mask = timestep.action_mask.row_mut(agent_id);

            for &action in SurvivalAction::TABLE {
                mask[action as usize] = !action.is_move()
                    || !self.state.render_map[(agent.position + action.direction()).idx()]
                        .move_blocked();
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
            self.state.objects = Array2::from_elem(dim, [None; 2]);
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

        self.state.objects.fill([None; 2]);
        self.state.entities.clear();
        self.state.agents.clear();

        for i in 0..self.num_agents() {
            let id = self.state.spawn(Entity {
                position: Position::new(20, 20 + i as i32),
                slot: Slot::Upper,
                tile: SurvivalObs::AgentGeneric,
            });
            self.state.agents.push(id);
        }

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.prepare_render();
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.state.agent_order.shuffle(&mut self.state.rngs);

        for turn in 0..self.num_agents() {
            let agent_id = self.state.agent_order[turn];
            timestep.last_action[agent_id] = actions[agent_id];
            timestep.reward[agent_id] = 0.0;

            let action = SurvivalAction::from_id(actions[agent_id]);
            if action.is_move() {
                // the render map is stale mid-step, so check the layers directly
                let id = self.state.agents[agent_id];
                let target = self.state.entities[id].position + action.direction();
                let slot = self.state.entities[id].slot as usize;
                let occupied = self.state.objects[target.idx()][slot].is_some();
                if !occupied && !self.state.tiles[target.idx()].move_blocked() {
                    self.state.move_entity(id, target);
                }
            }
        }

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

        grid_render_state.agent_positions.clear();
        for agent_id in 0..self.num_agents() {
            let local_pos = self.state.agent(agent_id).position
                - Position::new(self.pad_width, self.pad_height);
            grid_render_state.agent_positions.push(local_pos);
        }
    }

    fn num_tasks(&self) -> usize {
        1
    }
}
