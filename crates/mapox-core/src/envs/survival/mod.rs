use ndarray::{Array2, s};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use slotmap::{SlotMap, new_key_type};

use crate::{
    env::Environment,
    envs::common::{Position, UI_HEIGHT, fov, stamp::stamp_circle, vocab_enum::VocabEnum},
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_GENERIC, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, NOOP, TILE_DECOR_1, TILE_DECOR_2,
        TILE_DECOR_3, TILE_DECOR_4, TILE_DESTRUCTIBLE_WALL, TILE_EMPTY, TILE_MASK, TILE_UI,
        TILE_WALL, TILE_WATER,
    },
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
    vocab_enum,
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
    render_map: Array2<SurvivalObs>, // the render target for the agent views
    lighting: Array2<bool>,          // true for a lit tile and false for a dark tile
    free_positions: Vec<Position>,   // these are used to calculate spawn positions
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

// find_return's terrain tiles; the room only uses walls and floor so far
vocab_enum!(SurvivalObs {
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
    AgentGeneric => AGENT_GENERIC,
});

impl SurvivalObs {
    fn move_blocked(self) -> bool {
        use SurvivalObs::*;
        matches!(
            self,
            TileWall | TileDestructibleWall | TileWater | AgentGeneric
        )
    }

    fn spawnable(self) -> bool {
        use SurvivalObs::*;
        matches!(
            self,
            TileEmpty | TileDecor1 | TileDecor2 | TileDecor3 | TileDecor4
        )
    }

    /// Water is the one blocking tile an agent can see straight over.
    fn opaque(self) -> bool {
        use SurvivalObs::*;
        matches!(self, TileWall | TileDestructibleWall)
    }
}

vocab_enum!(SurvivalAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
    Noop => NOOP,
});

impl SurvivalAction {
    fn direction(self) -> Position {
        use SurvivalAction::*;
        match self {
            MoveUp => Position::new(0, 1),
            MoveRight => Position::new(1, 0),
            MoveDown => Position::new(0, -1),
            MoveLeft => Position::new(-1, 0),
            Noop => Position::new(0, 0),
        }
    }

    fn is_move(self) -> bool {
        !matches!(self, SurvivalAction::Noop)
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
                free_positions: Vec::new(),
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

    fn calculate_free_positions(&mut self) {
        self.state.free_positions.clear();
        for x in self.pad_width..self.width - self.pad_width {
            for y in self.pad_height..self.height - self.pad_height {
                let position = Position::new(x, y);
                if self.state.tiles[position.idx()].spawnable() {
                    self.state.free_positions.push(position);
                }
            }
        }

        self.state.free_positions.shuffle(&mut self.state.rngs);
    }

    /// Rebuild the render map from scratch: the tile layer, then the objects layer on top
    /// (the upper slot wins over the lower one).
    fn render(&mut self) {
        let state = &mut self.state;
        state.render_map.assign(&state.tiles);
        for (dst, cell) in state.render_map.iter_mut().zip(&state.objects) {
            if let Some(id) = cell[Slot::Upper as usize].or(cell[Slot::Lower as usize]) {
                *dst = state.entities[id].tile;
            }
        }

        // render lights
        state.lighting.fill(false);
        stamp_circle(&mut state.lighting, Position::new(10, 10), 5, true);

        state
            .render_map
            .zip_mut_with(&state.lighting, |target, lit| {
                if !lit {
                    *target = SurvivalObs::Mask;
                }
            });
    }

    fn encode_observations(&self, timestep: &mut TimeStepMut) {
        let fov_height = self.config.view_height as usize;

        for agent_id in 0..self.num_agents() {
            let agent = self.state.agent(agent_id);
            // wall padding keeps the view window inside the map
            let mut view = timestep.obs.slice_mut(s![agent_id, .., ..fov_height, 0]);
            fov::encode_visible(
                &self.state.render_map,
                agent.position,
                &mut view,
                SurvivalObs::Mask,
                |tile| tile.opaque(),
            );

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
            self.state.render_map = Array2::from_elem(dim, SurvivalObs::TileEmpty);
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
        self.calculate_free_positions();

        for _ in 0..self.num_agents() {
            let position = self.state.free_positions.pop().unwrap();
            let id = self.state.spawn(Entity {
                position,
                slot: Slot::Upper,
                tile: SurvivalObs::AgentGeneric,
            });
            self.state.agents.push(id);
        }

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.render();
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
        self.render();
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
