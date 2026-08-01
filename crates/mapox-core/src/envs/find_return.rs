use crate::{
    env::Environment,
    envs::common::Position,
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_GENERIC, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, TILE_DESTRUCTIBLE_WALL,
        TILE_EMPTY, TILE_FLAG, TILE_WALL,
    },
    timestep::TimeStepMut,
    vocab::Vocabulary,
};

#[derive(Debug, Default, Clone)]
pub struct FindReturnConfig {
    pub num_agents: usize,
    pub num_flags: usize,

    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,

    pub mapgen_threshold: f64,
    pub digging_timeout: i32,
    pub treasure_reward: f64,
}

#[derive(Debug, Default, Clone)]
pub struct FindReturnAgent {
    pub position: Position,
    pub found_reward: bool,
}

#[derive(Debug, Default, Clone)]
pub struct FindReturnState {
    pub agents: Vec<FindReturnAgent>,
    pub time: i32,
    pub map: Vec<u8>,
}

#[derive(Debug, Default, Clone)]
pub struct FindReturn {
    pub config: FindReturnConfig,

    unpadded_width: i32,
    unppaded_height: i32,

    pad_width: i32,
    pad_height: i32,

    width: i32,
    height: i32,

    obs_spec: ObservationSpec,
    action_spec: ActionSpec,

    obs_vocab: Vocabulary,
    action_vocab: Vocabulary,

    obs_tile_empty: usize,
    obs_tile_destructible_wall: usize,
    obs_tile_wall: usize,
    obs_tile_flag: usize,
    obs_agent_generic: usize,
    action_move_up: usize,
    action_move_right: usize,
    action_move_down: usize,
    action_move_left: usize,
}

impl FindReturn {
    fn new(config: &FindReturnConfig) -> Self {
        let mut action_vocab = Vocabulary::new();
        let mut obs_vocab = Vocabulary::new();

        let obs_tile_empty = obs_vocab.add(TILE_EMPTY);
        let obs_tile_destructible_wall = obs_vocab.add(TILE_DESTRUCTIBLE_WALL);
        let obs_tile_wall = obs_vocab.add(TILE_WALL);
        let obs_tile_flag = obs_vocab.add(TILE_FLAG);
        let obs_agent_generic = obs_vocab.add(AGENT_GENERIC);

        let action_move_up = action_vocab.add(MOVE_UP);
        let action_move_right = action_vocab.add(MOVE_RIGHT);
        let action_move_down = action_vocab.add(MOVE_DOWN);
        let action_move_left = action_vocab.add(MOVE_LEFT);

        let unpadded_width = config.width;
        let unppaded_height = config.height;

        let pad_width = config.view_width / 2;
        let pad_height = config.view_height / 2;

        let width = unpadded_width + pad_width;
        let height = unppaded_height + pad_height;

        let obs_spec = ObservationSpec::new(width, height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        Self {
            config: config.clone(),

            unpadded_width,
            unppaded_height,
            pad_width,
            pad_height,
            width,
            height,

            obs_spec,
            action_spec,

            action_vocab,
            obs_vocab,

            action_move_up,
            action_move_right,
            action_move_down,
            action_move_left,
            obs_tile_empty,
            obs_tile_destructible_wall,
            obs_tile_wall,
            obs_tile_flag,
            obs_agent_generic,
        }
    }
}

impl Environment for FindReturn {
    type EnvState = FindReturnState;

    fn init_state(&self) -> FindReturnState {
        FindReturnState::default()
    }

    fn reset(&self, state: &mut Self::EnvState, seed: u64, timestep: &mut TimeStepMut) {
        todo!()
    }

    fn step(&self, state: &mut Self::EnvState, actions: &[i32], timestep: &mut TimeStepMut) {
        todo!()
    }

    fn observation_spec(&self) -> ObservationSpec {
        self.obs_spec
    }

    fn action_spec(&self) -> ActionSpec {
        self.action_spec
    }

    fn num_agents(&self) -> usize {
        1
    }

    fn obs_vocab(&self) -> &Vocabulary {
        &self.obs_vocab
    }

    fn action_vocab(&self) -> &Vocabulary {
        &self.action_vocab
    }

    fn get_render_settings(&self) -> GridRenderSettings {
        todo!()
    }

    fn render_state_into(&self, state: &Self::EnvState, grid_render_state: &mut GridRenderState) {
        todo!()
    }
}
