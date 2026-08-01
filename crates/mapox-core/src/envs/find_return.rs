use crate::{
    env::Environment,
    envs::common::Position,
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
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
    pub state: Option<FindReturnState>,
}

impl FindReturn {
    fn new() {}
}

impl Environment for FindReturn {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        todo!()
    }
    fn step(&mut self, actions: &[i32], timestep: &mut TimeStepMut) {
        todo!()
    }

    fn observation_spec(&self) -> ObservationSpec {
        todo!()
    }
    fn action_spec(&self) -> ActionSpec {
        todo!()
    }

    fn num_actions(&self) -> usize {
        todo!()
    }
    fn obs_vocab(&self) -> &Vocabulary {
        todo!()
    }
    fn action_vocab(&self) -> &Vocabulary {
        todo!()
    }

    fn get_render_settings(&self) -> GridRenderSettings {
        todo!()
    }
    fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
        todo!()
    }
}
