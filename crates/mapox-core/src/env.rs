use crate::render::env::{GridRenderSettings, GridRenderState};
use crate::spec::{ActionSpec, ObservationSpec};
use crate::timestep::TimeStepMut;
use crate::vocab::Vocabulary;

pub trait Environment {
    type EnvState;

    fn init_state(&self) -> Self::EnvState;

    fn reset(&self, state: &mut Self::EnvState, seed: u64, timestep: &mut TimeStepMut);
    fn step(&self, state: &mut Self::EnvState, actions: &[i32], timestep: &mut TimeStepMut);

    fn observation_spec(&self) -> ObservationSpec;
    fn action_spec(&self) -> ActionSpec;

    fn num_agents(&self) -> usize;
    fn obs_vocab(&self) -> &Vocabulary;
    fn action_vocab(&self) -> &Vocabulary;

    fn get_render_settings(&self) -> GridRenderSettings;
    fn render_state_into(&self, state: &Self::EnvState, grid_render_state: &mut GridRenderState);
}
