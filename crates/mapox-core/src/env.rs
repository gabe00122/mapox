use crate::render::env::{GridRenderSettings, GridRenderState};
use crate::spec::{ActionSpec, ObservationSpec};
use crate::timestep::TimeStepMut;
use crate::vocab::{VocabId, Vocabulary};

pub trait Environment: Send + Sync {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut);
    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut);

    fn observation_spec(&self) -> ObservationSpec;
    fn action_spec(&self) -> ActionSpec;

    fn num_agents(&self) -> usize;
    fn obs_vocab(&self) -> &Vocabulary;
    fn action_vocab(&self) -> &Vocabulary;

    fn get_render_settings(&self) -> GridRenderSettings;
    fn render_state_into(&self, grid_render_state: &mut GridRenderState);

    fn num_tasks(&self) -> usize;
}
