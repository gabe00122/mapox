// use rand::{RngExt, SeedableRng, rngs::SmallRng};
// use rayon::prelude::*;

// use crate::env::Environment;
// use crate::render::env::{GridRenderSettings, GridRenderState};
// use crate::spec::{ActionSpec, ObservationSpec};
// use crate::timestep::TimeStepMut;
// use crate::vocab::{VocabId, Vocabulary};
// use crate::wrappers::task_id_wrapper::TaskIdWrapper;

// struct EnvironmentInfo {
//     action_vocab: Vocabulary,
//     obs_vocab: Vocabulary,
//     env: Box<dyn Environment>,
// }

// pub struct MultitaskWrapper {
//     envs: Vec<EnvironmentInfo>,
// }

// impl MultitaskWrapper {
//     pub fn new(envs: Vec<Box<dyn Environment>>) -> Self {
//         let wrappers = envs.iter().
//         Self { envs }
//     }
// }

// impl Environment for MultitaskWrapper {
//     fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {

//     }

//     fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {

//     }

//     fn observation_spec(&self) -> ObservationSpec {
//         self.envs[0].observation_spec()
//     }

//     fn action_spec(&self) -> ActionSpec {
//         self.envs[0].action_spec()
//     }

//     fn num_agents(&self) -> usize {
//         self.envs.iter().map(|env| env.num_agents()).sum()
//     }

//     fn obs_vocab(&self) -> &Vocabulary {
//         self.envs[0].obs_vocab()
//     }

//     fn action_vocab(&self) -> &Vocabulary {
//         self.envs[0].action_vocab()
//     }

//     fn get_render_settings(&self) -> GridRenderSettings {
//         self.envs[0].get_render_settings()
//     }

//     fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
//         self.envs[0].render_state_into(grid_render_state);
//     }
// }
