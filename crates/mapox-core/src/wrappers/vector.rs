use ndarray::Axis;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;

use crate::env::Environment;
use crate::render::env::{GridRenderSettings, GridRenderState};
use crate::spec::{ActionSpec, ObservationSpec};
use crate::timestep::TimeStepMut;
use crate::vocab::{VocabId, Vocabulary};

pub struct VectorWrapper {
    envs: Vec<Box<dyn Environment>>,
}

impl VectorWrapper {
    pub fn new(envs: Vec<Box<dyn Environment>>) -> Self {
        Self { envs }
    }
}

/// Splits the flat `(envs * agents, ...)` buffers into one `TimeStepMut` per
/// env so each env can fill its own slice from a rayon worker. The chunks are
/// zipped lazily so no per-step collection is allocated.
fn split_timestep<'a>(
    timestep: &'a mut TimeStepMut<'_>,
    agents_per_env: usize,
) -> impl IndexedParallelIterator<Item = TimeStepMut<'a>> {
    (
        timestep.obs.axis_chunks_iter_mut(Axis(0), agents_per_env),
        timestep.time.axis_chunks_iter_mut(Axis(0), agents_per_env),
        timestep
            .terminated
            .axis_chunks_iter_mut(Axis(0), agents_per_env),
        timestep
            .last_action
            .axis_chunks_iter_mut(Axis(0), agents_per_env),
        timestep
            .reward
            .axis_chunks_iter_mut(Axis(0), agents_per_env),
        timestep
            .action_mask
            .axis_chunks_iter_mut(Axis(0), agents_per_env),
        timestep
            .task_ids
            .axis_chunks_iter_mut(Axis(0), agents_per_env),
    )
        .into_par_iter()
        .map(
            |(obs, time, terminated, last_action, reward, action_mask, task_ids)| TimeStepMut {
                obs,
                time,
                terminated,
                last_action,
                reward,
                action_mask,
                task_ids,
            },
        )
}

impl Environment for VectorWrapper {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let seeds: Vec<u64> = (0..self.envs.len()).map(|_| rng.random()).collect();

        let chunks = split_timestep(timestep, self.envs[0].num_agents());

        self.envs
            .par_iter_mut()
            .zip(seeds)
            .zip(chunks)
            .for_each(|((env, seed), mut timestep)| env.reset(seed, &mut timestep));
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        let agents_per_env = self.envs[0].num_agents();
        let chunks = split_timestep(timestep, agents_per_env);

        self.envs
            .par_iter_mut()
            .zip(actions.par_chunks(agents_per_env))
            .zip(chunks)
            .for_each(|((env, actions), mut timestep)| env.step(actions, &mut timestep));
    }

    fn observation_spec(&self) -> ObservationSpec {
        self.envs[0].observation_spec()
    }

    fn action_spec(&self) -> ActionSpec {
        self.envs[0].action_spec()
    }

    fn num_agents(&self) -> usize {
        self.envs.len() * self.envs[0].num_agents()
    }

    fn obs_vocab(&self) -> &Vocabulary {
        self.envs[0].obs_vocab()
    }

    fn action_vocab(&self) -> &Vocabulary {
        self.envs[0].action_vocab()
    }

    fn get_render_settings(&self) -> GridRenderSettings {
        self.envs[0].get_render_settings()
    }

    fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
        self.envs[0].render_state_into(grid_render_state);
    }
}
