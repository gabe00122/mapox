use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::env::Environment;
use crate::render::env::{GridRenderSettings, GridRenderState};
use crate::spec::{ActionSpec, ObservationSpec};
use crate::timestep::TimeStepMut;
use crate::vocab::{VocabId, Vocabulary};
use crate::wrappers::vocab_wrapper::VocabWrapper;

struct EnvironmentInfo {
    offset: usize,
    env: VocabWrapper,
}

pub struct MultitaskWrapper {
    lens: Vec<usize>,
    envs: Vec<EnvironmentInfo>,
    obs_vocab: Vocabulary,
    action_vocab: Vocabulary,
    num_agents: usize,
}

impl MultitaskWrapper {
    pub fn new(envs: Vec<Box<dyn Environment>>) -> Self {
        let mut obs_vocab = Vocabulary::new();
        let mut action_vocab = Vocabulary::new();

        for env in envs.iter() {
            obs_vocab.extend_with(env.obs_vocab());
            action_vocab.extend_with(env.action_vocab());
        }

        let lens: Vec<usize> = envs.iter().map(|env| env.num_agents()).collect();
        let num_agents = envs.iter().map(|env| env.num_agents()).sum();

        let wrappers = envs
            .into_iter()
            .scan(0, |offset, env| {
                let num_agents = env.num_agents();
                let info = EnvironmentInfo {
                    offset: *offset,
                    env: VocabWrapper::new(env, &action_vocab, &obs_vocab),
                };
                *offset += num_agents;
                Some(info)
            })
            .collect();

        Self {
            envs: wrappers,
            lens,
            num_agents,
            obs_vocab,
            action_vocab,
        }
    }
}

impl Environment for MultitaskWrapper {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.envs
            .par_iter_mut()
            .zip(timestep.partition_mut(&self.lens))
            .enumerate()
            .for_each(|(i, (env, mut timestep))| {
                let mut rng = SmallRng::seed_from_u64(seed + (i as u64));
                let seed = rng.next_u64();
                env.env.reset(seed, &mut timestep);
            });
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.envs
            .par_iter_mut()
            .zip(timestep.partition_mut(&self.lens).par_iter_mut())
            .for_each(|(env, timestep)| {
                let actions = &actions[env.offset..env.offset + env.env.num_agents()];
                env.env.step(actions, timestep);
            });
    }

    fn observation_spec(&self) -> ObservationSpec {
        // TODO: We need to assert the width and height are the same for all envs
        let ObservationSpec { width, height, .. } = self.envs[0].env.observation_spec();

        ObservationSpec {
            width,
            height,
            num_types: self.obs_vocab.len(),
        }
    }

    fn action_spec(&self) -> ActionSpec {
        ActionSpec {
            num_actions: self.action_vocab.len(),
        }
    }

    fn num_agents(&self) -> usize {
        self.num_agents
    }

    fn obs_vocab(&self) -> &Vocabulary {
        &self.obs_vocab
    }

    fn action_vocab(&self) -> &Vocabulary {
        &self.action_vocab
    }

    fn get_render_settings(&self) -> GridRenderSettings {
        self.envs[0].env.get_render_settings()
    }

    fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
        self.envs[0].env.render_state_into(grid_render_state);
    }
}
