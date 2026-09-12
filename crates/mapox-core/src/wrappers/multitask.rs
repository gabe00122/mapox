use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::env::Environment;
use crate::make::{MultiEnvSpec, make};
use crate::render::env::{GridRenderSettings, GridRenderState};
use crate::spec::{ActionSpec, ObservationSpec};
use crate::timestep::TimeStepMut;
use crate::vocab::{VocabId, Vocabulary};
use crate::wrappers::task_id_wrapper::TaskIdWrapper;
use crate::wrappers::vocab_wrapper::VocabWrapper;

pub struct MultitaskWrapper {
    lens: Vec<usize>,
    offsets: Vec<usize>, // offsets agents by env group
    envs: Vec<VocabWrapper>,
    obs_vocab: Vocabulary,
    action_vocab: Vocabulary,
    num_agents: usize,
    num_tasks: usize,
    task_offsets: Vec<usize>, // offsets of envs by task group
    enjoy_mode: Option<usize>,
}

impl MultitaskWrapper {
    pub fn new(specs: &[MultiEnvSpec], length: usize) -> Result<Self, String> {
        let num_tasks = specs.len(); // this assumes each task has only one sub task, to support arbitrarily nested subtasks we need to gather the num_tasks from actual child task instances
        let mut task_offsets: Vec<usize> = Vec::new();
        let mut envs: Vec<Box<dyn Environment>> = Vec::new();

        for (task_id, spec) in specs.iter().enumerate() {
            task_offsets.push(envs.len());
            for _ in 0..spec.num {
                envs.push(Box::new(TaskIdWrapper::new(
                    make(&spec.env, length)?,
                    task_id as i32,
                )));
            }
        }

        let mut obs_vocab = Vocabulary::new();
        let mut action_vocab = Vocabulary::new();

        for env in envs.iter() {
            obs_vocab.extend_with(env.obs_vocab());
            action_vocab.extend_with(env.action_vocab());
        }

        let lens: Vec<usize> = envs.iter().map(|env| env.num_agents()).collect();
        let mut offsets: Vec<usize> = Vec::with_capacity(lens.len());
        let mut s = 0;
        for len in &lens {
            offsets.push(s);
            s += len;
        }
        let num_agents = lens.iter().sum();

        let wrappers = envs
            .into_iter()
            .map(|env| VocabWrapper::new(env, &action_vocab, &obs_vocab))
            .collect();

        Ok(Self {
            envs: wrappers,
            lens,
            offsets,
            num_agents,
            num_tasks,
            obs_vocab,
            action_vocab,
            task_offsets,
            enjoy_mode: None,
        })
    }
}

impl Environment for MultitaskWrapper {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        if let Some(env_idx) = self.enjoy_mode {
            self.envs[env_idx].reset(seed, timestep);
            return;
        }

        self.envs
            .par_iter_mut()
            .zip(timestep.partition_mut(&self.lens))
            .enumerate()
            .for_each(|(i, (env, mut timestep))| {
                let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
                let seed = rng.next_u64();
                env.reset(seed, &mut timestep);
            });
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        if let Some(env_idx) = self.enjoy_mode {
            self.envs[env_idx].step(actions, timestep);
            return;
        }

        let (envs, offsets) = (&mut self.envs, &self.offsets);

        envs.par_iter_mut()
            .zip(offsets)
            .zip(timestep.partition_mut(&self.lens).par_iter_mut())
            .for_each(|((env, &offset), timestep)| {
                let actions = &actions[offset..offset + env.num_agents()];
                env.step(actions, timestep);
            });
    }

    fn observation_spec(&self) -> ObservationSpec {
        // TODO: We need to assert the width and height are the same for all envs
        let ObservationSpec { width, height, .. } = self.envs[0].observation_spec();

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
        if let Some(idx) = self.enjoy_mode {
            self.envs[idx].num_agents()
        } else {
            self.num_agents
        }
    }

    fn obs_vocab(&self) -> &Vocabulary {
        &self.obs_vocab
    }

    fn action_vocab(&self) -> &Vocabulary {
        &self.action_vocab
    }

    fn get_render_settings(&self) -> GridRenderSettings {
        let idx = self.enjoy_mode.unwrap_or(0);
        self.envs[idx].get_render_settings()
    }

    fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
        let idx = self.enjoy_mode.unwrap_or(0);
        self.envs[idx].render_state_into(grid_render_state);
    }

    fn num_tasks(&self) -> usize {
        self.num_tasks
    }

    fn set_enjoy_mode(&mut self, task_num: Option<usize>) {
        if let Some(idx) = self.enjoy_mode {
            self.envs[idx].set_enjoy_mode(None);
        }
        self.enjoy_mode = task_num.map(|tm| self.task_offsets[tm]);
        if let Some(idx) = self.enjoy_mode {
            // The multitask wrapper currently assumes it's child tasks have no child tasks of their own, this could change in the future
            self.envs[idx].set_enjoy_mode(Some(0));
        }
    }
}
