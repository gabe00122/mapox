use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::env::Environment;
use crate::error::MapoxError;
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
    task_names: Vec<String>,
    enjoy_mode: Option<usize>,
}

impl MultitaskWrapper {
    pub fn new(specs: &[MultiEnvSpec], length: usize) -> Result<Self, MapoxError> {
        let num_tasks = specs.len(); // this assumes each task has only one sub task, to support arbitrarily nested subtasks we need to gather the num_tasks from actual child task instances
        let mut task_offsets: Vec<usize> = Vec::new();
        let mut envs: Vec<Box<dyn Environment>> = Vec::new();
        let mut task_names = Vec::with_capacity(num_tasks);

        if specs.is_empty() {
            return Err(MapoxError::InvalidConfig {
                reason: "multitask env requires at least one task".into(),
            });
        }
        for spec in specs {
            if spec.num == 0 {
                return Err(MapoxError::InvalidConfig {
                    reason: format!("task {:?} must contain at least one env", spec.name),
                });
            }
            if task_names.contains(&spec.name) {
                return Err(MapoxError::InvalidConfig {
                    reason: format!("duplicate task name {:?}", spec.name),
                });
            }
            task_names.push(spec.name.clone());
        }

        for (task_id, spec) in specs.iter().enumerate() {
            task_offsets.push(envs.len());
            for _ in 0..spec.num {
                envs.push(Box::new(TaskIdWrapper::new(
                    make(&spec.env, length)?,
                    task_id as i32,
                )));
            }
        }

        // The obs vocab is unioned across tasks, but the spatial shape must be
        // identical: downstream buffers are allocated once from one spec.
        let expected = envs[0].observation_spec();
        for (task_id, spec) in specs.iter().enumerate() {
            let start = task_offsets[task_id];
            for env in &envs[start..start + spec.num] {
                let observed = env.observation_spec();
                if observed.width != expected.width || observed.height != expected.height {
                    return Err(MapoxError::ObservationShapeMismatch {
                        task: spec.name.clone(),
                        expected_task: specs[0].name.clone(),
                        width: observed.width,
                        height: observed.height,
                        expected_width: expected.width,
                        expected_height: expected.height,
                    });
                }
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
            task_names,
            enjoy_mode: None,
        })
    }
}

/// The seed env `i` is reset with when the whole batch is reset with `seed`.
/// Enjoy mode hands its one env the seed unchanged instead.
fn env_seed(seed: u64, i: usize) -> u64 {
    SmallRng::seed_from_u64(seed.wrapping_add(i as u64)).next_u64()
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
            .for_each(|(i, (env, mut timestep))| env.reset(env_seed(seed, i), &mut timestep));
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
        // `new` rejects tasks whose width or height differ; `num_types` is the
        // union of the per-env vocabs already tracked by this wrapper.
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

    fn consume_metrics(&mut self) -> serde_json::Value {
        let mut metrics = serde_json::Map::new();
        for (task, name) in self.task_names.iter().enumerate() {
            let start = self.task_offsets[task];
            let end = self
                .task_offsets
                .get(task + 1)
                .copied()
                .unwrap_or(self.envs.len());
            let mean = crate::env::mean_metrics(
                self.envs[start..end]
                    .iter_mut()
                    .map(|env| env.consume_metrics()),
            );
            metrics.insert(name.clone(), mean);
        }
        serde_json::Value::Object(metrics)
    }

    fn num_tasks(&self) -> usize {
        self.num_tasks
    }

    fn task_names(&self) -> Vec<String> {
        self.task_names.clone()
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::envs::{find_return::FindReturnConfig, scouts::ScoutsConfig, snake::SnakeConfig};
    use crate::make::EnvConfig;
    use crate::policy::{Policy, RandomPolicy};
    use crate::timestep::TimeStepBuffers;

    /// One task per env kind, with several copies each so the first copy of
    /// a later task sits at a nonzero env and agent offset in the batch.
    pub(crate) fn mixed_specs() -> Vec<MultiEnvSpec> {
        let spec = |name: &str, num, env| MultiEnvSpec {
            name: name.into(),
            num,
            env: Box::new(env),
        };
        vec![
            spec(
                "scouts",
                2,
                EnvConfig::RustScouts(Box::new(ScoutsConfig {
                    num_scouts: 2,
                    num_harvesters: 1,
                    width: 16,
                    height: 16,
                    view_width: 11,
                    view_height: 11,
                    ..ScoutsConfig::default()
                })),
            ),
            spec(
                "fr",
                3,
                EnvConfig::RustFindReturn(Box::new(FindReturnConfig {
                    num_agents: 3,
                    width: 16,
                    height: 16,
                    view_width: 11,
                    view_height: 11,
                    // short enough that the flags unlock inside an episode
                    preparation_steps: 4,
                    ..FindReturnConfig::default()
                })),
            ),
            spec(
                "snake",
                2,
                EnvConfig::RustSnake(Box::new(SnakeConfig {
                    num_agents: 4,
                    width: 16,
                    height: 16,
                    view_width: 11,
                    view_height: 11,
                    food_spawn_prob: 0.05,
                    ..SnakeConfig::default()
                })),
            ),
        ]
    }

    /// A policy trained on the batch must see the same timesteps when one
    /// task is played alone: enjoy mode has to reproduce, bit for bit, the
    /// rows the batch gives that task's first copy, across an episode end.
    #[test]
    fn enjoy_mode_timesteps_match_the_batch_rows() {
        const LENGTH: usize = 12;
        let specs = mixed_specs();

        let mut batch = MultitaskWrapper::new(&specs, LENGTH).unwrap();
        let mut batch_buffers = TimeStepBuffers::new(&batch);
        let mut batch_actions = vec![0; batch.num_agents()];
        let mut policy = RandomPolicy::new();
        policy.reset(batch.num_agents(), 7).unwrap();

        // one enjoy-mode env per task, each driven by its task's first copy's
        // share of the batch actions
        let mut played: Vec<_> = (0..specs.len())
            .map(|task| {
                let mut env = MultitaskWrapper::new(&specs, LENGTH).unwrap();
                env.set_enjoy_mode(Some(task));
                let buffers = TimeStepBuffers::new(&env);
                let env_idx = batch.task_offsets[task];
                (env, buffers, batch.offsets[env_idx], batch.lens[env_idx])
            })
            .collect();

        for (task, (env, _, _, len)) in played.iter().enumerate() {
            assert_eq!(env.num_agents(), *len, "task {task} agent count");
        }

        let compare = |batch_buffers: &TimeStepBuffers, played: &[_], when: &str| {
            for (task, (_, buffers, offset, len)) in played.iter().enumerate() {
                let (buffers, offset, len): (&TimeStepBuffers, usize, usize) =
                    (buffers, *offset, *len);
                let expected = batch_buffers.rows(offset, len);
                assert_eq!(
                    expected.differing_fields(buffers),
                    Vec::<&str>::new(),
                    "task {} ({}) diverged from its batch rows {when}",
                    task,
                    specs[task].name,
                );
                assert!(buffers.task_ids.iter().all(|&id| id == task as i32));
            }
        };

        for (episode, seed) in [(0, 11u64), (1, 12)] {
            batch.reset(seed, &mut batch_buffers.view_mut());
            for (task, (env, buffers, ..)) in played.iter_mut().enumerate() {
                let env_idx = batch.task_offsets[task];
                env.reset(env_seed(seed, env_idx), &mut buffers.view_mut());
            }
            compare(
                &batch_buffers,
                &played,
                &format!("at episode {episode} reset"),
            );

            for step in 1..=LENGTH {
                policy
                    .act(&batch_buffers.view(), &mut batch_actions)
                    .unwrap();
                batch.step(&batch_actions, &mut batch_buffers.view_mut());
                for (env, buffers, offset, len) in played.iter_mut() {
                    let actions = &batch_actions[*offset..*offset + *len];
                    env.step(actions, &mut buffers.view_mut());
                }
                compare(
                    &batch_buffers,
                    &played,
                    &format!("at episode {episode} step {step}"),
                );
            }
            assert!(batch_buffers.terminated.iter().all(|&done| done));
        }
    }
}
