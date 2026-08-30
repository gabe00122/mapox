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
                let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::common::Position;
    use crate::timestep::TimeStepBuffers;
    use ndarray::Array2;
    use parking_lot::Mutex;
    use std::sync::Arc;

    /// Recorded calls to the env behind the wrapper.
    #[derive(Debug, Default)]
    struct FakeLog {
        reset_seeds: Vec<u64>,
        step_actions: Vec<Vec<VocabId>>,
    }

    /// A recording stand-in: stamps its partition of the timestep with
    /// per-env markers, and records the reset seeds and (local-vocab)
    /// actions it receives. Each env is given a different local vocabulary
    /// that overlaps the others', so a shortcut that skipped the remap onto
    /// the merged vocab would show up in the assertions.
    struct FakeEnv {
        index: u8,
        num_agents: usize,
        obs_vocab: Vocabulary,
        action_vocab: Vocabulary,
        calls: u32,
        log: Arc<Mutex<FakeLog>>,
    }

    impl FakeEnv {
        fn new(
            index: u8,
            num_agents: usize,
            obs_vocab: Vocabulary,
            action_vocab: Vocabulary,
            log: Arc<Mutex<FakeLog>>,
        ) -> Self {
            Self {
                index,
                num_agents,
                obs_vocab,
                action_vocab,
                calls: 0,
                log,
            }
        }

        fn stamp(&mut self, timestep: &mut TimeStepMut) {
            self.calls += 1;
            timestep
                .time
                .fill(1000 * self.index as i32 + self.calls as i32);
            timestep.task_ids.fill(self.index as i32);
            timestep.reward.fill(self.index as f32);
            timestep.terminated.fill(false);
            // Local 0 is this env's *first* action; the wrapper's remap
            // decides which global id it becomes.
            timestep.last_action.fill(0);
            // The env's *last* local tile, for the same reason.
            timestep.obs.fill((self.obs_vocab.len() - 1) as VocabId);
            // The wrapper narrows this down to this env's actions.
            timestep.action_mask.fill(true);
        }
    }

    impl Environment for FakeEnv {
        fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
            self.log.lock().reset_seeds.push(seed);
            self.stamp(timestep);
        }

        fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
            self.log.lock().step_actions.push(actions.to_vec());
            self.stamp(timestep);
        }

        fn observation_spec(&self) -> ObservationSpec {
            ObservationSpec::new(3, 2, self.obs_vocab.len())
        }

        fn action_spec(&self) -> ActionSpec {
            ActionSpec::new(self.action_vocab.len())
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
            GridRenderSettings {
                // Marker: only the first env's reaches the caller.
                view_width: 10 + self.index as usize,
                ..Default::default()
            }
        }

        fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
            grid_render_state.tilemap = Array2::from_elem((2, 2), self.index as VocabId);
            grid_render_state.agent_positions = (0..self.num_agents)
                .map(|_| Position::new(self.index as i32, 0))
                .collect();
        }
    }

    fn vocab(symbols: &[&'static str]) -> Vocabulary {
        let mut vocab = Vocabulary::new();
        vocab.extend(symbols.iter().copied());
        vocab
    }

    /// Two envs with overlapping vocabularies in different id order. Env 0
    /// (2 agents) sees floor/wall/scout and env 1 (3 agents) sees
    /// wall/floor/treasure, so in the merged vocab (floor=0, wall=1,
    /// scout=2, treasure=3) env 1's tiles must be remapped. Same for
    /// actions: env 1's south/stay/east are local 0/1/2 but global 2/0/3.
    fn merged_env() -> (MultitaskWrapper, Vec<Arc<Mutex<FakeLog>>>) {
        let logs: Vec<Arc<Mutex<FakeLog>>> = (0..2)
            .map(|_| Arc::new(Mutex::new(FakeLog::default())))
            .collect();
        let envs: Vec<Box<dyn Environment>> = vec![
            Box::new(FakeEnv::new(
                0,
                2,
                vocab(&["floor", "wall", "scout"]),
                vocab(&["stay", "north", "south"]),
                logs[0].clone(),
            )),
            Box::new(FakeEnv::new(
                1,
                3,
                vocab(&["wall", "floor", "treasure"]),
                vocab(&["south", "stay", "east"]),
                logs[1].clone(),
            )),
        ];
        (MultitaskWrapper::new(envs), logs)
    }

    /// The per-env seed the wrapper derives: the base seed remixed with the
    /// env's index.
    fn derived_seed(base: u64, env_index: u64) -> u64 {
        let mut rng = SmallRng::seed_from_u64(base + env_index);
        rng.next_u64()
    }

    #[test]
    fn merging_combines_each_envs_vocabularies_and_specs() {
        let (env, _) = merged_env();

        assert_eq!(env.num_agents(), 5);
        let obs_spec = env.observation_spec();
        assert_eq!(
            (obs_spec.width, obs_spec.height, obs_spec.num_types),
            (3, 2, 4)
        );
        assert_eq!(env.action_spec().num_actions, 4);
        assert_eq!(
            env.obs_vocab().symbols(),
            &["floor", "wall", "scout", "treasure"][..]
        );
        assert_eq!(
            env.action_vocab().symbols(),
            &["stay", "north", "south", "east"][..]
        );
    }

    #[test]
    fn reset_seeds_each_env_differently_and_stamps_its_own_partition() {
        let (mut env, logs) = merged_env();
        let seed0 = derived_seed(42, 0);
        let seed1 = derived_seed(42, 1);
        assert_ne!(seed0, seed1);
        assert_ne!(seed0, 42);
        assert_ne!(seed1, 42);

        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(42, &mut buffers.view_mut());

        assert_eq!(logs[0].lock().reset_seeds, vec![seed0]);
        assert_eq!(logs[1].lock().reset_seeds, vec![seed1]);

        // The buffers take their shape from the merged specs...
        assert_eq!(buffers.obs.dim(), (5, 3, 2, 1));
        assert_eq!(buffers.action_mask.dim(), (5, 4));

        // ...and each env's stamp lands on its own agent rows, remapped
        // into the merged vocab: env 0's last tile is scout (2) and env
        // 1's is treasure (3); local action 0 is stay (global 0) for env
        // 0 but south (global 2) for env 1.
        for agent in 0..2 {
            assert_eq!(buffers.time[agent], 1);
            assert_eq!(buffers.task_ids[agent], 0);
            assert_eq!(buffers.reward[agent], 0.0);
            assert_eq!(buffers.last_action[agent], 0);
            assert_eq!(buffers.obs[[agent, 0, 0, 0]], 2);
            assert_eq!(
                buffers
                    .action_mask
                    .row(agent)
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                vec![true, true, true, false]
            );
        }
        for agent in 2..5 {
            assert_eq!(buffers.time[agent], 1001);
            assert_eq!(buffers.task_ids[agent], 1);
            assert_eq!(buffers.reward[agent], 1.0);
            assert_eq!(buffers.last_action[agent], 2);
            assert_eq!(buffers.obs[[agent, 0, 0, 0]], 3);
            assert_eq!(
                buffers
                    .action_mask
                    .row(agent)
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                vec![true, false, true, true]
            );
        }
        assert!(!buffers.terminated.iter().any(|&t| t));

        // The same base seed re-derives the same per-env seeds...
        env.reset(42, &mut buffers.view_mut());
        assert_eq!(logs[0].lock().reset_seeds, vec![seed0, seed0]);
        assert_eq!(logs[1].lock().reset_seeds, vec![seed1, seed1]);

        // ...and a different base seed derives different ones.
        env.reset(43, &mut buffers.view_mut());
        assert_ne!(logs[0].lock().reset_seeds[2], seed0);
        assert_ne!(logs[1].lock().reset_seeds[2], seed1);
    }

    #[test]
    fn step_sends_each_env_only_its_agents_and_decodes_its_actions() {
        let (mut env, logs) = merged_env();
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(7, &mut buffers.view_mut());

        // Global actions: env 0 gets south, south; env 1 gets east, stay,
        // east, which are local 2, 1, 2 in env 1's vocab.
        env.step(&[2, 2, 3, 0, 3], &mut buffers.view_mut());
        assert_eq!(logs[0].lock().step_actions, vec![vec![2, 2]]);
        assert_eq!(logs[1].lock().step_actions, vec![vec![2, 1, 2]]);

        // The step's stamps landed in the same partitions as reset's.
        for agent in 0..2 {
            assert_eq!(buffers.time[agent], 2);
        }
        for agent in 2..5 {
            assert_eq!(buffers.time[agent], 1002);
        }

        // A second step accumulates per env: global north, stay for env 0
        // (local 1, 0); east, south, east for env 1 (local 2, 0, 2).
        env.step(&[1, 0, 3, 2, 3], &mut buffers.view_mut());
        assert_eq!(logs[0].lock().step_actions, vec![vec![2, 2], vec![1, 0]]);
        assert_eq!(
            logs[1].lock().step_actions,
            vec![vec![2, 1, 2], vec![2, 0, 2]]
        );
    }

    #[test]
    fn a_single_env_still_gets_a_remixed_seed() {
        let log = Arc::new(Mutex::new(FakeLog::default()));
        let env = FakeEnv::new(
            0,
            3,
            vocab(&["floor", "wall"]),
            vocab(&["stay", "north"]),
            log.clone(),
        );
        let mut wrapped = MultitaskWrapper::new(vec![Box::new(env)]);

        assert_eq!(wrapped.num_agents(), 3);
        assert_eq!(wrapped.observation_spec().num_types, 2);
        assert_eq!(wrapped.action_spec().num_actions, 2);

        let mut buffers = TimeStepBuffers::new(&wrapped);
        wrapped.reset(99, &mut buffers.view_mut());

        let expected = derived_seed(99, 0);
        assert_eq!(log.lock().reset_seeds, vec![expected]);
        assert_ne!(expected, 99);
    }

    #[test]
    fn rendering_delegates_to_the_first_env() {
        let (env, _) = merged_env();

        // Only env 0's marker reaches the caller, not env 1's.
        assert_eq!(env.get_render_settings().view_width, 10);

        let mut state = GridRenderState::default();
        env.render_state_into(&mut state);
        assert!(state.tilemap.iter().all(|&tile| tile == 0));
        assert_eq!(
            state
                .agent_positions
                .iter()
                .map(|p| p.idx())
                .collect::<Vec<_>>(),
            vec![[0, 0]; 2]
        );
    }
}
