//! Env stepping throughput: `cargo run --release -p mapox-core --example bench`.

use std::time::Instant;

use mapox_core::env::Environment;
use mapox_core::envs::find_return::{FindReturn, FindReturnConfig};
use mapox_core::timestep::{OBS_CHANNELS, TimeStepMut};
use mapox_core::vocab::VocabId;
use mapox_core::wrappers::vector::VectorWrapper;
use ndarray::{Array1, Array2, Array4};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

const AGENTS_PER_ENV: usize = 32;
const TOTAL_AGENTS: usize = 4096;
const WARMUP_STEPS: usize = 200;
const BENCH_STEPS: usize = 20000;

struct Buffers {
    obs: Array4<VocabId>,
    time: Array1<i32>,
    terminated: Array1<bool>,
    last_action: Array1<i32>,
    reward: Array1<f32>,
    action_mask: Array2<bool>,
    task_ids: Array1<i32>,
}

impl Buffers {
    fn new(env: &impl Environment) -> Self {
        let num_agents = env.num_agents();
        let obs_spec = env.observation_spec();

        Self {
            obs: Array4::zeros((
                num_agents,
                obs_spec.width as usize,
                obs_spec.height as usize,
                OBS_CHANNELS,
            )),
            time: Array1::zeros(num_agents),
            terminated: Array1::default(num_agents),
            last_action: Array1::zeros(num_agents),
            reward: Array1::zeros(num_agents),
            action_mask: Array2::default((num_agents, env.action_spec().num_actions)),
            task_ids: Array1::zeros(num_agents),
        }
    }

    fn timestep(&mut self) -> TimeStepMut<'_> {
        TimeStepMut {
            obs: self.obs.view_mut(),
            time: self.time.view_mut(),
            terminated: self.terminated.view_mut(),
            last_action: self.last_action.view_mut(),
            reward: self.reward.view_mut(),
            action_mask: self.action_mask.view_mut(),
            task_ids: self.task_ids.view_mut(),
        }
    }
}

fn bench(label: &str, env: &mut impl Environment) {
    let mut buffers = Buffers::new(env);
    let mut rng = SmallRng::seed_from_u64(0);

    let num_agents = env.num_agents();
    let num_actions = env.action_spec().num_actions;
    let actions: Vec<i32> = (0..num_agents)
        .map(|_| rng.random_range(0..num_actions as i32))
        .collect();

    env.reset(0, &mut buffers.timestep());

    for _ in 0..WARMUP_STEPS {
        env.step(&actions, &mut buffers.timestep());
    }

    let start = Instant::now();
    for _ in 0..BENCH_STEPS {
        env.step(&actions, &mut buffers.timestep());
    }
    let elapsed = start.elapsed();

    let steps_per_sec = BENCH_STEPS as f64 / elapsed.as_secs_f64();
    let agent_steps_per_sec = steps_per_sec * num_agents as f64;
    println!(
        "{label}: {steps_per_sec:.0} steps/s, {agent_steps_per_sec:.3e} agent-steps/s \
         ({num_agents} agents, {BENCH_STEPS} steps in {elapsed:.2?})"
    );
}

fn main() {
    let config = FindReturnConfig {
        num_agents: AGENTS_PER_ENV,
        ..Default::default()
    };
    let vec_count = TOTAL_AGENTS / AGENTS_PER_ENV;

    bench("single env", &mut FindReturn::new(&config));
    bench(
        &format!("vector ({vec_count} envs)"),
        &mut VectorWrapper::new(FindReturn::new(&config), vec_count),
    );
}
