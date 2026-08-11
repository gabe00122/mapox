//! Env stepping throughput: `cargo run --release -p mapox-core --example bench`.

use std::time::Instant;

use mapox_core::env::Environment;
use mapox_core::envs::find_return::FindReturnConfig;
use mapox_core::make::EnvConfig;
use mapox_core::make::{make, make_vec};
use mapox_core::timestep::TimeStepBuffers;
use mapox_core::vocab::VocabId;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

const AGENTS_PER_ENV: usize = 8;
const TOTAL_AGENTS: usize = 4096;
const WARMUP_STEPS: usize = 200;
const BENCH_STEPS: usize = 20000;

fn format_rate(rate: f64) -> String {
    const UNITS: &[(f64, &str)] = &[(1_000_000_000.0, "B"), (1_000_000.0, "M"), (1_000.0, "K")];

    for &(threshold, suffix) in UNITS {
        if rate >= threshold {
            return format!("{:.2}{suffix}", rate / threshold);
        }
    }

    format!("{rate:.0}")
}

fn bench(label: &str, env: &mut Box<dyn Environment>) {
    let mut buffers = TimeStepBuffers::new(env.as_ref());
    let mut rng = SmallRng::seed_from_u64(0);

    let num_agents = env.num_agents();
    let num_actions = env.action_spec().num_actions;
    let actions: Vec<VocabId> = (0..num_agents)
        .map(|_| {
            VocabId::try_from(rng.random_range(0..num_actions))
                .expect("action vocab exceeds VocabId capacity")
        })
        .collect();

    env.reset(0, &mut buffers.view_mut());

    for _ in 0..WARMUP_STEPS {
        env.step(&actions, &mut buffers.view_mut());
    }

    let start = Instant::now();
    for _ in 0..BENCH_STEPS {
        env.step(&actions, &mut buffers.view_mut());
    }
    let elapsed = start.elapsed();

    let steps_per_sec = BENCH_STEPS as f64 / elapsed.as_secs_f64();
    let agent_steps_per_sec = steps_per_sec * num_agents as f64;
    let steps_per_sec = format_rate(steps_per_sec);
    let agent_steps_per_sec = format_rate(agent_steps_per_sec);
    println!(
        "{label}: {steps_per_sec} steps/s, {agent_steps_per_sec} agent-steps/s \
         ({num_agents} agents, {BENCH_STEPS} steps in {elapsed:.2?})"
    );
}

fn main() {
    let config = FindReturnConfig {
        num_agents: AGENTS_PER_ENV,
        ..Default::default()
    };
    let vec_count = TOTAL_AGENTS / AGENTS_PER_ENV;

    bench(
        "single env",
        &mut make(&EnvConfig::RustFindReturn(config.clone())),
    );
    bench(
        &format!("vector ({vec_count} envs)"),
        &mut make_vec(&EnvConfig::RustFindReturn(config.clone()), vec_count),
    );
}
