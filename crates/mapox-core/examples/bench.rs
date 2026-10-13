//! Survival env stepping throughput:
//! `cargo run --release -p mapox-core --example bench`.

use std::time::Instant;

use mapox_core::env::Environment;
use mapox_core::envs::survival::config::SurvivalConfig;
use mapox_core::make::{EnvConfig, make};
use mapox_core::timestep::TimeStepBuffers;
use mapox_core::vocab::VocabId;
use rand::{RngExt, SeedableRng, rngs::StdRng};

const AGENTS_PER_ENV: usize = 8;
const TOTAL_AGENTS: usize = 4096;
const EPISODE_LENGTH: usize = 1024;
const ACTION_POOL: usize = 64;

fn format_rate(rate: f64) -> String {
    const UNITS: &[(f64, &str)] = &[(1_000_000_000.0, "B"), (1_000_000.0, "M"), (1_000.0, "K")];

    for &(threshold, suffix) in UNITS {
        if rate >= threshold {
            return format!("{:.2}{suffix}", rate / threshold);
        }
    }

    format!("{rate:.0}")
}

fn bench(label: &str, env: &mut Box<dyn Environment>, warmup: usize, steps: usize) {
    let mut buffers = TimeStepBuffers::new(env.as_ref());
    let mut rng = StdRng::seed_from_u64(0);

    // Random actions, drawn up front so the timed loop is env work only.
    let num_agents = env.num_agents();
    let num_actions = env.action_spec().num_actions;
    let actions: Vec<Vec<VocabId>> = (0..ACTION_POOL)
        .map(|_| {
            (0..num_agents)
                .map(|_| VocabId::try_from(rng.random_range(0..num_actions)).unwrap())
                .collect()
        })
        .collect();

    let start = Instant::now();
    env.reset(0, &mut buffers.view_mut());
    let reset_time = start.elapsed();

    for i in 0..warmup {
        env.step(&actions[i % ACTION_POOL], &mut buffers.view_mut());
    }

    let start = Instant::now();
    for i in 0..steps {
        env.step(&actions[i % ACTION_POOL], &mut buffers.view_mut());
    }
    let elapsed = start.elapsed();

    let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
    let agent_steps_per_sec = steps_per_sec * num_agents as f64;
    let step_us = elapsed.as_secs_f64() * 1e6 / steps as f64;
    println!(
        "{label}: {} steps/s, {} agent-steps/s, {step_us:.1} µs/step, reset {reset_time:.2?} \
         ({num_agents} agents, {steps} steps in {elapsed:.2?})",
        format_rate(steps_per_sec),
        format_rate(agent_steps_per_sec),
    );
}

fn main() {
    let config = EnvConfig::RustSurvival(Box::new(SurvivalConfig {
        num_agents: AGENTS_PER_ENV,
        ..Default::default()
    }));
    let num = TOTAL_AGENTS / AGENTS_PER_ENV;

    bench(
        "single env",
        &mut make(&config, EPISODE_LENGTH).unwrap(),
        EPISODE_LENGTH,
        20 * EPISODE_LENGTH,
    );
    bench(
        &format!("vector ({num} envs)"),
        &mut make(
            &EnvConfig::RustVec {
                num,
                env: Box::new(config),
            },
            EPISODE_LENGTH,
        )
        .unwrap(),
        EPISODE_LENGTH,
        4 * EPISODE_LENGTH,
    );
}
