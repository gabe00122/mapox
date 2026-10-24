//! Single-core step throughput of every rust env at its default config.
//!
//! Each env is stepped directly, with no VectorWrapper, so everything runs on
//! the calling thread. Actions are sampled uniformly from each agent's action
//! mask outside the timed region, and episodes are reset whenever they end;
//! resets are timed separately and left out of the steps/s figure.
//!
//! Run: cargo run --release -p mapox-core --example bench_envs --
//!          [--steps N] [--length L] [--seed S] [ENV...]

use std::time::{Duration, Instant};

use mapox_core::{
    env::Environment,
    envs::{
        find_return::FindReturnConfig, hide_seek::HideSeekConfig, maze::MazeConfig,
        pacman::PacmanConfig, scouts::ScoutsConfig, survival::config::SurvivalConfig,
    },
    make::{EnvConfig, make},
    timestep::TimeStepBuffers,
    vocab::VocabId,
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

const WARMUP_STEPS: usize = 256;

struct Args {
    steps: usize,
    length: usize,
    seed: u64,
    envs: Vec<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        steps: 20_000,
        length: 256,
        seed: 0,
        envs: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    let value = |it: &mut dyn Iterator<Item = String>, flag: &str| -> Result<u64, String> {
        it.next()
            .ok_or(format!("{flag} needs a value"))?
            .parse()
            .map_err(|e| format!("{flag}: {e}"))
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--steps" => args.steps = value(&mut it, "--steps")? as usize,
            "--length" => args.length = value(&mut it, "--length")? as usize,
            "--seed" => args.seed = value(&mut it, "--seed")?,
            _ if !arg.starts_with('-') => args.envs.push(arg),
            _ => return Err(format!("unexpected argument {arg:?}")),
        }
    }
    if args.steps == 0 || args.length == 0 {
        return Err("--steps and --length must be positive".into());
    }
    Ok(args)
}

fn all_envs() -> Vec<(&'static str, EnvConfig)> {
    vec![
        (
            "find_return",
            EnvConfig::RustFindReturn(Box::<FindReturnConfig>::default()),
        ),
        (
            "scouts",
            EnvConfig::RustScouts(Box::<ScoutsConfig>::default()),
        ),
        (
            "pacman",
            EnvConfig::RustPacman(Box::<PacmanConfig>::default()),
        ),
        (
            "survival",
            EnvConfig::RustSurvival(Box::<SurvivalConfig>::default()),
        ),
        ("maze", EnvConfig::RustMaze(Box::<MazeConfig>::default())),
        (
            "hide_seek",
            EnvConfig::RustHideSeek(Box::<HideSeekConfig>::default()),
        ),
    ]
}

/// One uniformly random legal action per agent.
fn sample_actions(rng: &mut SmallRng, buffers: &TimeStepBuffers, actions: &mut [VocabId]) {
    for (action, mask) in actions.iter_mut().zip(buffers.action_mask.rows()) {
        let legal = mask.iter().filter(|&&ok| ok).count();
        *action = if legal == 0 {
            0
        } else {
            let pick = rng.random_range(0..legal);
            mask.iter()
                .enumerate()
                .filter(|&(_, &ok)| ok)
                .nth(pick)
                .unwrap()
                .0 as VocabId
        };
    }
}

struct BenchResult {
    agents: usize,
    step_time: Duration,
    resets: usize,
    reset_time: Duration,
}

fn bench(env: &mut dyn Environment, args: &Args) -> BenchResult {
    let mut buffers = TimeStepBuffers::new(env);
    let mut actions = vec![0; env.num_agents()];
    let mut rng = SmallRng::seed_from_u64(args.seed);
    let mut episode = args.seed;

    env.reset(episode, &mut buffers.view_mut());

    let mut result = BenchResult {
        agents: env.num_agents(),
        step_time: Duration::ZERO,
        resets: 0,
        reset_time: Duration::ZERO,
    };
    for i in 0..WARMUP_STEPS + args.steps {
        let timed = i >= WARMUP_STEPS;

        // every agent in an env shares one clock, so agent 0 speaks for all
        if buffers.terminated[0] {
            episode += 1;
            let start = Instant::now();
            env.reset(episode, &mut buffers.view_mut());
            if timed {
                result.reset_time += start.elapsed();
                result.resets += 1;
            }
        }

        sample_actions(&mut rng, &buffers, &mut actions);
        let start = Instant::now();
        env.step(&actions, &mut buffers.view_mut());
        if timed {
            result.step_time += start.elapsed();
        }
    }
    result
}

fn human(x: f64) -> String {
    for (div, suffix) in [(1e9, "B"), (1e6, "M"), (1e3, "K")] {
        if x >= div {
            return format!("{:.2}{suffix}", x / div);
        }
    }
    format!("{x:.0}")
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };

    let known = all_envs();
    for name in &args.envs {
        if !known.iter().any(|(known, _)| known == name) {
            let names: Vec<_> = known.iter().map(|(name, _)| *name).collect();
            eprintln!("unknown env {name:?}, expected one of {}", names.join(", "));
            std::process::exit(2);
        }
    }

    println!(
        "single core, default configs, {} steps, episode length {}",
        args.steps, args.length
    );
    println!(
        "{:<12} {:>7} {:>12} {:>14} {:>10} {:>10}",
        "env", "agents", "env steps/s", "agent steps/s", "us/step", "us/reset"
    );
    for (name, config) in known {
        if !args.envs.is_empty() && !args.envs.iter().any(|env| env == name) {
            continue;
        }
        let mut env = make(&config, args.length).expect("default configs build");
        let r = bench(env.as_mut(), &args);

        let steps_per_sec = args.steps as f64 / r.step_time.as_secs_f64();
        let reset_us = if r.resets == 0 {
            "-".to_string()
        } else {
            format!("{:.1}", r.reset_time.as_secs_f64() * 1e6 / r.resets as f64)
        };
        println!(
            "{:<12} {:>7} {:>12} {:>14} {:>10.2} {:>10}",
            name,
            r.agents,
            human(steps_per_sec),
            human(steps_per_sec * r.agents as f64),
            r.step_time.as_secs_f64() * 1e6 / args.steps as f64,
            reset_us,
        );
    }
}
