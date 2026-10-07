//! Rollout throughput of survival vs. find-return under the VectorWrapper,
//! shaped like vectorized training: `num_envs` copies of `agents` each, reset
//! every `length` steps, random actions drawn uniformly from each action mask.
//!
//! Run: cargo run --release -p mapox-core --example bench_vec -- [num_envs] [agents] [length] [episodes]

use std::time::{Duration, Instant};

use mapox_core::envs::find_return::FindReturnConfig;
use mapox_core::envs::survival::SurvivalConfig;
use mapox_core::make::{EnvConfig, make};
use mapox_core::timestep::TimeStepBuffers;
use mapox_core::vocab::VocabId;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// One uniformly random unmasked action per agent.
fn sample_actions(buffers: &TimeStepBuffers, rng: &mut SmallRng, actions: &mut [VocabId]) {
    for (agent, action) in actions.iter_mut().enumerate() {
        let mask = buffers.action_mask.row(agent);
        let valid = mask.iter().filter(|&&ok| ok).count();
        let mut pick = rng.random_range(0..valid.max(1));
        *action = mask
            .iter()
            .position(|&ok| {
                let hit = ok && pick == 0;
                if ok {
                    pick = pick.wrapping_sub(1);
                }
                hit
            })
            .unwrap_or(0) as VocabId;
    }
}

struct Run {
    resets: Vec<Duration>,
    /// step time summed per quarter of the episode
    quarters: [Duration; 4],
    steps: usize,
    sampling: Duration,
}

fn run(config: &EnvConfig, length: usize, episodes: usize) -> (usize, Run) {
    let mut env = make(config, length).unwrap();
    let num_agents = env.num_agents();
    let mut buffers = TimeStepBuffers::new(&*env);
    let mut actions = vec![0; num_agents];
    let mut rng = SmallRng::seed_from_u64(0);

    // warm-up: page in buffers, spin up the rayon pool
    env.reset(1_000, &mut buffers.view_mut());
    for _ in 0..64.min(length) {
        sample_actions(&buffers, &mut rng, &mut actions);
        env.step(&actions, &mut buffers.view_mut());
    }

    let mut out = Run {
        resets: Vec::new(),
        quarters: [Duration::ZERO; 4],
        steps: 0,
        sampling: Duration::ZERO,
    };
    for episode in 0..episodes {
        let start = Instant::now();
        env.reset(episode as u64, &mut buffers.view_mut());
        out.resets.push(start.elapsed());

        for t in 0..length {
            let start = Instant::now();
            sample_actions(&buffers, &mut rng, &mut actions);
            out.sampling += start.elapsed();

            let start = Instant::now();
            env.step(&actions, &mut buffers.view_mut());
            out.quarters[t * 4 / length] += start.elapsed();
            out.steps += 1;
        }
        let _ = env.consume_metrics();
    }
    (num_agents, out)
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn report(name: &str, config: &EnvConfig, single: &EnvConfig, length: usize, episodes: usize) {
    let (num_agents, vec) = run(config, length, episodes);
    let (_, one) = run(single, length, episodes);

    let step_total: Duration = vec.quarters.iter().sum();
    let reset_total: Duration = vec.resets.iter().sum();
    let step_us = us(step_total) / vec.steps as f64;
    let reset_ms = us(reset_total) / 1e3 / vec.resets.len() as f64;
    let wall = step_total + reset_total;
    let one_step_us = us(one.quarters.iter().sum()) / one.steps as f64;
    let one_reset_ms = us(one.resets.iter().sum()) / 1e3 / one.resets.len() as f64;
    let quarter_steps = vec.steps as f64 / 4.0;

    println!("{name}  ({num_agents} agents)");
    println!(
        "  vec step   {step_us:>9.0} us   {:>6.1}M agent steps/s",
        num_agents as f64 / step_us
    );
    println!(
        "  by quarter {}",
        vec.quarters
            .iter()
            .map(|q| format!("{:>6.0}", us(*q) / quarter_steps))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "  vec reset  {reset_ms:>9.1} ms   ({:.1}% of a {length}-step episode)",
        100.0 * us(reset_total) / us(wall)
    );
    println!(
        "  with reset {:>9.0} us   {:>6.1}M agent steps/s",
        us(wall) / vec.steps as f64,
        (num_agents * vec.steps) as f64 / wall.as_secs_f64() / 1e6
    );
    println!(
        "  1 env, 1 thread: step {one_step_us:.1} us, reset {one_reset_ms:.2} ms -> parallel speedup {:.1}x",
        one_step_us * (num_agents as f64 / one_agents(single)) / step_us
    );
    println!(
        "  (action sampling, untimed: {:.0} us/step)",
        us(vec.sampling) / vec.steps as f64
    );
}

fn one_agents(config: &EnvConfig) -> f64 {
    make(config, 1).unwrap().num_agents() as f64
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric args"))
        .collect();
    let num_envs = args.first().copied().unwrap_or(256);
    let agents = args.get(1).copied().unwrap_or(8);
    let length = args.get(2).copied().unwrap_or(1024);
    let episodes = args.get(3).copied().unwrap_or(2);

    println!(
        "{num_envs} envs x {agents} agents, {length}-step episodes x {episodes}, {} rayon threads\n",
        rayon::current_num_threads()
    );

    let envs = [
        (
            "find_return",
            EnvConfig::RustFindReturn(Box::new(FindReturnConfig {
                num_agents: agents,
                ..Default::default()
            })),
        ),
        (
            "survival",
            EnvConfig::RustSurvival(Box::new(SurvivalConfig {
                num_agents: agents,
                ..Default::default()
            })),
        ),
    ];
    for (name, single) in envs {
        let vec = EnvConfig::RustVec {
            num: num_envs,
            env: Box::new(single.clone()),
        };
        report(name, &vec, &single, length, episodes);
        println!();
    }
}
