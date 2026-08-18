use burn::tensor::backend::Backend;
use mapox_core::make::{EnvConfig, make};
use mapox_core::render::{RenderApp, open_window};

use mapox_burn::{BurnPolicy, load_policy};

type MainError = Box<dyn std::error::Error + Send + Sync>;

struct Args {
    policy_path: String,
    seed: u64,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        policy_path: String::new(),
        seed: 0,
    };
    let mut it = std::env::args().skip(1);
    let value = |it: &mut dyn Iterator<Item = String>, flag: &str| {
        it.next().ok_or(format!("{flag} needs a value"))
    };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--seed" => {
                args.seed = value(&mut it, "--seed")?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?
            }
            _ if args.policy_path.is_empty() && !arg.starts_with('-') => args.policy_path = arg,
            _ => return Err(format!("unexpected argument {arg:?}")),
        }
    }
    if args.policy_path.is_empty() {
        return Err("usage: demo <policy.safetensors> [--seed N]".into());
    }
    Ok(args)
}

fn run<B: Backend>(args: &Args) -> Result<(), MainError>
where
    B::Device: Default,
{
    let loaded = load_policy::<B>(&args.policy_path, &B::Device::default())?;
    println!(
        "loaded {} ({} layers, hidden {}, seq {}) on {}",
        loaded.meta.source,
        loaded.meta.model.num_layers,
        loaded.meta.model.hidden_features,
        loaded.meta.max_seq_length,
        B::name(&B::Device::default()),
    );

    let length = loaded.meta.max_seq_length;

    let env_json = loaded
        .meta
        .env_config
        .clone()
        .ok_or("export has no env config embedded (fixture file?)")?;
    let env_config: EnvConfig = serde_json::from_str(&env_json)?;
    let env = make(&env_config, length);

    let policy = BurnPolicy::new(loaded, env.num_agents(), args.seed);

    open_window(RenderApp::new(env, length, args.seed, Box::new(policy)))?;
    Ok(())
}

fn main() -> Result<(), MainError> {
    // BurnPolicy reports its per-cycle reward through `log`. wgpu and cubecl
    // stay at warn: at info they bury it under adapter and autotune dumps.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,mapox_burn=info"),
    )
    .init();
    let args = parse_args()?;

    run::<burn::backend::Flex>(&args)
}
