//! wasm-bindgen exports for the browser: the page imports the generated JS
//! module, awaits `init()`, then calls [`start`] with its canvas and the
//! bytes of a policy exported by jaxrl's `scripts/export_burn_policy.py`.
//!
//! This is the browser shell around [`mapox_burn`] — the same
//! [`BurnPolicy`](mapox_burn::BurnPolicy) and
//! [`RenderApp`](mapox_core::render::RenderApp) that
//! `cargo run -p mapox-burn --example demo` runs natively, compiled to wasm
//! and running on the same CPU backend. The env comes from the training run's
//! own config, embedded in the export, so the page never has to be told which
//! env the policy was trained on.
#![cfg(target_arch = "wasm32")]

use mapox_burn::burn::backend::Flex;
use mapox_burn::burn::backend::flex::FlexDevice;
use mapox_burn::{BurnPolicy, LoadedPolicy, load_policy_bytes};
use mapox_core::env::Environment;
use mapox_core::make::{EnvConfig, make};
use mapox_core::policy::Policy;
use mapox_core::render::RenderApp;
use wasm_bindgen::prelude::*;

/// Starts the viewer on `canvas`, driven by the policy in `policy_bytes` (a
/// `*.safetensors` bundle, fetched by the page). Resolves once the app is
/// running; the app then owns the canvas for the life of the page.
#[wasm_bindgen]
pub async fn start(
    canvas: web_sys::HtmlCanvasElement,
    policy_bytes: Vec<u8>,
    // u32 rather than the u64 the policy wants: wasm-bindgen maps u64 to a JS
    // BigInt, and a plain number literal from the page would be a TypeError
    seed: u32,
) -> Result<(), JsValue> {
    // Without this a Rust panic surfaces as an opaque "unreachable executed";
    // with it the message and backtrace land in the browser console.
    console_error_panic_hook::set_once();
    // eframe reports backend choices and failures through `log`, and so does
    // BurnPolicy's per-cycle reward line; without a logger they vanish, and a
    // broken canvas stays a silent black rectangle.
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let (env, length, policy) = setup(&policy_bytes, seed)?;

    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(move |_cc| Ok(Box::new(RenderApp::new(env, length, seed.into(), policy)))),
        )
        .await
}

/// What [`setup`] hands back: an env, the episode length to run it for, and
/// the policy that drives it.
type Running = (Box<dyn Environment + Send + Sync>, usize, Box<dyn Policy>);

/// Load the policy and rebuild the env the run was trained on.
fn setup(policy_bytes: &[u8], seed: u32) -> Result<Running, JsValue> {
    let loaded: LoadedPolicy<Flex> = load_policy_bytes(policy_bytes, &FlexDevice)
        .map_err(|err| JsValue::from_str(&format!("loading policy: {err}")))?;
    log::info!(
        "loaded {} ({} layers, hidden {}, seq {})",
        loaded.meta.source,
        loaded.meta.model.num_layers,
        loaded.meta.model.hidden_features,
        loaded.meta.max_seq_length,
    );

    let env_json =
        loaded.meta.env_config.as_deref().ok_or_else(|| {
            JsValue::from_str("export has no env config embedded (fixture file?)")
        })?;
    let env_config: EnvConfig = serde_json::from_str(env_json)
        .map_err(|err| JsValue::from_str(&format!("parsing the embedded env config: {err}")))?;
    let env = make(&env_config);

    let policy = BurnPolicy::<Flex>::new(loaded, env.num_agents(), seed.into());
    let length = policy.context_length();
    Ok((env, length, Box::new(policy)))
}
