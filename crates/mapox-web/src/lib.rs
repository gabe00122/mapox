//! wasm-bindgen exports for the browser: the page imports the generated JS
//! module, awaits `init()`, then constructs a [`WebHandle`] and starts it
//! with its canvas and the bytes of a policy exported by jaxrl's
//! `scripts/export_burn_policy.py`.
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

/// The running viewer, as the page holds it. Construct one, [`start`] it on a
/// canvas, and [`destroy`] it to stop the app and release the canvas and GPU
/// resources; the wasm instance itself lives as long as the page does.
///
/// [`start`]: WebHandle::start
/// [`destroy`]: WebHandle::destroy
#[wasm_bindgen]
pub struct WebHandle {
    runner: eframe::WebRunner,
}

#[wasm_bindgen]
impl WebHandle {
    /// Installs eframe's panic hook, which logs a Rust panic's message and
    /// backtrace to the browser console (otherwise an opaque "unreachable
    /// executed") and backs [`has_panicked`](Self::has_panicked).
    #[wasm_bindgen(constructor)]
    #[expect(clippy::new_without_default, reason = "a JS constructor")]
    pub fn new() -> Self {
        // eframe reports backend choices and failures through `log`, and so
        // does BurnPolicy's per-cycle reward line; without a logger they
        // vanish, and a broken canvas stays a silent black rectangle.
        eframe::WebLogger::init(log::LevelFilter::Debug).ok();
        Self {
            runner: eframe::WebRunner::new(),
        }
    }

    /// Starts the viewer on `canvas`, driven by the policy in `policy_bytes`
    /// (a `*.safetensors` bundle, fetched by the page). Resolves once the app
    /// is running; it then runs until [`destroy`](Self::destroy).
    pub async fn start(
        &self,
        canvas: web_sys::HtmlCanvasElement,
        policy_bytes: Vec<u8>,
        // u32 rather than the u64 the policy wants: wasm-bindgen maps u64 to
        // a JS BigInt, and a plain number literal from the page would be a
        // TypeError
        seed: u32,
    ) -> Result<(), JsValue> {
        let (env, length, policy) = setup(&policy_bytes, seed)?;

        self.runner
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(move |_cc| {
                    Ok(Box::new(RenderApp::new(env, length, seed.into(), policy)))
                }),
            )
            .await
    }

    /// Stops the app: unhooks its event listeners, cancels the next frame,
    /// and drops the app along with its GPU resources.
    pub fn destroy(&self) {
        self.runner.destroy();
    }

    /// Whether the app has panicked; it stops drawing when it does.
    pub fn has_panicked(&self) -> bool {
        self.runner.has_panicked()
    }

    /// The panic message, if the app has panicked.
    pub fn panic_message(&self) -> Option<String> {
        self.runner.panic_summary().map(|summary| summary.message())
    }
}

/// What [`setup`] hands back: an env, the episode length to run it for, and
/// the policy that drives it.
type Running = (Box<dyn Environment>, usize, Box<dyn Policy>);

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
    // the episode runs for one policy context, which the export declares
    let length = loaded.meta.max_seq_length;
    let env = make(&env_config, length)
        .map_err(|err| JsValue::from_str(&format!("building the env: {err}")))?;

    let policy = BurnPolicy::<Flex>::new(loaded, seed.into());
    Ok((env, length, Box::new(policy)))
}
