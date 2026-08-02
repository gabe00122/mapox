//! wasm-bindgen exports for the browser. The page imports the generated JS
//! module, awaits `init()`, then calls [`start`] with its canvas.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn version() -> String {
    mapox_core::version().to_owned()
}

/// Starts the demo on `canvas`. Resolves once the app is running; the app
/// then owns the canvas for the life of the page.
#[wasm_bindgen]
pub async fn start(canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
    // Without this a Rust panic surfaces as an opaque "unreachable executed";
    // with it the message and backtrace land in the browser console.
    console_error_panic_hook::set_once();
    // eframe reports backend choices and failures through `log`; without a
    // logger they vanish, and a broken canvas stays a silent black rectangle.
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|_cc| Ok(Box::new(mapox_core::render::RenderApp::demo()))),
        )
        .await
}
