use wasm_bindgen::prelude::wasm_bindgen;

/// Returns the version of the compiled wasm module.
#[wasm_bindgen]
pub fn version() -> String {
    mapox_core::version().to_string()
}
