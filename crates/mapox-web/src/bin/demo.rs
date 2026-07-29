fn main() {
    mapox_core::render::open_window();
}

/// Raw wasm exports for miniquad's JS glue, which instantiates the module
/// itself and so cannot use wasm-bindgen. JS reads the string out of
/// `wasm_memory` with the bundle's `UTF8ToString(ptr, len)` helper.
#[unsafe(no_mangle)]
pub extern "C" fn version_ptr() -> *const u8 {
    mapox_core::version().as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn version_len() -> usize {
    mapox_core::version().len()
}
