//! Native tileset demo: `cargo run -p mapox-core --example demo`.
//! The python extension opens the same window via `mapox.run_demo()`.

fn main() -> eframe::Result {
    mapox_core::render::open_window(mapox_core::render::RenderApp::demo())
}
