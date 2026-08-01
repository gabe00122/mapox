pub mod env;
pub mod envs;
pub mod render;
pub mod spec;
pub mod symbols;
pub mod timestep;
pub mod vocab;

/// Returns the version of the core crate.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
