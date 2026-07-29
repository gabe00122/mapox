pub mod env;
pub mod envs;
pub mod render;
pub mod timestep;

/// Returns the version of the core crate.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
