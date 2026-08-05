pub mod env;
pub mod envs;
pub mod make;
pub mod map_gen;
pub mod policy;
pub mod render;
pub mod spec;
pub mod symbols;
pub mod timestep;
pub mod vocab;
pub mod wrappers;

/// Returns the version of the core crate.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
