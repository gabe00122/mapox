mod env;
mod envs;
mod map_gen;
use envs::find_return;

/// A Python module implemented in Rust.
#[pyo3::pymodule]
mod _core {
    use crate::envs;

    /// Returns the version of the compiled extension.
    #[pyo3::pyfunction]
    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}
