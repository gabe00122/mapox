/// A Python module implemented in Rust.
#[pyo3::pymodule]
mod _core {
    /// Returns the version of the compiled extension.
    #[pyo3::pyfunction]
    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}
