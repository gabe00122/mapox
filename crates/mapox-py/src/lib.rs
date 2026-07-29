/// A Python module implemented in Rust.
#[pyo3::pymodule]
mod _core {
    /// Returns the version of the compiled extension.
    #[pyo3::pyfunction]
    fn version() -> &'static str {
        mapox_core::version()
    }

    /// Opens the viewer window, blocking until it is closed.
    #[pyo3::pyfunction]
    fn run_demo(py: pyo3::Python<'_>) -> pyo3::PyResult<()> {
        // the window loop is long-running and touches no python objects, so
        // hand the GIL back to other threads while it runs.
        py.detach(mapox_core::render::open_window)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
    }
}
