#[pyo3::pymodule]
mod _core {
    #[pyo3::pyfunction]
    fn version() -> &'static str {
        mapox_core::version()
    }

    #[pyo3::pyfunction]
    fn run_demo(py: pyo3::Python<'_>) -> pyo3::PyResult<()> {
        py.detach(mapox_core::render::open_window)
            .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))
    }
}
