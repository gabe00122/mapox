#[pyo3::pymodule]
mod _core {
    use mapox_core::{
        env::Environment,
        make::{EnvConfig, make},
        timestep::{OBS_CHANNELS, TimeStepMut},
    };
    use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray4};
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;

    #[pyfunction]
    fn version() -> &'static str {
        mapox_core::version()
    }

    #[pyfunction]
    fn run_demo(py: Python<'_>) -> PyResult<()> {
        py.detach(mapox_core::render::open_window)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    #[pyclass]
    struct Env {
        inner: Box<dyn Environment + Send + Sync>,
    }

    #[pymethods]
    impl Env {
        #[new]
        fn new(config_json: &str) -> PyResult<Self> {
            let config: EnvConfig = serde_json::from_str(config_json)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            Ok(Self {
                inner: make(&config),
            })
        }

        #[getter]
        fn num_agents(&self) -> usize {
            self.inner.num_agents()
        }

        #[getter]
        fn num_actions(&self) -> usize {
            self.inner.action_spec().num_actions
        }

        /// Expected shape of the `obs` buffer: (num_agents, view_width, view_height, channels).
        #[getter]
        fn observation_shape(&self) -> (usize, usize, usize, usize) {
            let spec = self.inner.observation_spec();
            (
                self.inner.num_agents(),
                spec.width as usize,
                spec.height as usize,
                OBS_CHANNELS,
            )
        }

        #[getter]
        fn obs_symbols(&self) -> Vec<&'static str> {
            self.inner.obs_vocab().symbols().to_vec()
        }

        #[getter]
        fn action_symbols(&self) -> Vec<&'static str> {
            self.inner.action_vocab().symbols().to_vec()
        }

        #[allow(clippy::too_many_arguments)]
        fn reset(
            &mut self,
            seed: u64,
            mut obs: PyReadwriteArray4<'_, i8>,
            mut time: PyReadwriteArray1<'_, i32>,
            mut terminated: PyReadwriteArray1<'_, bool>,
            mut last_action: PyReadwriteArray1<'_, i32>,
            mut reward: PyReadwriteArray1<'_, f32>,
            mut action_mask: PyReadwriteArray2<'_, bool>,
            mut task_ids: PyReadwriteArray1<'_, i32>,
        ) {
            let mut timestep = TimeStepMut {
                obs: obs.as_array_mut(),
                time: time.as_array_mut(),
                terminated: terminated.as_array_mut(),
                last_action: last_action.as_array_mut(),
                reward: reward.as_array_mut(),
                action_mask: action_mask.as_array_mut(),
                task_ids: task_ids.as_array_mut(),
            };
            self.inner.reset(seed, &mut timestep);
        }

        #[allow(clippy::too_many_arguments)]
        fn step(
            &mut self,
            actions: PyReadonlyArray1<'_, i32>,
            mut obs: PyReadwriteArray4<'_, i8>,
            mut time: PyReadwriteArray1<'_, i32>,
            mut terminated: PyReadwriteArray1<'_, bool>,
            mut last_action: PyReadwriteArray1<'_, i32>,
            mut reward: PyReadwriteArray1<'_, f32>,
            mut action_mask: PyReadwriteArray2<'_, bool>,
            mut task_ids: PyReadwriteArray1<'_, i32>,
        ) -> PyResult<()> {
            let actions = actions.as_slice()?;
            let mut timestep = TimeStepMut {
                obs: obs.as_array_mut(),
                time: time.as_array_mut(),
                terminated: terminated.as_array_mut(),
                last_action: last_action.as_array_mut(),
                reward: reward.as_array_mut(),
                action_mask: action_mask.as_array_mut(),
                task_ids: task_ids.as_array_mut(),
            };
            self.inner.step(actions, &mut timestep);
            Ok(())
        }
    }
}
