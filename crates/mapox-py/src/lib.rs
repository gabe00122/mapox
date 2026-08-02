#[pyo3::pymodule]
mod _core {
    use mapox_core::{
        env::Environment,
        envs::find_return::FindReturnConfig,
        make::{EnvConfig, make},
        policy::{Policy, PolicyError, PolicyInputs, RandomPolicy},
        render::{RenderApp, open_window},
        timestep::{OBS_CHANNELS, TimeStepMut},
    };
    use numpy::{
        AllowTypeChange, PyArray1, PyArray2, PyArray4, PyArrayLike1, PyReadonlyArray1,
        PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray4,
    };
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;

    #[pyfunction]
    fn version() -> &'static str {
        mapox_core::version()
    }

    /// A python callable driving the render loop's agents: called once per
    /// env step with copies of the timestep arrays, under a freshly attached
    /// interpreter (`run_demo` detaches for the window's whole lifetime).
    struct PyPolicy {
        callable: Py<PyAny>,
    }

    impl Policy for PyPolicy {
        fn act(
            &mut self,
            inputs: &PolicyInputs<'_>,
            actions: &mut [i32],
        ) -> Result<(), PolicyError> {
            Python::attach(|py| {
                let result = (|| -> PyResult<()> {
                    let obs = PyArray4::from_array(py, &inputs.obs);
                    let reward = PyArray1::from_array(py, &inputs.reward);
                    let terminated = PyArray1::from_array(py, &inputs.terminated);
                    let action_mask = PyArray2::from_array(py, &inputs.action_mask);

                    let returned = self
                        .callable
                        .call1(py, (obs, reward, terminated, action_mask))?;
                    let returned: PyArrayLike1<'_, i32, AllowTypeChange> = returned.extract(py)?;
                    let returned = returned.as_array();
                    if returned.len() != actions.len() {
                        return Err(PyValueError::new_err(format!(
                            "policy returned {} actions for {} agents",
                            returned.len(),
                            actions.len()
                        )));
                    }
                    // zip, not copy_from_slice: dtype coercion can hand back
                    // a non-contiguous array
                    for (action, &returned) in actions.iter_mut().zip(returned.iter()) {
                        *action = returned;
                    }
                    Ok(())
                })();
                result.map_err(|err| {
                    err.print(py);
                    Box::new(err) as PolicyError
                })
            })
        }
    }

    /// Opens the viewer window and blocks until it closes. `policy` drives
    /// every agent (falling back to a random policy when omitted, or after
    /// the callable raises); the keyboard overrides the focused agent.
    #[pyfunction]
    #[pyo3(signature = (config_json=None, policy=None))]
    fn run_demo(
        py: Python<'_>,
        config_json: Option<&str>,
        policy: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let config = match config_json {
            Some(json) => serde_json::from_str::<EnvConfig>(json)
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
            None => EnvConfig::FindReturn(FindReturnConfig::default()),
        };
        let env = make(&config);
        let policy: Box<dyn Policy> = match policy {
            Some(callable) => Box::new(PyPolicy { callable }),
            None => Box::new(RandomPolicy::new(0)),
        };

        let app = RenderApp::new(env, policy);
        py.detach(move || open_window(app))
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
            mut obs: PyReadwriteArray4<'_, u8>,
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
            mut obs: PyReadwriteArray4<'_, u8>,
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
