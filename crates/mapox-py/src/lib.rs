#[pyo3::pymodule]
mod _core {
    use mapox_core::{
        env::Environment,
        make::{EnvConfig, MultitaskEnvSpec, make_multitask, make_vec},
        policy::{Policy, PolicyError, RandomPolicy},
        render::{RenderApp, open_window},
        timestep::{OBS_CHANNELS, TimeStepMut, TimeStepRef},
        vocab::VocabId,
    };
    use numpy::{
        AllowTypeChange, PyArray1, PyArray2, PyArray4, PyArrayLike1, PyReadonlyArray1,
        PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray4,
    };
    use pyo3::prelude::*;
    use pyo3::{
        Python,
        exceptions::{PyRuntimeError, PyValueError},
    };

    struct PyPolicy {
        callable: Py<PyAny>,
    }

    impl Policy for PyPolicy {
        fn act(
            &mut self,
            timestep: &TimeStepRef<'_>,
            actions: &mut [VocabId],
        ) -> Result<(), PolicyError> {
            Python::attach(|py| {
                let result = (|| -> PyResult<()> {
                    let obs = PyArray4::from_array(py, &timestep.obs);
                    let time = PyArray1::from_array(py, &timestep.time);
                    let terminated = PyArray1::from_array(py, &timestep.terminated);
                    let last_action = PyArray1::from_array(py, &timestep.last_action);
                    let reward = PyArray1::from_array(py, &timestep.reward);
                    let action_mask = PyArray2::from_array(py, &timestep.action_mask);

                    let returned = self.callable.call_method1(
                        py,
                        "act",
                        (obs, time, terminated, last_action, reward, action_mask),
                    )?;
                    let returned: PyArrayLike1<'_, i32, AllowTypeChange> = returned.extract(py)?;
                    let returned = returned.as_array();
                    if returned.len() != actions.len() {
                        return Err(PyValueError::new_err(format!(
                            "policy returned {} actions for {} agents",
                            returned.len(),
                            actions.len()
                        )));
                    }

                    for (action, &returned) in actions.iter_mut().zip(returned.iter()) {
                        *action = VocabId::try_from(returned).map_err(|_| {
                            PyValueError::new_err(format!(
                                "policy returned action id outside VocabId range: {returned}"
                            ))
                        })?;
                    }
                    Ok(())
                })();
                result.map_err(|err| {
                    err.print(py);
                    Box::new(err) as PolicyError
                })
            })
        }

        fn reset(&mut self, num_agents: usize, seed: u64) -> Result<(), PolicyError> {
            Python::attach(|py| {
                if let Err(err) = self.callable.call_method1(py, "reset", (num_agents, seed)) {
                    err.print(py);
                }
            });

            Ok(())
        }
    }

    #[pyfunction]
    #[pyo3(signature=(env, length, seed, policy=None))]
    fn enjoy(
        py: Python<'_>,
        env: &Bound<'_, Env>,
        length: usize,
        seed: u64,
        policy: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let env = env.borrow_mut().inner.take().unwrap(); // inner needs to be Optional to take it
        let policy: Box<dyn Policy> = match policy {
            Some(callable) => Box::new(PyPolicy { callable }),
            None => Box::new(RandomPolicy::new()),
        };

        let app = RenderApp::new(env, length, seed, policy);
        py.detach(move || open_window(app))
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    #[pyclass]
    struct Env {
        inner: Option<Box<dyn Environment + Send + Sync>>,
    }

    impl Env {
        fn env(&self) -> &(dyn Environment + Send + Sync) {
            self.inner.as_deref().unwrap()
        }

        fn env_mut(&mut self) -> &mut (dyn Environment + Send + Sync) {
            self.inner.as_deref_mut().expect("The inner env is missing")
        }
    }

    #[pymethods]
    impl Env {
        #[new]
        fn new(config_json: &str, length: usize, num_envs: usize) -> PyResult<Self> {
            let value: serde_json::Value = serde_json::from_str(config_json)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;

            let inner: Box<dyn Environment + Send + Sync> = if value.is_array() {
                let specs: Vec<MultitaskEnvSpec> = serde_json::from_value(value)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                make_multitask(&specs, length).map_err(|err| PyValueError::new_err(err))?
            } else {
                let config: EnvConfig = serde_json::from_value(value)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                make_vec(&config, length, num_envs)
            };

            Ok(Self { inner: Some(inner) })
        }

        #[getter]
        fn num_agents(&self) -> usize {
            self.env().num_agents()
        }

        #[getter]
        fn num_actions(&self) -> usize {
            self.env().action_spec().num_actions
        }

        /// Expected shape of the `obs` buffer: (num_agents, view_width, view_height, channels).
        #[getter]
        fn observation_shape(&self) -> (usize, usize, usize, usize) {
            let env = self.env();

            let spec = env.observation_spec();
            (
                env.num_agents(),
                spec.width as usize,
                spec.height as usize,
                OBS_CHANNELS,
            )
        }

        #[getter]
        fn obs_symbols(&self) -> Vec<&'static str> {
            self.env().obs_vocab().symbols().to_vec()
        }

        #[getter]
        fn action_symbols(&self) -> Vec<&'static str> {
            self.env().action_vocab().symbols().to_vec()
        }

        #[allow(clippy::too_many_arguments)]
        fn reset(
            &mut self,
            py: Python<'_>,
            seed: u64,
            mut obs: PyReadwriteArray4<'_, VocabId>,
            mut time: PyReadwriteArray1<'_, i32>,
            mut terminated: PyReadwriteArray1<'_, bool>,
            mut last_action: PyReadwriteArray1<'_, VocabId>,
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
            py.detach(|| {
                self.env_mut().reset(seed, &mut timestep);
            });
        }

        #[allow(clippy::too_many_arguments)]
        fn step(
            &mut self,
            py: Python<'_>,
            actions: PyReadonlyArray1<'_, VocabId>,
            mut obs: PyReadwriteArray4<'_, VocabId>,
            mut time: PyReadwriteArray1<'_, i32>,
            mut terminated: PyReadwriteArray1<'_, bool>,
            mut last_action: PyReadwriteArray1<'_, VocabId>,
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

            py.detach(|| {
                self.env_mut().step(actions, &mut timestep);
            });
            Ok(())
        }
    }
}
