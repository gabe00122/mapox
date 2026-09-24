use serde_json::Value;

use crate::render::env::{GridRenderSettings, GridRenderState};
use crate::spec::{ActionSpec, ObservationSpec};
use crate::timestep::TimeStepMut;
use crate::vocab::{VocabId, Vocabulary};

pub trait Environment: Send + Sync {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut);
    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut);

    fn observation_spec(&self) -> ObservationSpec;
    fn action_spec(&self) -> ActionSpec;

    fn num_agents(&self) -> usize;
    fn obs_vocab(&self) -> &Vocabulary;
    fn action_vocab(&self) -> &Vocabulary;

    fn get_render_settings(&self) -> GridRenderSettings;
    fn render_state_into(&self, grid_render_state: &mut GridRenderState);

    fn num_tasks(&self) -> usize;
    /// Names of this env's tasks in task-id order. Single-task envs have no
    /// subtasks and return an empty list.
    fn task_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn set_enjoy_mode(&mut self, _task_num: Option<usize>) {}

    fn consume_metrics(&mut self) -> Value;
}

pub(crate) fn mean_metrics(mut metrics: impl ExactSizeIterator<Item = Value>) -> Value {
    fn add(total: &mut Value, value: Value) {
        match (total, value) {
            (Value::Object(total), Value::Object(value)) => {
                assert_eq!(total.len(), value.len(), "metric structures must match");
                for (key, value) in value {
                    add(total.get_mut(&key).expect("metric keys must match"), value);
                }
            }
            (total @ Value::Number(_), Value::Number(value)) => {
                *total = Value::from(total.as_f64().unwrap() + value.as_f64().unwrap());
            }
            _ => panic!("metrics must contain only objects and numbers"),
        }
    }

    fn divide(value: &mut Value, count: f64) {
        match value {
            Value::Object(object) => {
                for value in object.values_mut() {
                    divide(value, count);
                }
            }
            Value::Number(number) => *value = Value::from(number.as_f64().unwrap() / count),
            _ => panic!("metrics must contain only objects and numbers"),
        }
    }

    let count = metrics.len() as f64;
    let mut total = metrics.next().expect("metric groups must not be empty");
    for value in metrics {
        add(&mut total, value);
    }
    divide(&mut total, count);
    total
}
