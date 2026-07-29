use crate::timestep::TimeStep;
use rand::rngs::SmallRng;

pub trait Environment {
    fn reset(&mut self, seed: u64) -> TimeStep;
    /// `actions` has length `num_agents`, values in `0..action_spec().n`.
    fn step(&mut self, actions: &[i32]) -> TimeStep;
}
