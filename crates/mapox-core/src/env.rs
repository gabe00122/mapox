use crate::timestep::TimeStepMut;

pub trait Environment {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut);
    fn step(&mut self, actions: &[i32], timestep: &mut TimeStepMut);
}
