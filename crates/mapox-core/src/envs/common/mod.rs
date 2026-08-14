//! The parts a grid env is built out of, kept apart from any one of them:
//! [`Position`] to say where something is, [`map_gen`] to lay out a map, and
//! [`fov`] to work out how much of it an agent can see.

pub mod fov;
pub mod map_gen;
mod position;

pub use position::Position;
