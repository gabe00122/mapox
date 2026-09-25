pub mod fov;
pub mod map_gen;
mod position;
pub mod vocab_enum;

pub use position::Position;

/// Rows of UI band every env appends to its field of view. An env's
/// `view_height` config is the field of view alone; observations are
/// `view_height + UI_HEIGHT` rows tall.
pub const UI_HEIGHT: usize = 2;
