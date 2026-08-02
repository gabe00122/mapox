use ndarray::Array2;

use crate::{envs::common::Position, vocab::Vocabulary};

#[derive(Debug, Default, Clone)]
pub struct GridRenderState {
    /// Unified tile ids over the padded map, indexed `[x, y]`.
    pub tilemap: Array2<u8>,
    pub agent_positions: Vec<Position>,
}

#[derive(Debug, Default, Clone)]
pub struct GridRenderSettings {
    pub obs_vocab: Vocabulary,
    pub tile_width: usize,
    pub tile_height: usize,
    pub view_width: usize,
    pub view_height: usize,
}
