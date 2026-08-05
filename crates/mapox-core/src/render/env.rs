use ndarray::Array2;

use crate::{
    envs::common::Position,
    vocab::{VocabId, Vocabulary},
};

#[derive(Debug, Default, Clone)]
pub struct GridRenderState {
    pub tilemap: Array2<VocabId>,
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
