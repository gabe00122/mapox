use ndarray::{Array2, ArrayView3};

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
    pub ui_height: usize,
}

impl GridRenderSettings {
    /// Height of the field-of-view band, excluding the synthetic UI band rows.
    pub fn fov_height(&self) -> usize {
        self.view_height.saturating_sub(self.ui_height)
    }
}

/// Which map tiles at least one agent can currently see, read back out of
/// the encoded observations: a tile counts as seen when an agent's view
/// window covers it and the observation there is not the `mask` tile. A
/// vocab without a mask symbol passes `None` for `mask`, and the whole
/// window counts — with nothing able to hide a tile, line of sight is the
/// window itself.
///
/// `obs` is the tile channel of the timestep's observations, sliced down to
/// `(num_agents, view_width, view_height)`; its first `fov_height` rows are
/// the view window centred on the agent and the rest are the UI band, which
/// maps to no map tile. `positions` are the agents' map coordinates, as in
/// [`GridRenderState::agent_positions`].
pub fn visible_tiles(
    settings: &GridRenderSettings,
    positions: &[Position],
    obs: ArrayView3<'_, VocabId>,
    mask: Option<VocabId>,
) -> Array2<bool> {
    let mut seen = Array2::from_elem((settings.tile_width, settings.tile_height), false);

    let half_width = settings.view_width as i32 / 2;
    let half_height = settings.fov_height() as i32 / 2;

    for (agent, position) in positions.iter().enumerate() {
        for view_x in 0..settings.view_width {
            // an even window runs [-half, half - 1], matching the FOV encoder
            let x = position.x - half_width + view_x as i32;
            if x < 0 || x >= settings.tile_width as i32 {
                continue;
            }

            for view_y in 0..settings.fov_height() {
                let y = position.y - half_height + view_y as i32;
                if y < 0 || y >= settings.tile_height as i32 {
                    continue;
                }

                if mask.is_none_or(|mask| obs[[agent, view_x, view_y]] != mask) {
                    seen[[x as usize, y as usize]] = true;
                }
            }
        }
    }

    seen
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    const MASK: VocabId = 0;
    const TILE: VocabId = 1;

    /// A 7x5 map with one agent dead centre, a 3x3 view window, and no UI
    /// band: the window covers x in 2..=4 and y in 1..=3.
    fn setup() -> (GridRenderSettings, Vec<Position>, Array3<VocabId>) {
        let settings = GridRenderSettings {
            tile_width: 7,
            tile_height: 5,
            view_width: 3,
            view_height: 3,
            ui_height: 0,
            ..Default::default()
        };
        let positions = vec![Position::new(3, 2)];
        let obs = Array3::from_elem((1, 3, 3), TILE);
        (settings, positions, obs)
    }

    #[test]
    fn without_a_mask_the_whole_window_is_seen() {
        let (settings, positions, obs) = setup();
        let seen = visible_tiles(&settings, &positions, obs.view(), None);

        for x in 0..7 {
            for y in 0..5 {
                let in_window = (2..=4).contains(&x) && (1..=3).contains(&y);
                assert_eq!(seen[[x, y]], in_window, "tile ({x}, {y})");
            }
        }
    }

    #[test]
    fn masked_observation_cells_are_not_seen() {
        let (settings, positions, mut obs) = setup();
        // the centre of the window is the agent's own tile, (3, 2) on the map
        obs[[0, 1, 1]] = MASK;

        let seen = visible_tiles(&settings, &positions, obs.view(), Some(MASK));

        assert!(!seen[[3, 2]]);
        assert_eq!(seen.iter().filter(|&&seen| seen).count(), 8);
    }

    /// The UI band holds no map tiles; its rows must not light anything up
    /// even when they contain non-mask values.
    #[test]
    fn the_ui_band_lights_nothing() {
        let (mut settings, positions, _) = setup();
        settings.view_height = 4;
        settings.ui_height = 1;
        let obs = Array3::from_elem((1, 3, 4), TILE);

        let seen = visible_tiles(&settings, &positions, obs.view(), Some(MASK));

        for y in 0..5 {
            assert_eq!(seen[[3, y]], (1..=3).contains(&y), "column tile y={y}");
        }
    }

    /// An agent in the corner has most of its window off the map; only the
    /// in-bounds slice is marked, and nothing panics.
    #[test]
    fn the_window_clips_at_the_map_edge() {
        let (settings, _, obs) = setup();
        let positions = vec![Position::new(0, 0)];

        let seen = visible_tiles(&settings, &positions, obs.view(), Some(MASK));

        for x in 0..7 {
            for y in 0..5 {
                assert_eq!(seen[[x, y]], x <= 1 && y <= 1, "tile ({x}, {y})");
            }
        }
    }

    /// A tile hidden from one agent but visible to another is still seen:
    /// the union of every agent's line of sight is what stays bright.
    #[test]
    fn any_agent_seeing_a_tile_is_enough() {
        let (settings, _, _) = setup();
        let positions = vec![Position::new(3, 2), Position::new(6, 4)];
        let mut obs = Array3::from_elem((2, 3, 3), MASK);
        obs[[1, 1, 1]] = TILE; // the second agent's own tile, map (6, 4)

        let seen = visible_tiles(&settings, &positions, obs.view(), Some(MASK));

        assert_eq!(seen.iter().filter(|&&seen| seen).count(), 1);
        assert!(seen[[6, 4]]);
    }
}
