//! The seasons: summer, then winter for the episode's last
//! `winter_length` steps. Winter kills the plants and freezes the water, and
//! its cold is what drains temperature (see [`Survival::tick_stats`]).

use crate::envs::common::Position;

use super::Survival;

impl Survival {
    /// The first step of winter.
    fn winter_start(&self) -> usize {
        self.length
            .saturating_sub(self.config.winter_length as usize)
    }

    pub(super) fn is_winter(&self, time: usize) -> bool {
        time >= self.winter_start()
    }

    /// Winter sets in on its first step, and is there from reset if the
    /// episode is all winter.
    pub(super) fn tick_season(&mut self, time: usize) {
        if time == self.winter_start() {
            self.freeze();
        }
    }

    /// The cold takes hold, for the rest of the episode: every tile it
    /// changes changes (bushes die, grass dies back, water freezes), and
    /// nothing picked grows back.
    fn freeze(&mut self) {
        self.state.regrowing.clear();
        let changed: Vec<_> = self
            .state
            .base_map
            .indexed_iter()
            .filter_map(|((x, y), tile)| {
                tile.frozen()
                    .map(|frozen| (Position::new(x as i32, y as i32), frozen))
            })
            .collect();
        for (position, tile) in changed {
            self.set_ground(position, tile);
        }
    }
}
