//! What grows: berry bushes picked bare and fruiting again.

use crate::envs::common::Position;

use super::{Survival, tiles::SurvivalObs};

impl Survival {
    /// Strips a ripe bush of its berries; it fruits again `bush_regrow_steps`
    /// on.
    pub(super) fn pick_bush(&mut self, bush: Position) {
        self.set_ground(bush, SurvivalObs::TileBush);
        let ripe = self.state.time + self.config.bush_regrow_steps as usize;
        self.state.regrowing.push_back((bush, ripe));
    }

    /// Bushes picked long enough ago fruit again, if they are still there:
    /// an axe may have cleared one meanwhile.
    pub(super) fn regrow_bushes(&mut self) {
        while let Some(&(bush, ripe)) = self.state.regrowing.front() {
            if ripe > self.state.time {
                break;
            }
            self.state.regrowing.pop_front();
            if self.state.base_map[bush.idx()] == SurvivalObs::TileBush {
                self.set_ground(bush, SurvivalObs::TileBerryBush);
            }
        }
    }
}
