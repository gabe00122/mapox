//! Fires: lit from a campfire, burning down, stoked back up with wood, and
//! the light they cast. Light is the shared `lit` layer other systems read:
//! agents see lit ground by night, and spiders keep off it.

use crate::envs::common::Position;

use super::{
    Survival, SurvivalState,
    tiles::{SurvivalObs, within},
};

#[derive(Debug, Clone, Copy)]
pub(super) struct Fire {
    pub(super) position: Position,
    /// Steps left before it goes out.
    pub(super) burn_left: u32,
}

impl Survival {
    /// Sets a fire burning on `position`, for a full `fire_burn_steps`.
    pub(super) fn kindle(&mut self, position: Position) {
        let burn_left = self.config.fire_burn_steps;
        self.set_ground(position, self.fire_tile(burn_left));
        self.state.fires.push(Fire {
            position,
            burn_left,
        });
    }

    /// Stokes the fire on `position` back up to a full `fire_burn_steps`.
    pub(super) fn stoke(&mut self, position: Position) {
        let burn_left = self.config.fire_burn_steps;
        let fire = self
            .state
            .fires
            .iter_mut()
            .find(|fire| fire.position == position)
            .expect("only a burning fire can be stoked");
        fire.burn_left = burn_left;
        self.set_ground(position, self.fire_tile(burn_left));
    }

    /// A fire shows on the map for `burn_left` steps counting this one, so it
    /// is in its last `fire_low_steps` below that many.
    fn fire_tile(&self, burn_left: u32) -> SurvivalObs {
        if burn_left < self.config.fire_low_steps {
            SurvivalObs::TileFireLow
        } else {
            SurvivalObs::TileFire
        }
    }

    /// Every fire burns a step down, going low near the end and out after
    /// it.
    pub(super) fn burn_fires(&mut self) {
        let mut i = 0;
        while i < self.state.fires.len() {
            let Fire {
                position,
                burn_left,
            } = self.state.fires[i];
            if burn_left == 0 {
                self.state.fires.swap_remove(i);
                self.set_ground(position, SurvivalObs::TileEmpty);
                continue;
            }
            self.state.fires[i].burn_left = burn_left - 1;
            let tile = self.fire_tile(burn_left - 1);
            if self.state.base_map[position.idx()] != tile {
                self.set_ground(position, tile);
            }
            i += 1;
        }
    }

    /// Marks the ground each fire lights.
    pub(super) fn light_up(&mut self) {
        let SurvivalState { fires, lit, .. } = &mut self.state;
        let radius = self.config.fire_light_radius;
        lit.fill(false);
        for fire in fires.iter() {
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let offset = Position::new(dx, dy);
                    if !within(offset, radius) {
                        continue;
                    }
                    if let Some(cell) = lit.get_mut((fire.position + offset).idx()) {
                        *cell = true;
                    }
                }
            }
        }
    }
}
