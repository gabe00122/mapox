//! Fires: lit from a campfire, burning down, stoked back up with fuel, and
//! the light they cast; and torches, burning down in hand and lighting the
//! ground around whoever holds them. Light is the shared `lit` layer other
//! systems read: agents see lit ground by night, and spiders keep off it.
//! Only a fire's light warms.

use ndarray::Array2;

use crate::envs::common::Position;

use super::{
    Survival, SurvivalState,
    items::Item,
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

    /// Stokes the fire on `position` with `fuel` more steps of burning, up
    /// to a full `fire_burn_steps`.
    pub(super) fn stoke(&mut self, position: Position, fuel: u32) {
        let full = self.config.fire_burn_steps;
        let fire = self
            .state
            .fires
            .iter_mut()
            .find(|fire| fire.position == position)
            .expect("only a burning fire can be stoked");
        fire.burn_left = (fire.burn_left + fuel).min(full);
        let burn_left = fire.burn_left;
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

    /// Whether `position` is warmed by a fire: it is if the fire's light
    /// reaches it.
    pub(super) fn by_fire(&self, position: Position) -> bool {
        let radius = self.config.fire_light_radius;
        self.state
            .fires
            .iter()
            .any(|fire| within(position - fire.position, radius))
    }

    /// Torches held in hand burn down a step, and one burnt out is gone.
    pub(super) fn burn_torches(&mut self) {
        let burn_steps = self.config.torch_burn_steps;
        for agent in &mut self.state.agents {
            if let Some(Item::Torch { burnt }) = agent.hands {
                let burnt = burnt + 1;
                agent.hands = (burnt < burn_steps).then_some(Item::Torch { burnt });
            }
        }
    }

    /// Marks the ground each fire lights, and each torch held in hand.
    pub(super) fn light_up(&mut self) {
        let SurvivalState {
            fires, agents, lit, ..
        } = &mut self.state;
        lit.fill(false);
        for fire in fires.iter() {
            light_around(lit, fire.position, self.config.fire_light_radius);
        }
        for agent in agents.iter() {
            if let Some(Item::Torch { .. }) = agent.hands {
                light_around(lit, agent.position, self.config.torch_light_radius);
            }
        }
    }
}

fn light_around(lit: &mut Array2<bool>, center: Position, radius: i32) {
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let offset = Position::new(dx, dy);
            if !within(offset, radius) {
                continue;
            }
            if let Some(cell) = lit.get_mut((center + offset).idx()) {
                *cell = true;
            }
        }
    }
}
