use hecs::Entity;

use crate::envs::survival::SurvivalState;

const TICK_HUNGER_EVERY: usize = 2;

#[derive(Debug, Clone, Copy)]
pub(super) struct Health {
    pub amount: u8,
    pub max: u8,
}

impl Health {
    pub fn add(&mut self, rhs: u8) {
        self.amount = self.amount.saturating_add(rhs).min(self.max);
    }
}

impl Default for Health {
    fn default() -> Self {
        Self {
            amount: 99,
            max: 99,
        }
    }
}

pub(super) struct Hunger {
    pub amount: u8,
    pub max: u8,
}

impl Default for Hunger {
    fn default() -> Self {
        Self {
            amount: 99,
            max: 99,
        }
    }
}

impl Hunger {
    pub fn add(&mut self, rhs: u8) {
        self.amount = self.amount.saturating_add(rhs).min(self.max);
    }
}

impl SurvivalState {
    pub(super) fn tick_hunger(&mut self) {
        if self.time % TICK_HUNGER_EVERY == TICK_HUNGER_EVERY - 1 {
            for hunger in self.world.query_mut::<&mut Hunger>() {
                hunger.amount = hunger.amount.saturating_sub(1);
            }
        }
    }

    pub(super) fn tick_starvation(&mut self) {
        for (hunger, health) in self.world.query_mut::<(&Hunger, &mut Health)>() {
            if hunger.amount == 0 {
                health.amount = health.amount.saturating_sub(1);
            }
        }
    }

    pub(super) fn tick_death(&mut self) {
        let mut dead = Vec::new();

        for (entity, health) in self.world.query_mut::<(Entity, &Health)>() {
            if health.amount == 0 {
                dead.push(entity);
            }
        }

        for id in dead {
            self.despawn(id);
        }
    }
}
