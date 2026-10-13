use hecs::Entity;

use crate::envs::survival::SurvivalState;

const TICK_HUNGER_EVERY: usize = 2;

#[derive(Debug, Clone, Copy)]
pub(super) struct Health {
    pub amount: i32,
    pub changed: i32,
    pub max: i32,
}

impl Health {
    pub fn update(&mut self, value: i32) {
        let before = self.amount;
        self.amount = (self.amount + value).clamp(0, self.max);
        self.changed += self.amount - before;
    }
}

impl Default for Health {
    fn default() -> Self {
        Self {
            amount: 99,
            max: 99,
            changed: 0,
        }
    }
}

pub(super) struct Hunger {
    pub amount: i32,
    pub changed: i32,
    pub max: i32,
}

impl Default for Hunger {
    fn default() -> Self {
        Self {
            amount: 99,
            max: 99,
            changed: 0,
        }
    }
}

impl Hunger {
    pub fn update(&mut self, value: i32) {
        let before = self.amount;
        self.amount = (self.amount + value).clamp(0, self.max);
        self.changed += self.amount - before;
    }
}

impl SurvivalState {
    pub(super) fn reset_hunger_changed(&mut self) {
        for hunger in self.world.query_mut::<&mut Hunger>() {
            hunger.changed = 0;
        }
    }

    pub(super) fn reset_health_changed(&mut self) {
        for health in self.world.query_mut::<&mut Health>() {
            health.changed = 0;
        }
    }

    pub(super) fn tick_hunger(&mut self) {
        if self.time % TICK_HUNGER_EVERY == TICK_HUNGER_EVERY - 1 {
            for hunger in self.world.query_mut::<&mut Hunger>() {
                hunger.update(-1);
            }
        }
    }

    pub(super) fn tick_starvation(&mut self) {
        for (hunger, health) in self.world.query_mut::<(&Hunger, &mut Health)>() {
            if hunger.amount == 0 {
                health.update(-1);
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
