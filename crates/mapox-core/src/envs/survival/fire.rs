use hecs::Entity;

use crate::envs::survival::SurvivalState;

// fire could be replaced by a Fuel and Active component which could be reused for things like a furness and lights
// Active could be reused to make plants regenerate in the future too

pub(super) struct Fueled {
    pub fuel: i32,
}

impl SurvivalState {
    pub(super) fn tick_fueled(&mut self) {
        let mut burnt_out = Vec::new();

        for (entity, fire) in self.world.query_mut::<(Entity, &mut Fueled)>() {
            fire.fuel -= 1;
            if fire.fuel <= 0 {
                burnt_out.push(entity);
            }
        }

        for id in burnt_out {
            self.despawn(id);
        }
    }
}
