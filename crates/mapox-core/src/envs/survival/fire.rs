use hecs::Entity;

use crate::envs::survival::SurvivalState;

pub(super) struct Fire {
    pub fuel: i32,
}

impl SurvivalState {
    pub(super) fn tick_fire(&mut self) {
        let mut burnt_out = Vec::new();

        for (entity, fire) in self.world.query_mut::<(Entity, &mut Fire)>() {
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
