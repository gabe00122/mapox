use crate::envs::survival::{SurvivalObs, SurvivalState};

// fire could be replaced by a Fuel and Active component which could be reused for things like a furness and lights
// Active could be reused to make plants regenerate in the future too

pub(super) struct ActiveObs {
    pub enabled: SurvivalObs,
    pub disabled: SurvivalObs,
}

pub(super) struct Active {
    pub is_active: bool,
}

impl Active {
    /// Whether an entity is on: those without an `Active` always are.
    pub fn is_on(active: Option<&Active>) -> bool {
        active.is_none_or(|active| active.is_active)
    }
}

impl Default for Active {
    fn default() -> Self {
        Self { is_active: true }
    }
}

pub(super) struct Fueled {
    pub fuel: i32,
    pub burn_rate: i32,
}

impl SurvivalState {
    pub(super) fn tick_fueled(&mut self) {
        // let mut burnt_out = Vec::new();

        for (active, fire) in self.world.query_mut::<(&mut Active, &mut Fueled)>() {
            fire.fuel -= fire.burn_rate;
            if fire.fuel <= 0 {
                active.is_active = false;
            }
        }

        // for id in burnt_out {
        //     self.despawn(id);
        // }
    }

    pub(super) fn tick_active_obs(&mut self) {
        for (active, active_obs, obs) in self
            .world
            .query_mut::<(&Active, &ActiveObs, &mut SurvivalObs)>()
        {
            *obs = if active.is_active {
                active_obs.enabled
            } else {
                active_obs.disabled
            }
        }
    }
}
