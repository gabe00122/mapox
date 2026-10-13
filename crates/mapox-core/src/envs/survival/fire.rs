use crate::envs::{
    common::Position,
    survival::{SurvivalObs, SurvivalState, needs::Health},
};

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
    pub max_fuel: i32,
    pub burn_rate: i32,
}

pub(super) struct DamageToAbove {
    pub amount: i32,
}

impl SurvivalState {
    pub(super) fn tick_fueled(&mut self) {
        for (active, fire) in self.world.query_mut::<(&mut Active, &mut Fueled)>() {
            if !active.is_active {
                continue;
            }
            fire.fuel = (fire.fuel - fire.burn_rate).max(0);
            if fire.fuel == 0 {
                active.is_active = false;
            }
        }
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

    pub(super) fn tick_damage_to_above(&mut self) {
        for (active, damage_to_above, position) in self
            .world
            .query::<(Option<&Active>, &DamageToAbove, &Position)>()
            .iter()
        {
            if Active::is_on(active)
                && let Some(mut target_health) = self.spatial_index[position.idx()][1]
                    .and_then(|target_id| self.world.get::<&mut Health>(target_id).ok())
            {
                target_health.update(-damage_to_above.amount);
            }
        }
    }
}
