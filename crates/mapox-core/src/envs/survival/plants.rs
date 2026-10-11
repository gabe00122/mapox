use crate::envs::survival::{ItemType, Survival, fire::Active};

pub(super) struct ItemProducer {
    pub item: ItemType,
}

pub(super) struct ReactiveAfter {
    pub reset_delay: i32,
    pub just_active: bool,
    pub time_left: i32,
}

impl ReactiveAfter {
    pub(super) fn new(delay: i32) -> Self {
        Self {
            reset_delay: delay,
            just_active: true,
            time_left: 0,
        }
    }
}

pub(super) struct Chopable {
    pub item: Option<ItemType>, // the resource dropped when chopped if any
}

impl Survival {
    fn tick_reactivate_after(&mut self) {
        for (active, reactive_after) in self
            .state
            .world
            .query_mut::<(&mut Active, &mut ReactiveAfter)>()
        {
            reactive_after.reset_delay = (reactive_after.reset_delay - 1).max(0);
            if reactive_after.reset_delay == 0 {
                active.is_active = true;
            } else if !active.is_active && reactive_after.just_active {
                reactive_after.time_left = reactive_after.reset_delay;
            }

            reactive_after.just_active = active.is_active;
        }
    }
}
