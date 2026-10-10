use hecs::Entity;

use crate::envs::{
    common::{Direction, Position},
    survival::{Slot, SurvivalObs, SurvivalState},
};

#[derive(Debug, Clone, Copy)]
pub(super) enum ItemType {
    Rock,
    Stick,
    Wood,
    CutGrass,
}

impl ItemType {
    pub(super) fn obs(&self) -> SurvivalObs {
        SurvivalObs::Digit0
    }
}

#[derive(Default)]
pub(super) struct Inventory {
    pub hand: Option<ItemType>,
    pub back: Option<ItemType>,
}

impl SurvivalState {
    // fn item_can_put(&mut self, id: Entity) -> bool {
    //     false
    // }

    pub(super) fn item_take(&mut self, id: Entity) {
        let mut target_entity: Option<Entity> = None;

        if let Ok((position, direction, inventory)) = self
            .world
            .query_one::<(&Position, &Direction, &mut Inventory)>(id)
            .get()
        {
            if inventory.hand.is_some() {
                return;
            }

            let target_pos = *position + direction.to_pos();
            target_entity = self.spatial_index[target_pos.idx()][0]; // lower slot

            if let Some(item) = target_entity.and_then(|id| self.world.get::<&ItemType>(id).ok()) {
                inventory.hand = Some(*item);
            } else {
                // clear the target so things that aren't items aren't deleted
                target_entity = None;
            }
        }

        if let Some(target_entity) = target_entity {
            self.despawn(target_entity);
        }
    }
}
