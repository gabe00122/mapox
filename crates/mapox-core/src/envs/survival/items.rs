use hecs::Entity;

use crate::envs::{
    common::{Direction, Position},
    survival::{Slot, SurvivalObs, SurvivalState, prototypes::Prototype},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ItemType {
    Rock,
    Stick,
    Wood,
    CutGrass,
    StoneAxe,
}

// requirement, requirement, product
const RECIPES: &[(ItemType, ItemType, ItemType)] =
    &[(ItemType::Rock, ItemType::Stick, ItemType::StoneAxe)];

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
    pub(crate) fn item_craft(&mut self, id: Entity) {
        let Ok(inventory) = self.world.query_one_mut::<&mut Inventory>(id) else {
            return;
        };

        let (Some(hand), Some(back)) = (inventory.hand, inventory.back) else {
            return;
        };

        let Some(&(_, _, product)) = RECIPES
            .iter()
            .find(|&&(a, b, _)| (a == hand && b == back) || (a == back && b == hand))
        else {
            return;
        };

        inventory.hand = Some(product);
        inventory.back = None;
    }

    pub(super) fn item_swap(&mut self, id: Entity) {
        if let Ok(inventory) = self.world.query_one_mut::<&mut Inventory>(id) {
            std::mem::swap(&mut inventory.hand, &mut inventory.back);
        }
    }

    pub(super) fn item_put(&mut self, id: Entity) {
        let Ok((&position, direction, inventory)) =
            self.world
                .query_one_mut::<(&Position, &Direction, &mut Inventory)>(id)
        else {
            return;
        };

        let target_pos = position + direction.to_pos();
        let blocked = self.spatial_index[target_pos.idx()][0].is_some() // lower slot
            || self.tiles[target_pos.idx()].move_blocked();
        if blocked {
            return;
        }

        if let Some(item) = inventory.hand.take() {
            self.spawn_prototype(Prototype::Item(item), target_pos);
        }
    }

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
