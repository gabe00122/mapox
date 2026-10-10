use hecs::Entity;

use crate::envs::{
    common::{Direction, Position},
    survival::{
        Slot, SurvivalObs, SurvivalState, needs::Hunger, prototypes::Prototype, refine::RefineType,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ItemType {
    Rock,
    Stick,
    Wood,
    CutGrass,
    StoneAxe,
    CampfireKit,
    Berry,
    CookedBerry,
}

// requirement, requirement, product
const RECIPES: &[(ItemType, ItemType, ItemType)] = &[
    (ItemType::Rock, ItemType::Stick, ItemType::StoneAxe),
    (ItemType::Wood, ItemType::CutGrass, ItemType::CampfireKit),
];

impl ItemType {
    pub(super) fn obs(&self) -> SurvivalObs {
        match self {
            ItemType::Rock => SurvivalObs::ItemRock,
            ItemType::Stick => SurvivalObs::ItemStick,
            ItemType::Wood => SurvivalObs::ItemWood,
            ItemType::CutGrass => SurvivalObs::ItemCutGrass,
            ItemType::StoneAxe => SurvivalObs::ItemStoneAxe,
            _ => SurvivalObs::Digit0,
        }
    }

    fn hunger_restore(&self) -> Option<u8> {
        match self {
            ItemType::Berry => Some(10),
            ItemType::CookedBerry => Some(40),
            _ => None,
        }
    }
}

// Inventory could be extended to act like a stack with n slots
// instead of swap it would be n swap actions for swaping 0 with the nth position
#[derive(Default)]
pub(super) struct Inventory {
    pub hand: Option<ItemType>,
    pub back: Option<ItemType>,
}

impl SurvivalState {
    pub(super) fn item_refine(&mut self, id: Entity) {
        let mut query = self
            .world
            .query_one::<(&Position, &Direction, &mut Inventory)>(id);
        let Ok((&position, direction, inventory)) = query.get() else {
            return;
        };
        let Some(hand) = inventory.hand else {
            return;
        };

        let target_pos = position + direction.to_pos();
        let Some(output) =
            self.spatial_index[target_pos.idx()][0] // lower slot
                .and_then(|target| self.world.get::<&RefineType>(target).ok())
                .and_then(|refine_type| refine_type.refine(hand))
        else {
            return;
        };

        inventory.hand = Some(output);
    }

    pub(crate) fn item_use(&mut self, id: Entity) {
        let Ok((&position, direction, inventory)) = self
            .world
            .query_one_mut::<(&Position, &Direction, &Inventory)>(id)
        else {
            return;
        };

        let Some(item) = inventory.hand else {
            return;
        };

        if let Some(hunger_restore) = item.hunger_restore() {
            if let Ok((inventory, hunger)) = self
                .world
                .query_one_mut::<(&mut Inventory, &mut Hunger)>(id)
            {
                hunger.add(hunger_restore);
                inventory.hand = None;
            }
            return;
        }

        let target_pos = position + direction.to_pos();

        match item {
            ItemType::CampfireKit => {
                if self.is_blocked(target_pos, Slot::Lower) {
                    return;
                }

                if let Ok(inventory) = self.world.query_one_mut::<&mut Inventory>(id) {
                    inventory.hand = None;
                }
                self.spawn_prototype(Prototype::Fire, target_pos);
            }
            _ => {}
        }
    }

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
        let Ok((&position, direction, inventory)) = self
            .world
            .query_one_mut::<(&Position, &Direction, &Inventory)>(id)
        else {
            return;
        };

        let Some(item) = inventory.hand else {
            return;
        };

        let target_pos = position + direction.to_pos();
        if self.is_blocked(target_pos, Slot::Lower) {
            return;
        }

        if let Ok(inventory) = self.world.query_one_mut::<&mut Inventory>(id) {
            inventory.hand = inventory.back.take();
        }
        self.spawn_prototype(Prototype::Item(item), target_pos);
    }

    pub(super) fn item_take(&mut self, id: Entity) {
        let mut target_entity: Option<Entity> = None;

        if let Ok((position, direction, inventory)) = self
            .world
            .query_one::<(&Position, &Direction, &mut Inventory)>(id)
            .get()
        {
            if inventory.hand.is_some() && inventory.back.is_some() {
                return;
            }

            let target_pos = *position + direction.to_pos();
            target_entity = self.spatial_index[target_pos.idx()][0]; // lower slot

            if let Some(item) = target_entity.and_then(|id| self.world.get::<&ItemType>(id).ok()) {
                if inventory.hand.is_some() {
                    inventory.back = inventory.hand.take();
                }
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
