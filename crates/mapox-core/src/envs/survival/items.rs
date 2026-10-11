use hecs::Entity;

use crate::envs::{
    common::{Direction, Position},
    survival::{
        Slot, SurvivalObs, SurvivalState,
        fire::{Active, Fueled},
        needs::Hunger,
        plants::{Chopable, ItemProducer},
        prototypes::Prototype,
        refine::RefineType,
        world::Agent,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ItemType {
    Rock,
    Stick,
    Wood,
    CutGrass,
    StoneAxe,
    StonePixaxe,
    CampfireKit,
    Berry,
    CookedBerry,
}

// requirement, requirement, product
const RECIPES: &[(ItemType, ItemType, ItemType)] = &[
    (ItemType::Rock, ItemType::Stick, ItemType::StoneAxe),
    (ItemType::Wood, ItemType::CutGrass, ItemType::CampfireKit),
    (ItemType::StoneAxe, ItemType::Rock, ItemType::StonePixaxe),
];

impl ItemType {
    pub(super) fn obs(&self) -> SurvivalObs {
        match self {
            ItemType::Rock => SurvivalObs::ItemRock,
            ItemType::Stick => SurvivalObs::ItemStick,
            ItemType::Wood => SurvivalObs::ItemWood,
            ItemType::CutGrass => SurvivalObs::ItemCutGrass,
            ItemType::StoneAxe => SurvivalObs::ItemStoneAxe,
            ItemType::CampfireKit => SurvivalObs::ItemCampfireKit,
            ItemType::Berry => SurvivalObs::ItemBerry,
            ItemType::CookedBerry => SurvivalObs::ItemCookedBerry,
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

    fn fuel_value(&self) -> Option<u8> {
        match self {
            ItemType::Stick | ItemType::CutGrass => Some(50),
            ItemType::Wood => Some(200),
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
                .filter(|&target| Active::is_on(self.world.get::<&Active>(target).ok().as_deref()))
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
                //temporary reward
                if let Ok(agent) = self.world.query_one_mut::<&mut Agent>(id) {
                    agent.reward += 0.1;
                }

                if let Ok(inventory) = self.world.query_one_mut::<&mut Inventory>(id) {
                    inventory.hand = None;
                }
                self.spawn_prototype(Prototype::Fire, target_pos);
            }
            ItemType::StoneAxe => {
                // the lower slot also holds Slot::Both entities like trees
                let Some((target, drop)) =
                    self.spatial_index[target_pos.idx()][0].and_then(|target| {
                        let chopable = self.world.query_one_mut::<&Chopable>(target).ok()?;
                        Some((target, chopable.item))
                    })
                else {
                    return;
                };

                self.despawn(target);
                if let Some(drop) = drop {
                    self.spawn_prototype(Prototype::Item(drop), target_pos);
                }
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

        // putting fuel into a fueled thing (e.g. a fire) burns it up, relighting it if burnt out
        if let Some(fuel_value) = item.fuel_value()
            && let Some(target) = self.spatial_index[target_pos.idx()][0] // lower slot
            && let Ok((active, fueled)) = self
                .world
                .query_one_mut::<(&mut Active, &mut Fueled)>(target)
        {
            fueled.fuel = (fueled.fuel + fuel_value as i32).min(fueled.max_fuel);
            active.is_active = true;
            if let Ok(inventory) = self.world.query_one_mut::<&mut Inventory>(id) {
                inventory.hand = inventory.back.take();
            }
            return;
        }

        if self.is_blocked(target_pos, Slot::Lower) {
            return;
        }

        if let Ok(inventory) = self.world.query_one_mut::<&mut Inventory>(id) {
            inventory.hand = inventory.back.take();
        }
        self.spawn_prototype(Prototype::Item(item), target_pos);
    }

    pub(super) fn item_take(&mut self, id: Entity) {
        let Ok((&position, direction, inventory)) = self
            .world
            .query_one_mut::<(&Position, &Direction, &Inventory)>(id)
        else {
            return;
        };

        if inventory.hand.is_some() && inventory.back.is_some() {
            return;
        }

        let target_pos = position + direction.to_pos();
        let Some(target) = self.spatial_index[target_pos.idx()][0] else {
            return; // lower slot
        };

        let item = if let Ok(&item) = self.world.query_one_mut::<&ItemType>(target) {
            self.despawn(target);
            item
        } else if let Ok((active, producer)) = self
            .world
            .query_one_mut::<(&mut Active, &ItemProducer)>(target)
            && active.is_active
        {
            // producers stay put and are deactivated until they regrow
            active.is_active = false;
            producer.item
        } else {
            return;
        };

        if let Ok(inventory) = self.world.query_one_mut::<&mut Inventory>(id) {
            if inventory.hand.is_some() {
                inventory.back = inventory.hand.take();
            }
            inventory.hand = Some(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        env::Environment,
        envs::survival::{Survival, config::SurvivalConfig},
        timestep::TimeStepBuffers,
    };

    fn chop(prototype: Prototype) -> Option<ItemType> {
        let config = SurvivalConfig::default();
        let mut env = Survival::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(0, &mut buffers.view_mut());

        let state = &mut env.state;
        let agent = state.agents[0];
        let (&position, direction) = state
            .world
            .query_one_mut::<(&Position, &Direction)>(agent)
            .unwrap();
        let target_pos = position + direction.to_pos();
        for target in state.spatial_index[target_pos.idx()].into_iter().flatten() {
            state.despawn(target);
        }

        let target = state.spawn_prototype(prototype, target_pos);
        state
            .world
            .query_one_mut::<&mut Inventory>(agent)
            .unwrap()
            .hand = Some(ItemType::StoneAxe);
        state.item_use(agent);

        assert!(!state.world.contains(target), "the chopped entity is gone");
        assert_eq!(
            state.world.query_one_mut::<&Inventory>(agent).unwrap().hand,
            Some(ItemType::StoneAxe),
            "the axe is kept"
        );
        state.spatial_index[target_pos.idx()][0]
            .map(|item| *state.world.query_one_mut::<&ItemType>(item).unwrap())
    }

    #[test]
    fn chopping_a_tree_leaves_wood() {
        assert_eq!(chop(Prototype::Tree), Some(ItemType::Wood));
    }

    #[test]
    fn chopping_a_berry_bush_leaves_a_stick() {
        assert_eq!(chop(Prototype::BerryBush), Some(ItemType::Stick));
    }

    #[test]
    fn taking_tall_grass_gives_cut_grass_and_leaves_it_cut() {
        let config = SurvivalConfig::default();
        let mut env = Survival::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(0, &mut buffers.view_mut());

        let state = &mut env.state;
        let agent = state.agents[0];
        let (&position, direction) = state
            .world
            .query_one_mut::<(&Position, &Direction)>(agent)
            .unwrap();
        let target_pos = position + direction.to_pos();
        for target in state.spatial_index[target_pos.idx()].into_iter().flatten() {
            state.despawn(target);
        }
        *state.world.query_one_mut::<&mut Inventory>(agent).unwrap() = Inventory::default();

        let grass = state.spawn_prototype(Prototype::TallGrass, target_pos);
        state.item_take(agent);
        assert!(state.world.contains(grass), "the grass stays");
        assert_eq!(
            state.world.query_one_mut::<&Inventory>(agent).unwrap().hand,
            Some(ItemType::CutGrass)
        );

        env.state.tick_active_obs();
        assert_eq!(
            *env.state
                .world
                .query_one_mut::<&SurvivalObs>(grass)
                .unwrap(),
            SurvivalObs::TileCutGrass
        );
    }

    #[test]
    fn taking_from_a_berry_bush_deactivates_it_until_it_regrows() {
        let config = SurvivalConfig::default();
        let mut env = Survival::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(0, &mut buffers.view_mut());

        let state = &mut env.state;
        let agent = state.agents[0];
        let (&position, direction) = state
            .world
            .query_one_mut::<(&Position, &Direction)>(agent)
            .unwrap();
        let target_pos = position + direction.to_pos();
        for target in state.spatial_index[target_pos.idx()].into_iter().flatten() {
            state.despawn(target);
        }
        *state.world.query_one_mut::<&mut Inventory>(agent).unwrap() = Inventory::default();

        let bush = state.spawn_prototype(Prototype::BerryBush, target_pos);
        let is_active = |env: &mut Survival| {
            env.state
                .world
                .query_one_mut::<&Active>(bush)
                .unwrap()
                .is_active
        };

        env.state.item_take(agent);
        assert!(env.state.world.contains(bush), "the bush stays");
        assert!(!is_active(&mut env), "the bush is picked");
        assert_eq!(
            env.state
                .world
                .query_one_mut::<&Inventory>(agent)
                .unwrap()
                .hand,
            Some(ItemType::Berry)
        );

        env.state.item_take(agent);
        let inventory = env.state.world.query_one_mut::<&Inventory>(agent).unwrap();
        assert_eq!(inventory.back, None, "a picked bush gives nothing");

        for _ in 0..10 {
            env.tick_reactivate_after();
        }
        assert!(is_active(&mut env), "the bush regrows");
    }
}
