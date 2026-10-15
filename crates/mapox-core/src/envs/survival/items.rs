use hecs::Entity;
use strum::{EnumCount, EnumIter, IntoStaticStr};

use crate::envs::{
    common::{Direction, Position},
    survival::{
        Slot, Survival, SurvivalObs, SurvivalState,
        fire::{Active, Fueled},
        needs::Hunger,
        plants::{Chopable, ItemProducer},
        prototypes::Prototype,
        refine::RefineType,
        world::Agent,
    },
};

// the snake_case names are the metric keys
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumCount, EnumIter, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(super) enum ItemType {
    Rock,
    Stick,
    Wood,
    CutGrass,
    StoneAxe,
    StonePickaxe,
    CampfireKit,
    Berry,
    CookedBerry,
}

// requirement, requirement, product
const RECIPES: &[(ItemType, ItemType, ItemType)] = &[
    (ItemType::Rock, ItemType::Stick, ItemType::StoneAxe),
    (ItemType::Wood, ItemType::CutGrass, ItemType::CampfireKit),
    (ItemType::StoneAxe, ItemType::Rock, ItemType::StonePickaxe),
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

    fn hunger_restore(&self) -> Option<i32> {
        match self {
            ItemType::Berry => Some(10),
            ItemType::CookedBerry => Some(40),
            _ => None,
        }
    }

    fn fuel_value(&self) -> Option<i32> {
        match self {
            ItemType::Stick | ItemType::CutGrass => Some(50),
            ItemType::Wood => Some(200),
            _ => None,
        }
    }
}

// Inventory could be extended to act like a stack with n slots
// instead of swap it would be n swap actions for swapping 0 with the nth position
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Inventory {
    pub hand: Option<ItemType>,
    pub back: Option<ItemType>,
}

// Each item action is split in two: a `plan_*` works out what it would do
// without changing anything, or `None` when it would do nothing, and the
// `item_*` applies that plan. The action mask is `plan_*(id).is_some()`, so the mask and the action
// can't disagree. The plans run again when the action does: other agents move
// first, so a plan made when the mask was encoded may no longer hold.

pub(super) enum UsePlan {
    Eat(i32),
    PlaceCampfire(Position),
    Chop {
        target: Entity,
        at: Position,
        drop: Option<ItemType>,
    },
}

pub(super) enum PutPlan {
    // putting fuel into a fueled thing (e.g. a fire) burns it up, relighting it if burnt out
    Fuel { target: Entity, fuel: i32 },
    Drop { item: ItemType, at: Position },
}

pub(super) enum TakePlan {
    Pick(Entity, ItemType),
    // producers stay put and are deactivated until they regrow
    Harvest(Entity, ItemType),
}

impl SurvivalState {
    /// The tile `id` faces and what it holds, where every item action starts.
    /// `None` for a dead agent, whose entity is stale.
    fn item_context(&mut self, id: Entity) -> Option<(Position, Inventory)> {
        let (&position, direction, &inventory) = self
            .world
            .query_one_mut::<(&Position, &Direction, &Inventory)>(id)
            .ok()?;
        Some((position + direction.to_pos(), inventory))
    }

    pub(super) fn plan_refine(&mut self, id: Entity) -> Option<ItemType> {
        let (target_pos, inventory) = self.item_context(id)?;
        let hand = inventory.hand?;

        let target = self.spatial_index[target_pos.idx()][0]?; // lower slot
        let (active, refine_type) = self
            .world
            .query_one_mut::<(Option<&Active>, &RefineType)>(target)
            .ok()?;
        if !Active::is_on(active) {
            return None;
        }
        refine_type.refine(hand)
    }

    pub(super) fn plan_use(&mut self, id: Entity) -> Option<UsePlan> {
        let (target_pos, inventory) = self.item_context(id)?;
        let item = inventory.hand?;

        if let Some(hunger_restore) = item.hunger_restore() {
            return self
                .world
                .satisfies::<&Hunger>(id)
                .then_some(UsePlan::Eat(hunger_restore));
        }

        match item {
            ItemType::CampfireKit => (!self.is_blocked(target_pos, Slot::Lower))
                .then_some(UsePlan::PlaceCampfire(target_pos)),
            ItemType::StoneAxe => {
                // the lower slot also holds Slot::Both entities like trees
                let target = self.spatial_index[target_pos.idx()][0]?;
                let chopable = self.world.query_one_mut::<&Chopable>(target).ok()?;
                Some(UsePlan::Chop {
                    target,
                    at: target_pos,
                    drop: chopable.item,
                })
            }
            _ => None,
        }
    }

    pub(super) fn plan_craft(&mut self, id: Entity) -> Option<ItemType> {
        let &inventory = self.world.query_one_mut::<&Inventory>(id).ok()?;
        let (Some(hand), Some(back)) = (inventory.hand, inventory.back) else {
            return None;
        };

        RECIPES
            .iter()
            .find(|&&(a, b, _)| (a == hand && b == back) || (a == back && b == hand))
            .map(|&(_, _, product)| product)
    }

    /// Swapping two empty slots is the only swap that does nothing.
    pub(super) fn can_swap(&mut self, id: Entity) -> bool {
        self.world
            .query_one_mut::<&Inventory>(id)
            .is_ok_and(|inventory| inventory.hand.is_some() || inventory.back.is_some())
    }

    pub(super) fn plan_put(&mut self, id: Entity) -> Option<PutPlan> {
        let (target_pos, inventory) = self.item_context(id)?;
        let item = inventory.hand?;

        if let Some(fuel) = item.fuel_value()
            && let Some(target) = self.spatial_index[target_pos.idx()][0] // lower slot
            && self.world.satisfies::<(&Active, &Fueled)>(target)
        {
            return Some(PutPlan::Fuel { target, fuel });
        }

        (!self.is_blocked(target_pos, Slot::Lower)).then_some(PutPlan::Drop {
            item,
            at: target_pos,
        })
    }

    pub(super) fn plan_take(&mut self, id: Entity) -> Option<TakePlan> {
        let (target_pos, inventory) = self.item_context(id)?;
        if inventory.hand.is_some() && inventory.back.is_some() {
            return None;
        }

        let target = self.spatial_index[target_pos.idx()][0]?; // lower slot
        if let Ok(&item) = self.world.query_one_mut::<&ItemType>(target) {
            return Some(TakePlan::Pick(target, item));
        }

        let (active, producer) = self
            .world
            .query_one_mut::<(&Active, &ItemProducer)>(target)
            .ok()?;
        active
            .is_active
            .then_some(TakePlan::Harvest(target, producer.item))
    }
}

impl Survival {
    pub(super) fn item_refine(&mut self, id: Entity) {
        let Some(output) = self.state.plan_refine(id) else {
            return;
        };

        if let Ok(inventory) = self.state.world.query_one_mut::<&mut Inventory>(id) {
            inventory.hand = Some(output);
        }
        self.unlock_achievement(id, output);
    }

    pub(crate) fn item_use(&mut self, id: Entity) {
        let Some(plan) = self.state.plan_use(id) else {
            return;
        };

        match plan {
            UsePlan::Eat(hunger_restore) => {
                if let Ok((inventory, hunger)) = self
                    .state
                    .world
                    .query_one_mut::<(&mut Inventory, &mut Hunger)>(id)
                {
                    hunger.update(hunger_restore);
                    inventory.hand = None;
                }
            }
            UsePlan::PlaceCampfire(at) => {
                if let Ok(inventory) = self.state.world.query_one_mut::<&mut Inventory>(id) {
                    inventory.hand = None;
                }
                self.state.spawn_prototype(Prototype::Fire, at);
            }
            UsePlan::Chop { target, at, drop } => {
                self.state.despawn(target);
                if let Some(drop) = drop {
                    self.state.spawn_prototype(Prototype::Item(drop), at);
                }
            }
        }
    }

    pub(crate) fn item_craft(&mut self, id: Entity) {
        let Some(product) = self.state.plan_craft(id) else {
            return;
        };

        if let Ok(inventory) = self.state.world.query_one_mut::<&mut Inventory>(id) {
            inventory.hand = Some(product);
            inventory.back = None;
        }
        self.unlock_achievement(id, product);
    }

    pub(super) fn item_swap(&mut self, id: Entity) {
        if let Ok(inventory) = self.state.world.query_one_mut::<&mut Inventory>(id) {
            std::mem::swap(&mut inventory.hand, &mut inventory.back);
        }
    }

    pub(super) fn item_put(&mut self, id: Entity) {
        let Some(plan) = self.state.plan_put(id) else {
            return;
        };

        match plan {
            PutPlan::Fuel { target, fuel } => {
                if let Ok((active, fueled)) = self
                    .state
                    .world
                    .query_one_mut::<(&mut Active, &mut Fueled)>(target)
                {
                    fueled.fuel = (fueled.fuel + fuel).min(fueled.max_fuel);
                    active.is_active = true;
                }
            }
            PutPlan::Drop { item, at } => {
                self.state.spawn_prototype(Prototype::Item(item), at);
            }
        }

        if let Ok(inventory) = self.state.world.query_one_mut::<&mut Inventory>(id) {
            inventory.hand = inventory.back.take();
        }
    }

    pub(super) fn item_take(&mut self, id: Entity) {
        let Some(plan) = self.state.plan_take(id) else {
            return;
        };

        let item = match plan {
            TakePlan::Pick(target, item) => {
                self.state.despawn(target);
                item
            }
            TakePlan::Harvest(target, item) => {
                if let Ok(active) = self.state.world.query_one_mut::<&mut Active>(target) {
                    active.is_active = false;
                }
                item
            }
        };

        if let Ok(inventory) = self.state.world.query_one_mut::<&mut Inventory>(id) {
            if inventory.hand.is_some() {
                inventory.back = inventory.hand.take();
            }
            inventory.hand = Some(item);
        }
        self.unlock_achievement(id, item);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        env::Environment,
        envs::survival::{Survival, SurvivalAction, config::SurvivalConfig, plants::ReactiveAfter},
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
        env.item_use(agent);

        let state = &mut env.state;

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
        env.item_take(agent);

        let state = &mut env.state;
        assert!(state.world.contains(grass), "the grass stays");
        assert_eq!(
            state.world.query_one_mut::<&Inventory>(agent).unwrap().hand,
            Some(ItemType::CutGrass)
        );

        env.tick_active_obs();
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

        env.item_take(agent);
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

        env.item_take(agent);
        let inventory = env.state.world.query_one_mut::<&Inventory>(agent).unwrap();
        assert_eq!(inventory.back, None, "a picked bush gives nothing");

        for _ in 0..10 {
            env.tick_reactivate_after();
        }
        assert!(is_active(&mut env), "the bush regrows");
    }

    /// A fresh env whose first agent is empty-handed and faces a bare tile.
    fn facing_bare_tile() -> (Survival, TimeStepBuffers, Entity, Position) {
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
        state.tiles[target_pos.idx()] = SurvivalObs::TileEmpty;
        *state.world.query_one_mut::<&mut Inventory>(agent).unwrap() = Inventory::default();

        (env, buffers, agent, target_pos)
    }

    fn hold(env: &mut Survival, agent: Entity, hand: Option<ItemType>, back: Option<ItemType>) {
        *env.state
            .world
            .query_one_mut::<&mut Inventory>(agent)
            .unwrap() = Inventory { hand, back };
    }

    /// The first agent's legal item actions.
    fn legal_items(env: &mut Survival, buffers: &mut TimeStepBuffers) -> Vec<SurvivalAction> {
        env.encode_action_mask(&mut buffers.view_mut());
        let mask = buffers.action_mask.row(0);
        for action in [
            SurvivalAction::MoveUp,
            SurvivalAction::MoveRight,
            SurvivalAction::MoveDown,
            SurvivalAction::MoveLeft,
            SurvivalAction::Noop,
        ] {
            assert!(mask[action as usize], "{action:?} is always legal");
        }
        SurvivalAction::TABLE
            .iter()
            .copied()
            .filter(|&action| action.move_direction().is_none() && action != SurvivalAction::Noop)
            .filter(|&action| mask[action as usize])
            .collect()
    }

    #[test]
    fn empty_hands_facing_nothing_mask_every_item_action() {
        let (mut env, mut buffers, _, _) = facing_bare_tile();
        assert_eq!(legal_items(&mut env, &mut buffers), []);
    }

    #[test]
    fn full_hands_mask_take_and_offer_the_recipe() {
        use SurvivalAction::*;
        let (mut env, mut buffers, agent, target_pos) = facing_bare_tile();

        hold(&mut env, agent, Some(ItemType::Rock), Some(ItemType::Stick));
        assert_eq!(
            legal_items(&mut env, &mut buffers),
            [ItemPut, ItemSwap, ItemCombine]
        );

        env.item_craft(agent);
        assert_eq!(
            legal_items(&mut env, &mut buffers),
            [ItemPut, ItemSwap],
            "an axe needs something to chop"
        );

        env.state.spawn_prototype(Prototype::Tree, target_pos);
        assert_eq!(legal_items(&mut env, &mut buffers), [ItemSwap, ItemUse]);
    }

    #[test]
    fn a_picked_bush_masks_take_until_it_regrows() {
        use SurvivalAction::*;
        let (mut env, mut buffers, agent, target_pos) = facing_bare_tile();
        let bush = env.state.spawn_prototype(Prototype::BerryBush, target_pos);
        assert_eq!(legal_items(&mut env, &mut buffers), [ItemTake]);

        env.item_take(agent);
        // a berry is eaten, not put on the occupied tile
        assert_eq!(legal_items(&mut env, &mut buffers), [ItemSwap, ItemUse]);

        let delay = env
            .state
            .world
            .get::<&ReactiveAfter>(bush)
            .unwrap()
            .reset_delay;
        for _ in 0..delay {
            env.tick_reactivate_after();
        }
        assert_eq!(
            legal_items(&mut env, &mut buffers),
            [ItemTake, ItemSwap, ItemUse]
        );
    }

    #[test]
    fn a_fire_takes_fuel_and_cooks_only_while_lit() {
        use SurvivalAction::*;
        let (mut env, mut buffers, agent, target_pos) = facing_bare_tile();
        let fire = env.state.spawn_prototype(Prototype::Fire, target_pos);

        hold(&mut env, agent, Some(ItemType::Stick), None);
        assert_eq!(legal_items(&mut env, &mut buffers), [ItemPut, ItemSwap]);

        hold(&mut env, agent, Some(ItemType::CampfireKit), None);
        assert_eq!(
            legal_items(&mut env, &mut buffers),
            [ItemSwap],
            "the tile is taken"
        );

        hold(&mut env, agent, Some(ItemType::Berry), None);
        assert_eq!(
            legal_items(&mut env, &mut buffers),
            [ItemSwap, ItemUse, ItemRefine]
        );

        env.state
            .world
            .query_one_mut::<&mut Active>(fire)
            .unwrap()
            .is_active = false;
        assert_eq!(legal_items(&mut env, &mut buffers), [ItemSwap, ItemUse]);
    }
}
