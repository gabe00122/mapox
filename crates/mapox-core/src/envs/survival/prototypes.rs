use hecs::Entity;

use crate::envs::{
    common::{Direction, Position},
    survival::{
        SurvivalObs, SurvivalState,
        achievements::Achievements,
        fire::{Active, ActiveObs, DamageToAbove, Fueled},
        items::{Inventory, ItemType},
        lighting::LightEmitter,
        needs::{Health, Hunger},
        plants::{Choppable, ItemProducer, ReactiveAfter},
        refine::RefineType,
        survivor::DirectionObs,
        world::{Agent, Slot},
    },
};

pub(super) enum Prototype {
    Survivor { agent_index: usize },
    Fire,
    BerryBush,
    ThornBush,
    TallGrass,
    Tree,
    AppleTree,
    Portal,
    Item(ItemType),
}

impl SurvivalState {
    pub(super) fn spawn_prototype(&mut self, prototype: Prototype, position: Position) -> Entity {
        match prototype {
            Prototype::Survivor { agent_index } => self.spawn((
                Agent::new(agent_index),
                position,
                Direction::Up,
                Slot::Upper,
                SurvivalObs::AgentUp,
                DirectionObs {
                    tiles: [
                        SurvivalObs::AgentUp,
                        SurvivalObs::AgentRight,
                        SurvivalObs::AgentDown,
                        SurvivalObs::AgentLeft,
                    ],
                },
                Health::default(),
                Hunger::default(),
                Inventory::default(),
                Achievements::default(),
            )),
            Prototype::Fire => self.spawn((
                position,
                Slot::Lower,
                SurvivalObs::TileFire,
                LightEmitter { radius: 7 },
                RefineType::Cook,
                Active::default(),
                ActiveObs {
                    enabled: SurvivalObs::TileFire,
                    disabled: SurvivalObs::TileFireOut,
                },
                Fueled {
                    fuel: 200,
                    max_fuel: 500,
                    burn_rate: 2,
                },
                DamageToAbove { amount: 25 },
            )),
            Prototype::BerryBush => self.spawn((
                position,
                Slot::Lower,
                SurvivalObs::TileBerryBush,
                Active::default(),
                ActiveObs {
                    enabled: SurvivalObs::TileBerryBush,
                    disabled: SurvivalObs::TileBush,
                },
                ItemProducer {
                    item: ItemType::Berry,
                },
                ReactiveAfter::new(200),
                Choppable {
                    item: Some(ItemType::Stick),
                },
            )),
            // walkable, but it pricks whoever stands in it
            Prototype::ThornBush => self.spawn((
                position,
                Slot::Lower,
                SurvivalObs::TileThornBush,
                DamageToAbove { amount: 10 },
                Choppable {
                    item: Some(ItemType::Stick),
                },
            )),
            Prototype::TallGrass => self.spawn((
                position,
                Slot::Lower,
                SurvivalObs::TileTallGrass,
                Active::default(),
                ActiveObs {
                    enabled: SurvivalObs::TileTallGrass,
                    disabled: SurvivalObs::TileCutGrass,
                },
                ItemProducer {
                    item: ItemType::CutGrass,
                },
                ReactiveAfter::new(400),
            )),
            Prototype::Tree => self.spawn((
                position,
                Slot::Both,
                SurvivalObs::TileTree,
                Choppable {
                    item: Some(ItemType::Wood),
                },
            )),
            Prototype::AppleTree => self.spawn((
                position,
                Slot::Both,
                SurvivalObs::TileAppleTree,
                Active::default(),
                ActiveObs {
                    enabled: SurvivalObs::TileAppleTree,
                    disabled: SurvivalObs::TileTree,
                },
                ItemProducer {
                    item: ItemType::Apple,
                },
                ReactiveAfter::new(300),
                Choppable {
                    item: Some(ItemType::Wood),
                },
            )),
            // takes the lower slot so nothing can be put down over it
            Prototype::Portal => self.spawn((position, Slot::Both, SurvivalObs::TilePortal)),
            Prototype::Item(item_type) => {
                self.spawn((position, Slot::Lower, item_type.obs(), item_type))
            }
        }
    }
}
