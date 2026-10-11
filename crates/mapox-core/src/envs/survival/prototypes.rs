use hecs::Entity;

use crate::envs::{
    common::{Direction, Position},
    survival::{
        SurvivalObs, SurvivalState,
        fire::{Active, ActiveObs, Fueled},
        items::{Inventory, ItemType},
        lightning::LightEmitter,
        needs::{Health, Hunger},
        plants::{Chopable, ItemProducer, ReactiveAfter},
        refine::RefineType,
        survivor::DirectionObs,
        world::{Agent, Slot},
    },
};

pub(super) enum Prototype {
    Survivor { agent_index: usize },
    Fire,
    BerryBush,
    Tree,
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
                    fuel: 50,
                    max_fuel: 50,
                    burn_rate: 1,
                },
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
                ReactiveAfter::new(10),
                Chopable {
                    item: Some(ItemType::Stick),
                },
            )),
            Prototype::Tree => self.spawn((
                position,
                Slot::Both,
                SurvivalObs::TileTree,
                Chopable {
                    item: Some(ItemType::Wood),
                },
            )),
            Prototype::Item(item_type) => {
                self.spawn((position, Slot::Lower, item_type.obs(), item_type))
            }
        }
    }
}
