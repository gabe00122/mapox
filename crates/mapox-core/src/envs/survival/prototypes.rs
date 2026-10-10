use hecs::Entity;

use crate::envs::{
    common::{Direction, Position},
    survival::{
        SurvivalObs, SurvivalState,
        fire::Fire,
        items::{Inventory, Item, ItemType},
        lightning::LightEmitter,
        needs::{Health, Hunger},
        survivor::DirectionObs,
        world::{Agent, Slot},
    },
};

pub(super) enum Prototype {
    Survivor { agent_index: usize },
    Fire,
    Item(ItemType),
}

impl SurvivalState {
    pub(super) fn spawn_prototype(&mut self, prototype: Prototype, position: Position) -> Entity {
        match prototype {
            Prototype::Survivor { agent_index } => self.spawn((
                Agent { agent_index },
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
                Health { amount: 100 },
                Hunger { amount: 100 },
                Inventory::default(),
            )),
            Prototype::Fire => self.spawn((
                position,
                Slot::Lower,
                SurvivalObs::TileFire,
                LightEmitter { radius: 7 },
                Fire { fuel: 50 },
            )),
            Prototype::Item(item_type) => {
                self.spawn((position, Slot::Lower, item_type.obs(), Item { item_type }))
            }
        }
    }
}
