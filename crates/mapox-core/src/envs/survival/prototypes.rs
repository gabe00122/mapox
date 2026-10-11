use hecs::Entity;

use crate::envs::{
    common::{Direction, Position},
    survival::{
        SurvivalObs, SurvivalState,
        fire::{Active, ActiveObs, Fueled},
        items::{Inventory, ItemType},
        lightning::LightEmitter,
        needs::{Health, Hunger},
        refine::RefineType,
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
            Prototype::Item(item_type) => {
                self.spawn((position, Slot::Lower, item_type.obs(), item_type))
            }
        }
    }
}
