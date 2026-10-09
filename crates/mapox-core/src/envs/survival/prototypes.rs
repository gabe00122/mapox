use hecs::Entity;

use crate::envs::{
    common::Position,
    survival::{
        SurvivalObs, SurvivalState,
        fire::Fire,
        lightning::LightEmitter,
        needs::{Health, Hunger},
        world::{Agent, Slot},
    },
};

pub(super) enum Prototype {
    Survivor { agent_index: usize },
    Fire,
}

impl SurvivalState {
    pub(super) fn spawn_prototype(&mut self, prototype: Prototype, position: Position) -> Entity {
        match prototype {
            Prototype::Survivor { agent_index } => self.spawn((
                Agent { agent_index },
                position,
                Slot::Upper,
                SurvivalObs::AgentGeneric,
                Health { amount: 100 },
                Hunger { amount: 100 },
            )),
            Prototype::Fire => self.spawn((
                position,
                Slot::Lower,
                LightEmitter { radius: 7 },
                Fire { fuel: 50 },
            )),
        }
    }
}
