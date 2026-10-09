use hecs::{DynamicBundle, Entity};

use crate::envs::{common::Position, survival::SurvivalState};

pub(super) struct Agent {
    pub agent_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Slot {
    Lower = 0,
    Upper = 1,
}

pub(super) type EntityCell = [Option<Entity>; 2];

impl SurvivalState {
    pub(super) fn spawn(&mut self, components: impl DynamicBundle) -> Entity {
        let id = self.world.spawn(components);

        if let Ok((position, &slot)) = self.world.query_one_mut::<(&Position, &Slot)>(id) {
            self.spatial_index[position.idx()][slot as usize] = Some(id);
        }

        id
    }

    /// Move an entity, keeping the objects layer in sync.
    pub(super) fn move_entity(&mut self, id: Entity, target: Position) {
        let (position, &slot) = self
            .world
            .query_one_mut::<(&mut Position, &Slot)>(id)
            .expect("entity must have Position and Slot");

        let slot = slot as usize;
        self.spatial_index[position.idx()][slot] = None;
        *position = target;
        self.spatial_index[target.idx()][slot] = Some(id);
    }
}
