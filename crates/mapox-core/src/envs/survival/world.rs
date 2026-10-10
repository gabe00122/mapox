use hecs::{DynamicBundle, Entity};

use crate::envs::{common::Position, survival::SurvivalState};

pub(super) struct Agent {
    pub agent_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Slot {
    Lower,
    Upper,
    Both,
}

impl Slot {
    pub fn indices(self) -> std::ops::Range<usize> {
        match self {
            Slot::Lower => 0..1,
            Slot::Upper => 1..2,
            Slot::Both => 0..2,
        }
    }
}

pub(super) type EntityCell = [Option<Entity>; 2];

impl SurvivalState {
    fn set_cell(&mut self, position: Position, slot: Slot, entity: Option<Entity>) {
        let cell = &mut self.spatial_index[position.idx()];
        for i in slot.indices() {
            cell[i] = entity;
        }
    }

    pub(super) fn is_occupied(&self, position: Position, slot: Slot) -> bool {
        let cell = &self.spatial_index[position.idx()];
        slot.indices().any(|i| cell[i].is_some())
    }

    /// True if `slot` at `position` is taken or the terrain blocks movement.
    pub(super) fn is_blocked(&self, position: Position, slot: Slot) -> bool {
        self.is_occupied(position, slot) || self.tiles[position.idx()].move_blocked()
    }

    pub(super) fn spawn(&mut self, components: impl DynamicBundle) -> Entity {
        let id = self.world.spawn(components);

        if let Ok((&position, &slot)) = self.world.query_one_mut::<(&Position, &Slot)>(id) {
            self.set_cell(position, slot, Some(id));
        }

        id
    }

    pub(super) fn despawn(&mut self, id: Entity) {
        if let Ok((&position, &slot)) = self.world.query_one_mut::<(&Position, &Slot)>(id) {
            self.set_cell(position, slot, None);
        }

        let _ = self.world.despawn(id);
    }

    /// Move an entity, keeping the objects layer in sync.
    pub(super) fn move_entity(&mut self, id: Entity, target: Position) {
        let (position, &slot) = self
            .world
            .query_one_mut::<(&mut Position, &Slot)>(id)
            .expect("entity must have Position and Slot");

        let old = std::mem::replace(position, target);
        self.set_cell(old, slot, None);
        self.set_cell(target, slot, Some(id));
    }
}
