use rand::seq::SliceRandom;

use crate::{
    envs::{
        common::{Position, vocab_enum::VocabEnum},
        survival::{Slot, SurvivalAction, SurvivalState},
    },
    timestep::TimeStepMut,
    vocab::VocabId,
};

// Which direction the entity is facing
pub(super) struct Direction {
    pub front: Position,
}

impl Default for Direction {
    fn default() -> Self {
        Direction {
            front: Position::new(0, 1),
        }
    }
}

impl SurvivalState {
    pub(super) fn tick_survivors(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.agent_order.shuffle(&mut self.rngs);

        for i in 0..self.agent_order.len() {
            let agent_index = self.agent_order[i];
            let agent_id = self.agents[agent_index];
            let (&position, direction, &slot) = self
                .world
                .query_one_mut::<(&Position, &mut Direction, &Slot)>(agent_id)
                .expect("Agents should have Position, Direction and Slot");

            timestep.last_action[agent_index] = actions[agent_index];
            timestep.reward[agent_index] = 0.0;

            let action = SurvivalAction::from_id(actions[agent_index]);
            if let Some(move_direction) = action.move_direction() {
                direction.front = move_direction;
                let target = position + move_direction;
                let occupied = self.is_occupied(target, slot);
                if !occupied && !self.tiles[target.idx()].move_blocked() {
                    self.move_entity(agent_id, target);
                }
            }
        }
    }
}
