use rand::seq::SliceRandom;

use crate::{
    envs::{
        common::{Direction, Position, vocab_enum::VocabEnum},
        survival::{Slot, SurvivalAction, SurvivalObs, SurvivalState},
    },
    timestep::TimeStepMut,
    vocab::VocabId,
};

pub(super) struct DirectionObs {
    pub tiles: [SurvivalObs; 4],
}

impl SurvivalState {
    pub(super) fn tick_direction_tiles(&mut self) {
        for (obs, dir, dir_obs) in self
            .world
            .query_mut::<(&mut SurvivalObs, &Direction, &DirectionObs)>()
        {
            *obs = dir_obs.tiles[*dir as usize];
        }
    }

    pub(super) fn tick_survivors(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.agent_order.shuffle(&mut self.rngs);

        for i in 0..self.agent_order.len() {
            let agent_index = self.agent_order[i];
            let agent_id = self.agents[agent_index];

            timestep.last_action[agent_index] = actions[agent_index];

            let action = SurvivalAction::from_id(actions[agent_index]);
            match action {
                SurvivalAction::MoveUp
                | SurvivalAction::MoveRight
                | SurvivalAction::MoveDown
                | SurvivalAction::MoveLeft => {
                    if let Some(move_direction) = action.move_direction()
                        && let Ok((&position, direction, &slot)) =
                            self.world
                                .query_one_mut::<(&Position, &mut Direction, &Slot)>(agent_id)
                    {
                        *direction = move_direction;
                        let target = position + move_direction.to_pos();
                        if !self.is_blocked(target, slot) {
                            self.move_entity(agent_id, target);
                        }
                    }
                }
                SurvivalAction::ItemTake => self.item_take(agent_id),
                SurvivalAction::ItemSwap => self.item_swap(agent_id),
                SurvivalAction::ItemPut => self.item_put(agent_id),
                SurvivalAction::ItemCombine => self.item_craft(agent_id),
                SurvivalAction::ItemUse => self.item_use(agent_id),
                SurvivalAction::Noop => {}
            }
        }
    }
}
