use rand::seq::SliceRandom;

use crate::{
    envs::{
        common::{Direction, Position, vocab_enum::VocabEnum},
        survival::{
            Slot, Survival, SurvivalAction, SurvivalObs,
            needs::{Health, Hunger},
            world::Agent,
        },
    },
    timestep::TimeStepMut,
    vocab::VocabId,
};

pub(super) struct DirectionObs {
    pub tiles: [SurvivalObs; 4],
}

impl Survival {
    pub(super) fn tick_direction_tiles(&mut self) {
        for (obs, dir, dir_obs) in self
            .state
            .world
            .query_mut::<(&mut SurvivalObs, &Direction, &DirectionObs)>()
        {
            *obs = dir_obs.tiles[*dir as usize];
        }
    }

    pub(super) fn tick_survivors(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.state.agent_order.shuffle(&mut self.state.rngs);

        for i in 0..self.state.agent_order.len() {
            let agent_index = self.state.agent_order[i];
            let agent_id = self.state.agents[agent_index];

            timestep.last_action[agent_index] = actions[agent_index];

            let action = SurvivalAction::from_id(actions[agent_index]);
            match action {
                SurvivalAction::MoveUp
                | SurvivalAction::MoveRight
                | SurvivalAction::MoveDown
                | SurvivalAction::MoveLeft => {
                    if let Some(move_direction) = action.move_direction()
                        && let Ok((&position, direction, &slot)) = self
                            .state
                            .world
                            .query_one_mut::<(&Position, &mut Direction, &Slot)>(agent_id)
                    {
                        *direction = move_direction;
                        let target = position + move_direction.to_pos();
                        if !self.state.is_blocked(target, slot) {
                            self.state.move_entity(agent_id, target);
                        }
                    }
                }
                SurvivalAction::ItemTake => self.item_take(agent_id),
                SurvivalAction::ItemSwap1 => self.item_swap(agent_id, 1),
                SurvivalAction::ItemSwap2 => self.item_swap(agent_id, 2),
                SurvivalAction::ItemSwap3 => self.item_swap(agent_id, 3),
                SurvivalAction::ItemSwap4 => self.item_swap(agent_id, 4),
                SurvivalAction::ItemPut => self.item_put(agent_id),
                SurvivalAction::ItemCombine => self.item_craft(agent_id),
                SurvivalAction::ItemUse => self.item_use(agent_id),
                SurvivalAction::ItemRefine => self.item_refine(agent_id),
                SurvivalAction::Noop => {}
            }
        }
    }

    /// Pays for staying fed and healthy and charges for dying, all times `scale`.
    pub(super) fn calculate_survival_reward(&mut self, scale: f32) {
        for (agent, hunger, health) in self
            .state
            .world
            .query_mut::<(&mut Agent, &Hunger, &Health)>()
        {
            let mut reward = 2.5 * (hunger.changed as f32 / hunger.max as f32)
                + 5.0 * (health.changed as f32 / health.max as f32);
            if health.amount == 0 {
                reward -= 1.0;
            }
            agent.reward += scale * reward;
        }
    }
}
