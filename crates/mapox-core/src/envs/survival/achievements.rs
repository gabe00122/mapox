use hecs::Entity;
use strum::EnumCount;

use crate::envs::survival::{Survival, items::ItemType, world::Agent};

/// The items that have been in a survivor's inventory this life. It lives on
/// the survivor entity, so a respawn starts it fresh.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Achievements {
    unlocked: [bool; ItemType::COUNT],
}

impl Achievements {
    pub(super) fn has(&self, item: ItemType) -> bool {
        self.unlocked[item as usize]
    }

    /// True the first time `item` is unlocked.
    fn unlock(&mut self, item: ItemType) -> bool {
        !std::mem::replace(&mut self.unlocked[item as usize], true)
    }
}

impl Survival {
    /// Records the first time this life `item` enters the inventory of `id`,
    /// paying the achievement reward for it.
    pub(super) fn unlock_achievement(&mut self, id: Entity, item: ItemType) {
        if let Ok((agent, achievements)) = self
            .state
            .world
            .query_one_mut::<(&mut Agent, &mut Achievements)>(id)
            && achievements.unlock(item)
        {
            agent.reward += self.config.achievement_reward;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        env::Environment,
        envs::common::{Direction, Position},
        envs::survival::{
            Survival, config::SurvivalConfig, items::Inventory, prototypes::Prototype,
        },
        timestep::TimeStepBuffers,
    };

    /// The achievement reward `id` has earned since the last call.
    fn take_reward(env: &mut Survival, id: Entity) -> f32 {
        let agent = env.state.world.query_one_mut::<&mut Agent>(id).unwrap();
        std::mem::take(&mut agent.reward)
    }

    /// Empties the hands of `id` and gives it an item to take in front of it.
    fn offer_item(env: &mut Survival, id: Entity, item: ItemType) {
        let (&position, direction) = env
            .state
            .world
            .query_one_mut::<(&Position, &Direction)>(id)
            .unwrap();
        let target_pos = position + direction.to_pos();
        for target in env.state.spatial_index[target_pos.idx()]
            .into_iter()
            .flatten()
        {
            env.state.despawn(target);
        }
        *env.state.world.query_one_mut::<&mut Inventory>(id).unwrap() = Inventory::default();
        env.state.spawn_prototype(Prototype::Item(item), target_pos);
    }

    #[test]
    fn each_item_rewards_once_per_life() {
        let config = SurvivalConfig::default();
        let mut env = Survival::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(0, &mut buffers.view_mut());
        let reward = config.achievement_reward;

        let id = env.state.agents[0];
        take_reward(&mut env, id);
        for expected in [reward, 0.0] {
            offer_item(&mut env, id, ItemType::Rock);
            env.item_take(id);
            assert_eq!(take_reward(&mut env, id), expected, "taking a rock");
        }

        // a new life starts the achievements fresh
        env.state.despawn(id);
        env.tick_respawn();
        let id = env.state.agents[0];
        take_reward(&mut env, id);
        offer_item(&mut env, id, ItemType::Rock);
        env.item_take(id);
        assert_eq!(take_reward(&mut env, id), reward, "taking a rock again");
    }
}
