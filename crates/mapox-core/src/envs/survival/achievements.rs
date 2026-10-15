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
            Survival, SurvivalAction, config::SurvivalConfig, items::Inventory, needs::Health,
            prototypes::Prototype,
        },
        timestep::TimeStepBuffers,
        vocab::VocabId,
    };

    /// The achievement reward `id` has earned since the last call.
    fn take_reward(env: &mut Survival, id: Entity) -> f32 {
        let agent = env.state.world.query_one_mut::<&mut Agent>(id).unwrap();
        std::mem::take(&mut agent.reward)
    }

    fn hold(env: &mut Survival, id: Entity, hand: ItemType, back: Option<ItemType>) {
        *env.state.world.query_one_mut::<&mut Inventory>(id).unwrap() = Inventory {
            hand: Some(hand),
            back,
        };
    }

    /// Clears the tile the first agent faces and returns its position.
    fn clear_facing(env: &mut Survival, id: Entity) -> Position {
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
        target_pos
    }

    #[test]
    fn each_item_rewards_once_per_life() {
        let config = SurvivalConfig::default();
        let mut env = Survival::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(0, &mut buffers.view_mut());
        let id = env.state.agents[0];
        take_reward(&mut env, id);
        let target_pos = clear_facing(&mut env, id);

        // picking items up
        *env.state.world.query_one_mut::<&mut Inventory>(id).unwrap() = Inventory::default();
        for expected in [1.0, 0.0] {
            env.state
                .spawn_prototype(Prototype::Item(ItemType::Rock), target_pos);
            env.item_take(id);
            assert_eq!(take_reward(&mut env, id), expected, "taking a rock");
        }

        // crafting, with each product its own achievement
        for expected in [1.0, 0.0] {
            hold(&mut env, id, ItemType::Rock, Some(ItemType::Stick));
            env.item_craft(id);
            assert_eq!(take_reward(&mut env, id), expected, "crafting an axe");
        }
        hold(&mut env, id, ItemType::StoneAxe, Some(ItemType::Rock));
        env.item_craft(id);
        assert_eq!(take_reward(&mut env, id), 1.0, "crafting a pickaxe");

        // refining at a fire
        env.state.spawn_prototype(Prototype::Fire, target_pos);
        for expected in [1.0, 0.0] {
            hold(&mut env, id, ItemType::Berry, None);
            env.item_refine(id);
            assert_eq!(take_reward(&mut env, id), expected, "cooking a berry");
        }

        // a new life starts the achievements fresh
        env.state.despawn(id);
        env.tick_respawn();
        let id = env.state.agents[0];
        hold(&mut env, id, ItemType::Rock, Some(ItemType::Stick));
        env.item_craft(id);
        assert_eq!(take_reward(&mut env, id), 1.0, "crafting an axe again");
    }

    #[test]
    fn metrics_count_the_lives_that_held_each_item() {
        let config = SurvivalConfig {
            num_agents: 1,
            ..SurvivalConfig::default()
        };
        let mut env = Survival::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(0, &mut buffers.view_mut());

        let id = env.state.agents[0];
        hold(&mut env, id, ItemType::Rock, Some(ItemType::Stick));
        env.item_craft(id);
        // ends this life on the next step
        env.state
            .world
            .query_one_mut::<&mut Health>(id)
            .unwrap()
            .amount = 0;

        let noop = [SurvivalAction::Noop as VocabId];
        env.step(&noop, &mut buffers.view_mut());
        // the respawned life ends with the episode, having held nothing
        env.length = env.state.time + 1;
        env.step(&noop, &mut buffers.view_mut());

        let metrics = env.consume_metrics();
        assert_eq!(
            metrics["achievements"],
            serde_json::json!({
                "rock": 0.0,
                "stick": 0.0,
                "wood": 0.0,
                "cut_grass": 0.0,
                "stone_axe": 0.5,
                "stone_pickaxe": 0.0,
                "campfire_kit": 0.0,
                "berry": 0.0,
                "cooked_berry": 0.0,
            })
        );
        assert_eq!(
            env.consume_metrics()["achievements"]["stone_axe"],
            serde_json::json!(0.0)
        );
    }

    #[test]
    fn reward_scales_leave_the_metrics_alone() {
        for (scale, expected) in [(2.0, 2.0), (0.0, 0.0)] {
            let config = SurvivalConfig {
                num_agents: 1,
                survival_reward: 0.0,
                achievement_reward: scale,
                ..SurvivalConfig::default()
            };
            let mut env = Survival::new(&config, 1);
            let mut buffers = TimeStepBuffers::new(&env);
            env.reset(0, &mut buffers.view_mut());

            let id = env.state.agents[0];
            hold(&mut env, id, ItemType::Rock, Some(ItemType::Stick));
            env.step(
                &[SurvivalAction::ItemCombine as VocabId],
                &mut buffers.view_mut(),
            );
            assert_eq!(buffers.reward[0], expected, "achievement scale {scale}");

            let metrics = env.consume_metrics();
            assert_eq!(metrics["achievements"]["stone_axe"], serde_json::json!(1.0));
        }
    }
}
