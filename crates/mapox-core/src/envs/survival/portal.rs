//! The portal: every survivor comes into the world on or next to it, at the
//! start of an episode and again after each death.

use std::collections::VecDeque;

use hecs::Entity;
use ndarray::Array2;
use rand::seq::IndexedRandom;

use crate::envs::{
    common::{Direction, Position},
    survival::{
        Survival, SurvivalState,
        fire::DamageToAbove,
        prototypes::Prototype,
        world::{Agent, Slot},
    },
};

impl Survival {
    /// Puts the portal on a random cell of open ground with nothing on it.
    /// Map generation joins all open ground into one region, so every agent
    /// can walk away from it.
    pub(super) fn place_portal(&mut self) {
        let mut free = Vec::new();
        for x in self.pad_width..self.width - self.pad_width {
            for y in self.pad_height..self.height - self.pad_height {
                let position = Position::new(x, y);
                if self.state.spatial_index[position.idx()] == [None; 2]
                    && !self.state.tiles[position.idx()].move_blocked()
                {
                    free.push(position);
                }
            }
        }

        let &portal = free
            .choose(&mut self.state.rngs)
            .expect("no open ground left for the portal");
        self.state.spawn_prototype(Prototype::Portal, portal);
        self.state.portal = portal;
    }
}

impl SurvivalState {
    /// Whether a survivor could appear at `position`: ground it could walk
    /// onto, without a fire (lit or not) underneath to burn it.
    fn can_spawn_at(&self, position: Position) -> bool {
        !self.is_blocked(position, Slot::Upper)
            && self.spatial_index[position.idx()][0]
                .is_none_or(|id| !self.world.satisfies::<&DamageToAbove>(id))
    }

    /// The spawnable cell nearest `origin` by walking distance, if any. The
    /// search walks past other agents, since they move, but not around
    /// trees, walls or water.
    fn nearest_spawn(&self, origin: Position) -> Option<Position> {
        let mut seen = Array2::from_elem(self.tiles.dim(), false);
        let mut queue = VecDeque::from([origin]);
        seen[origin.idx()] = true;

        while let Some(position) = queue.pop_front() {
            if self.can_spawn_at(position) {
                return Some(position);
            }
            for direction in [
                Direction::Up,
                Direction::Right,
                Direction::Down,
                Direction::Left,
            ] {
                let next = position + direction.to_pos();
                // the wall padding keeps the search on the map
                let passable = !self.tiles[next.idx()].move_blocked()
                    && self.spatial_index[next.idx()][1]
                        .is_none_or(|id| self.world.satisfies::<&Agent>(id));
                if passable && !seen[next.idx()] {
                    seen[next.idx()] = true;
                    queue.push_back(next);
                }
            }
        }
        None
    }

    /// Tries to bring survivor `agent_index` into the world on the open cell
    /// nearest the portal. None if there is no room anywhere.
    pub(super) fn spawn_survivor(&mut self, agent_index: usize) -> Option<Entity> {
        let position = self.nearest_spawn(self.portal)?;
        Some(self.spawn_prototype(Prototype::Survivor { agent_index }, position))
    }

    /// Brings every dead agent back at the portal, fresh; one there is no
    /// room for tries again next step.
    pub(super) fn tick_respawn(&mut self) {
        for agent_index in 0..self.agents.len() {
            if !self.world.contains(self.agents[agent_index])
                && let Some(id) = self.spawn_survivor(agent_index)
            {
                self.agents[agent_index] = id;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        env::Environment,
        envs::survival::{SurvivalAction, SurvivalObs, config::SurvivalConfig, needs::Health},
        timestep::TimeStepBuffers,
    };

    fn position(env: &Survival, id: Entity) -> Position {
        *env.state.world.get::<&Position>(id).unwrap()
    }

    #[test]
    fn agents_start_packed_around_the_one_portal() {
        for seed in 0..10 {
            let mut env = Survival::new(&SurvivalConfig::default(), 512);
            let mut buffers = TimeStepBuffers::new(&env);
            env.reset(seed, &mut buffers.view_mut());

            let portals = env.state.world.query_mut::<&SurvivalObs>().into_iter();
            let portals = portals.filter(|&obs| *obs == SurvivalObs::TilePortal);
            assert_eq!(portals.count(), 1, "seed {seed}");

            let portal = env.state.portal;
            assert_eq!(position(&env, env.state.agents[0]), portal, "seed {seed}");
            for &agent in &env.state.agents {
                let offset = position(&env, agent) - portal;
                // eight agents fit within a few steps unless the portal is
                // walled in tight
                assert!(offset.x.abs() + offset.y.abs() <= 8, "seed {seed}");
            }
        }
    }

    #[test]
    fn a_dead_agent_comes_back_at_the_portal() {
        let mut env = Survival::new(&SurvivalConfig::default(), 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(0, &mut buffers.view_mut());

        // move agent 0 off the portal so it is free to respawn onto
        let dead = env.state.agents[0];
        let portal = env.state.portal;
        let away = (0..env.width)
            .flat_map(|x| (0..env.height).map(move |y| Position::new(x, y)))
            .find(|&p| {
                let offset = p - portal;
                offset.x.abs() + offset.y.abs() > 4 && env.state.can_spawn_at(p)
            })
            .unwrap();
        env.state.move_entity(dead, away);
        env.state.world.get::<&mut Health>(dead).unwrap().amount = 0;

        let actions = vec![SurvivalAction::Noop.into(); env.num_agents()];
        env.step(&actions, &mut buffers.view_mut());
        assert!(buffers.terminated[0], "the death is reported");

        let reborn = env.state.agents[0];
        assert!(!env.state.world.contains(dead));
        assert_eq!(position(&env, reborn), env.state.portal);
        assert_eq!(env.state.world.get::<&Health>(reborn).unwrap().amount, 99);
    }
}
