//! The portal: every survivor comes into the world on or next to it, at the
//! start of an episode and again after each death.
// Vibe coded and might need further evaluation

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
