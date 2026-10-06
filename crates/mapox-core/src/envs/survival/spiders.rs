//! Spider nests and the spiders out of them: hatched at nightfall, hunting
//! through the night, home by day, and never on lit ground.

use std::collections::VecDeque;

use ndarray::Array2;
use rand::seq::IndexedRandom;

use crate::envs::common::Position;

use super::{
    Survival,
    tiles::{DIRECTIONS, SurvivalObs, beside},
};

#[derive(Debug, Clone, Copy)]
pub(super) struct Spider {
    pub(super) position: Position,
    /// The nest it hatched from, and goes back to at dawn.
    pub(super) nest: Position,
}

impl Survival {
    /// The spiders' part of the world's turn, judged by the clock the agents
    /// will wake to: nests hatch at nightfall, and the spiders hunt by night
    /// and go home by day.
    pub(super) fn tick_spiders(&mut self) {
        let now = self.state.time + 1;
        if self.is_nightfall(now) {
            self.hatch_spiders();
        }
        if self.is_night(now) {
            self.hunt();
        } else {
            self.go_home();
        }
    }

    /// Open dark ground, whoever is standing on it: where spiders walk.
    fn dark_ground(&self, cell: Position) -> bool {
        self.state.base_map[cell.idx()].walkable() && !self.state.lit[cell.idx()]
    }

    /// Open dark ground with no one on it.
    fn spider_can_enter(&self, cell: Position) -> bool {
        self.state.map[cell.idx()].walkable() && !self.state.lit[cell.idx()]
    }

    /// Each nest whose spider is home, with dark open ground beside it,
    /// hatches a spider there.
    fn hatch_spiders(&mut self) {
        for i in 0..self.state.eggs.len() {
            let nest = self.state.eggs[i];
            if self.state.spiders.iter().any(|spider| spider.nest == nest) {
                continue;
            }
            let spot = DIRECTIONS
                .iter()
                .map(|&d| nest + d)
                .find(|&cell| self.spider_can_enter(cell));
            if let Some(position) = spot {
                self.state.spiders.push(Spider { position, nest });
                self.place_creature(position, SurvivalObs::Spider);
            }
        }
    }

    /// The spiders' turn by night. Each in turn walks out of any light it
    /// stands in; failing that bites an agent beside it in the dark; failing
    /// that closes on the nearest agent it can track; failing that wanders.
    fn hunt(&mut self) {
        if self.state.spiders.is_empty() {
            return;
        }
        let trail = self.scent();

        for i in 0..self.state.spiders.len() {
            let spider = self.state.spiders[i].position;
            if self.state.lit[spider.idx()] {
                self.flee_light(i);
                continue;
            }
            let around = DIRECTIONS.map(|d| spider + d);

            let prey = around.iter().find_map(|&cell| {
                if self.state.lit[cell.idx()] {
                    return None;
                }
                self.state.agents.iter().position(|a| a.position == cell)
            });
            if let Some(agent_id) = prey {
                let agent = &mut self.state.agents[agent_id];
                agent.health = agent.health.saturating_sub(self.config.spider_damage);
                self.metrics.spider_bites += 1.0;
                continue;
            }

            let open: Vec<Position> = around
                .into_iter()
                .filter(|&cell| self.spider_can_enter(cell))
                .collect();
            let next = if trail[spider.idx()] != u32::MAX {
                open.iter()
                    .copied()
                    .filter(|cell| trail[cell.idx()] < trail[spider.idx()])
                    .min_by_key(|cell| trail[cell.idx()])
            } else {
                open.choose(&mut self.state.rng).copied()
            };
            if let Some(next) = next {
                self.move_spider(i, next);
            }
        }
    }

    /// The spiders' turn by day. Each walks out of any light it stands in, or
    /// else heads home by the shortest dark way and burrows back into its
    /// nest once beside it. They bite no one on the way.
    fn go_home(&mut self) {
        let mut i = 0;
        while i < self.state.spiders.len() {
            let Spider { position, nest } = self.state.spiders[i];
            if self.state.lit[position.idx()] {
                self.flee_light(i);
            } else if beside(position, nest) {
                self.remove_creature(position);
                self.state.spiders.swap_remove(i);
                continue;
            } else {
                let dark = |cell| self.dark_ground(cell);
                if let Some(next) = self.first_step(position, dark, |cell| beside(cell, nest)) {
                    self.move_spider(i, next);
                }
            }
            i += 1;
        }
    }

    /// Takes a spider standing in light a step along the shortest way out
    /// of it.
    fn flee_light(&mut self, i: usize) {
        let from = self.state.spiders[i].position;
        let open = |cell: Position| self.state.base_map[cell.idx()].walkable();
        if let Some(next) = self.first_step(from, open, |cell| !self.state.lit[cell.idx()]) {
            self.move_spider(i, next);
        }
    }

    /// Moves a spider onto `next`, if no one is standing there.
    fn move_spider(&mut self, i: usize, next: Position) {
        if !self.state.map[next.idx()].walkable() {
            return;
        }
        self.remove_creature(self.state.spiders[i].position);
        self.place_creature(next, SurvivalObs::Spider);
        self.state.spiders[i].position = next;
    }

    /// The first step of a shortest walk from `from` through `passable` cells
    /// to the nearest cell `goal` accepts; `None` if no such cell is
    /// reachable. Who is standing where is left to the caller.
    fn first_step(
        &self,
        from: Position,
        passable: impl Fn(Position) -> bool,
        goal: impl Fn(Position) -> bool,
    ) -> Option<Position> {
        let mut seen = Array2::from_elem(self.state.map.dim(), false);
        seen[from.idx()] = true;
        // each cell reached, with the first step of the way it was reached by
        let mut frontier = VecDeque::new();
        for step in DIRECTIONS {
            let next = from + step;
            if passable(next) {
                seen[next.idx()] = true;
                frontier.push_back((next, next));
            }
        }
        while let Some((cell, first)) = frontier.pop_front() {
            if goal(cell) {
                return Some(first);
            }
            for step in DIRECTIONS {
                let next = cell + step;
                if !seen[next.idx()] && passable(next) {
                    seen[next.idx()] = true;
                    frontier.push_back((next, first));
                }
            }
        }
        None
    }

    /// How far each cell is, walking dark open ground, from the nearest agent
    /// standing in the dark: what a spider follows. `u32::MAX` past
    /// `spider_hunt_radius`, and through light, which hides the trail.
    fn scent(&self) -> Array2<u32> {
        let mut trail = Array2::from_elem(self.state.map.dim(), u32::MAX);
        let mut frontier = VecDeque::new();
        for agent in &self.state.agents {
            if self.dark_ground(agent.position) {
                trail[agent.position.idx()] = 0;
                frontier.push_back(agent.position);
            }
        }
        while let Some(cell) = frontier.pop_front() {
            let distance = trail[cell.idx()];
            if distance >= self.config.spider_hunt_radius {
                continue;
            }
            for step in DIRECTIONS {
                let next = cell + step;
                if trail[next.idx()] == u32::MAX && self.dark_ground(next) {
                    trail[next.idx()] = distance + 1;
                    frontier.push_back(next);
                }
            }
        }
        trail
    }
}
