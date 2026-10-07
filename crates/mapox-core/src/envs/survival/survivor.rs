//! The agents' bodies: their stats and inventory, and life and death.

use rand::RngExt;

use crate::envs::common::Position;

use super::{
    MAX_STAT, Survival, SurvivalConfig,
    items::{Item, Work},
    metrics::{Achievement, SurvivalMetrics},
    tiles::{AGENT_TILES, DIRECTIONS, SurvivalObs},
};

const _: () = assert!(
    Achievement::ALL.len() <= u32::BITS as usize,
    "`Survivor::unlocked` has a bit per achievement"
);

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Survivor {
    pub(super) position: Position,
    /// Facing as a `DIRECTIONS` index.
    pub(super) dir: u8,
    pub(super) hunger: u16,
    pub(super) health: u16,
    pub(super) temperature: u16,
    pub(super) hands: Option<Item>,
    pub(super) backpack: Option<Item>,
    /// Achievements unlocked this life, one bit per `Achievement`.
    pub(super) unlocked: u32,
    /// The job it is busy with, if any.
    pub(super) work: Option<Work>,
}

impl Survivor {
    pub(super) fn spawn(position: Position, dir: u8, config: &SurvivalConfig) -> Self {
        Self {
            position,
            dir,
            hunger: config.start_hunger,
            health: config.start_health,
            temperature: config.start_temperature,
            ..Default::default()
        }
    }

    pub(super) fn ahead(&self) -> Position {
        self.position + DIRECTIONS[self.dir as usize]
    }

    pub(super) fn tile(&self) -> SurvivalObs {
        AGENT_TILES[self.dir as usize]
    }

    pub(super) fn eat(&mut self, food: u16) {
        self.hunger = self.hunger.saturating_add(food).min(MAX_STAT);
    }

    pub(super) fn unlock(&mut self, metrics: &mut SurvivalMetrics, achievement: Achievement) {
        let bit = 1 << achievement as u32;
        if self.unlocked & bit == 0 {
            self.unlocked |= bit;
            metrics.achievements[achievement as usize] += 1.0;
        }
    }
}

impl Survival {
    pub(super) fn spawn(&mut self, agent_id: usize, position: Position) {
        let dir = self.state.rng.random_range(0..4);
        let agent = Survivor::spawn(position, dir, &self.config);
        self.place_creature(position, agent.tile());

        if agent_id < self.state.agents.len() {
            self.state.agents[agent_id] = agent;
        } else {
            self.state.agents.push(agent);
        }
    }

    /// Hunger drains, and temperature with it in winter off firelit ground;
    /// health follows them. Returns whether the agent died, starved, frozen
    /// or bitten.
    pub(super) fn tick_stats(&mut self, agent_id: usize) -> bool {
        // counting this step, so the first drain lands after a full interval
        let steps = self.state.time as u32 + 1;
        let winter = self.is_winter(steps as usize);
        let warm = self.by_fire(self.state.agents[agent_id].position);
        let config = &self.config;
        let agent = &mut self.state.agents[agent_id];

        if steps.is_multiple_of(config.hunger_interval) {
            agent.hunger = agent.hunger.saturating_sub(1);
        }
        if warm {
            agent.temperature = (agent.temperature + config.fire_warmth).min(MAX_STAT);
        } else if winter && steps.is_multiple_of(config.chill_interval) {
            agent.temperature = agent.temperature.saturating_sub(1);
        }

        let starving = agent.hunger == 0;
        let freezing = agent.temperature == 0;
        if starving || freezing {
            let damage =
                starving as u16 * config.starve_damage + freezing as u16 * config.freeze_damage;
            agent.health = agent.health.saturating_sub(damage);
        } else if agent.health > 0
            && agent.hunger >= config.regen_threshold
            && steps.is_multiple_of(config.regen_interval)
        {
            agent.health = (agent.health + 1).min(MAX_STAT);
        }

        agent.health == 0
    }

    /// Takes a dead agent off the map, leaving what it carried behind: the
    /// hands' item where it stood, the backpack's beside it. An item with no
    /// open ground left to land on is lost.
    pub(super) fn kill(&mut self, agent_id: usize) {
        let agent = self.state.agents[agent_id];
        let position = agent.position;
        self.remove_creature(position);

        let mut spots = std::iter::once(position).chain(DIRECTIONS.map(|d| position + d));
        for item in [agent.hands, agent.backpack].into_iter().flatten() {
            if let Some(spot) = spots.find(|spot| self.state.map[spot.idx()].is_floor()) {
                self.lay(spot, item);
            }
        }

        self.metrics.deaths += 1.0;
    }
}
