use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SurvivalConfig {
    pub num_agents: usize,

    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,
    /// How far an agent sees around itself in the dark. Past it, only lit
    /// ground shows.
    pub night_vision_radius: i32,

    /// Share of the map that is water (the lowest ground) and rock (the
    /// highest).
    pub water_fraction: f64,
    pub rock_fraction: f64,
    /// Share of the rock that is iron ore, in seams through it.
    pub iron_ore_fraction: f64,
    /// Share of the land between that is forest (the wettest) and scrub (the
    /// driest); the rest is meadow.
    pub forest_fraction: f64,
    pub scrub_fraction: f64,

    /// Scales the reward for staying fed and healthy and the penalty for
    /// dying; 0 turns it off.
    pub survival_reward: f32,
    /// Paid the first time each item enters a survivor's inventory in a
    /// life; 0 turns it off. The metrics count achievements either way.
    pub achievement_reward: f32,
}

impl Default for SurvivalConfig {
    fn default() -> Self {
        Self {
            num_agents: 8,
            width: 40,
            height: 40,
            view_width: 15,
            view_height: 15,
            night_vision_radius: 1,
            water_fraction: 0.10,
            rock_fraction: 0.15,
            iron_ore_fraction: 0.1,
            forest_fraction: 0.25,
            scrub_fraction: 0.25,
            survival_reward: 1.0,
            achievement_reward: 1.0,
        }
    }
}
