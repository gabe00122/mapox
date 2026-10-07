use serde::{Deserialize, Serialize};

/// Every stat runs from zero to this.
pub const MAX_STAT: u16 = 150;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SurvivalConfig {
    pub num_agents: usize,

    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,

    /// Shares of the map under water and rock: the lowest and the highest
    /// ground of an elevation field.
    pub water_fraction: f64,
    pub rock_fraction: f64,
    /// Shares of the land between that are forest and scrub: its wettest and
    /// its driest ground by a moisture field. The rest is meadow. See
    /// [`Biome::growth`](super::terrain::Biome::growth) for what each grows.
    pub forest_fraction: f64,
    pub scrub_fraction: f64,

    pub start_hunger: u16,
    pub start_health: u16,
    /// Hunger drops by one every this many steps.
    pub hunger_interval: u32,
    /// Health lost on every step spent at zero hunger.
    pub starve_damage: u16,
    /// Health grows back by one every `regen_interval` steps while hunger is
    /// at least `regen_threshold`.
    pub regen_threshold: u16,
    pub regen_interval: u32,

    /// Hunger a berry restores, raw and cooked, and a carrot.
    pub berry_food: u16,
    pub cooked_berry_food: u16,
    pub carrot_food: u16,
    /// Steps a picked bush takes to fruit again.
    pub bush_regrow_steps: u32,
    /// Steps felling a tree takes, the use that starts it included. See
    /// [`Job`](super::items::Job).
    pub chop_steps: u32,
    /// Steps harvesting tall grass by hand takes, the use that starts it
    /// included.
    pub harvest_steps: u32,
    /// Steps digging up a buried carrot by hand takes, the use that starts
    /// it included.
    pub dig_carrot_steps: u32,
    /// Steps a fire burns before it goes out.
    pub fire_burn_steps: u32,
    /// The last of those steps it burns low, when wood stokes it back up to
    /// a full `fire_burn_steps`.
    pub fire_low_steps: u32,
    /// How far a fire's light reaches, in cells.
    pub fire_light_radius: i32,

    /// Steps of day and then of night in each cycle; the episode starts at
    /// dawn.
    pub day_length: u32,
    pub night_length: u32,
    /// How far an agent sees around itself at night. Lit ground it sees as
    /// far as by day.
    pub night_vision_radius: i32,
    /// Steps at the end of each day over which sight closes in from the
    /// whole view to `night_vision_radius`, warning that night is coming.
    pub dusk_length: u32,

    /// Spider nests reset sets in the forest; each hatches a spider at every
    /// nightfall.
    pub num_spider_eggs: usize,
    /// Health a spider bite takes.
    pub spider_damage: u16,
    /// How far, walking, a spider tracks an agent.
    pub spider_hunt_radius: u32,
}

impl Default for SurvivalConfig {
    fn default() -> Self {
        Self {
            num_agents: 8,
            width: 80,
            height: 70,
            view_width: 15,
            view_height: 15,
            water_fraction: 0.10,
            rock_fraction: 0.15,
            forest_fraction: 0.35,
            scrub_fraction: 0.25,
            start_hunger: 100,
            start_health: MAX_STAT,
            hunger_interval: 2,
            starve_damage: 1,
            regen_threshold: 100,
            regen_interval: 4,
            berry_food: 20,
            cooked_berry_food: 50,
            carrot_food: 35,
            bush_regrow_steps: 200,
            chop_steps: 6,
            harvest_steps: 3,
            dig_carrot_steps: 1,
            fire_burn_steps: 200,
            fire_low_steps: 50,
            fire_light_radius: 4,
            day_length: 200,
            night_length: 100,
            night_vision_radius: 2,
            dusk_length: 50,
            num_spider_eggs: 4,
            spider_damage: 15,
            spider_hunt_radius: 12,
        }
    }
}
