//! What agents carry and make: items, the recipes that combine them, and the
//! jobs tools do.

use crate::envs::common::Position;

use super::{SurvivalConfig, metrics::Achievement, tiles::SurvivalObs};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Item {
    Stick,
    Stone,
    Wood,
    Berry,
    CookedBerry,
    Carrot,
    Grass,
    Axe,
    Campfire,
}

impl Item {
    pub(super) fn tile(self) -> SurvivalObs {
        match self {
            Item::Stick => SurvivalObs::ItemStick,
            Item::Stone => SurvivalObs::ItemStone,
            Item::Wood => SurvivalObs::ItemWood,
            Item::Berry => SurvivalObs::ItemBerry,
            Item::CookedBerry => SurvivalObs::ItemCookedBerry,
            Item::Carrot => SurvivalObs::ItemCarrot,
            Item::Grass => SurvivalObs::ItemGrass,
            Item::Axe => SurvivalObs::ItemAxe,
            Item::Campfire => SurvivalObs::ItemCampfire,
        }
    }

    /// The achievement for picking this up, for the raw materials.
    pub(super) fn collected(self) -> Option<Achievement> {
        match self {
            Item::Stick => Some(Achievement::CollectStick),
            Item::Stone => Some(Achievement::CollectStone),
            Item::Wood => Some(Achievement::CollectWood),
            Item::Berry => Some(Achievement::CollectBerry),
            Item::Carrot => Some(Achievement::CollectCarrot),
            Item::Grass => Some(Achievement::CollectGrass),
            _ => None,
        }
    }

    /// Hunger the item restores eaten, if it is food.
    pub(super) fn food(self, config: &SurvivalConfig) -> Option<(u16, Achievement)> {
        match self {
            Item::Berry => Some((config.berry_food, Achievement::EatBerry)),
            Item::CookedBerry => Some((config.cooked_berry_food, Achievement::EatCookedBerry)),
            Item::Carrot => Some((config.carrot_food, Achievement::EatCarrot)),
            _ => None,
        }
    }
}

/// Combining the item in hand with the backpack's makes the third, either
/// way round, into the hand.
pub(super) const RECIPES: &[(Item, Item, Item, Achievement)] = &[
    (Item::Stick, Item::Stone, Item::Axe, Achievement::MakeAxe),
    (
        Item::Wood,
        Item::Stone,
        Item::Campfire,
        Achievement::MakeCampfire,
    ),
];

pub(super) fn recipe(a: Item, b: Item) -> Option<(Item, Achievement)> {
    RECIPES
        .iter()
        .find(|&&(x, y, _, _)| (x, y) == (a, b) || (y, x) == (a, b))
        .map(|&(_, _, made, achievement)| (made, achievement))
}

/// Slow work. Using a job's tool on its tile in front starts it (with empty
/// hands for a job done by hand), and locks the agent in place, able only
/// to wait, until its steps are done; then the tile becomes what the job
/// leaves of it. Felling a tree with an axe, and by hand harvesting tall grass
/// and digging up carrots, are the first; digging walls and mining are meant
/// to join them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Job {
    Chop,
    Harvest,
    DigCarrot,
}

impl Job {
    const ALL: [Job; 3] = [Job::Chop, Job::Harvest, Job::DigCarrot];

    /// The job `tool` does on `tile`, if any; `None` is bare hands.
    pub(super) fn of(tool: Option<Item>, tile: SurvivalObs) -> Option<Job> {
        Self::ALL
            .into_iter()
            .find(|job| job.tool() == tool && job.works() == tile)
    }

    fn tool(self) -> Option<Item> {
        match self {
            Job::Chop => Some(Item::Axe),
            Job::Harvest | Job::DigCarrot => None,
        }
    }

    /// The tile the job works on.
    pub(super) fn works(self) -> SurvivalObs {
        match self {
            Job::Chop => SurvivalObs::TileTree,
            Job::Harvest => SurvivalObs::TileTallGrass,
            Job::DigCarrot => SurvivalObs::TileBuriedCarrot,
        }
    }

    /// What the job leaves of its tile when it's done.
    pub(super) fn leaves(self) -> SurvivalObs {
        match self {
            Job::Chop => SurvivalObs::ItemWood,
            Job::Harvest => SurvivalObs::ItemGrass,
            Job::DigCarrot => SurvivalObs::ItemCarrot,
        }
    }

    /// Steps the job takes, the action that starts it included.
    pub(super) fn steps(self, config: &SurvivalConfig) -> u32 {
        match self {
            Job::Chop => config.chop_steps,
            Job::Harvest => config.harvest_steps,
            Job::DigCarrot => config.dig_carrot_steps,
        }
    }

    pub(super) fn achievement(self) -> Achievement {
        match self {
            Job::Chop => Achievement::ChopTree,
            Job::Harvest => Achievement::HarvestGrass,
            Job::DigCarrot => Achievement::DigCarrot,
        }
    }
}

/// A job under way.
#[derive(Debug, Clone, Copy)]
pub(super) struct Work {
    pub(super) job: Job,
    /// The tile being worked, in front of the agent.
    pub(super) target: Position,
    pub(super) steps_left: u32,
}
