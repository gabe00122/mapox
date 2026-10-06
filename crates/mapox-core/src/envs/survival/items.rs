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

/// Slow work. Using a job's tool on its tile in front starts it, and locks the
/// agent in place, able only to wait, until its steps are done; then the
/// tile becomes what the job leaves of it. Felling a tree with an axe is the
/// first; digging and mining are meant to join it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Job {
    Chop,
}

impl Job {
    const ALL: [Job; 1] = [Job::Chop];

    /// The job `tool` does on `tile`, if any.
    pub(super) fn of(tool: Item, tile: SurvivalObs) -> Option<Job> {
        Self::ALL
            .into_iter()
            .find(|job| job.tool() == tool && job.works() == tile)
    }

    fn tool(self) -> Item {
        match self {
            Job::Chop => Item::Axe,
        }
    }

    /// The tile the job works on.
    pub(super) fn works(self) -> SurvivalObs {
        match self {
            Job::Chop => SurvivalObs::TileTree,
        }
    }

    /// What the job leaves of its tile when it's done.
    pub(super) fn leaves(self) -> SurvivalObs {
        match self {
            Job::Chop => SurvivalObs::ItemWood,
        }
    }

    /// Steps the job takes, the use that starts it included.
    pub(super) fn steps(self, config: &SurvivalConfig) -> u32 {
        match self {
            Job::Chop => config.chop_steps,
        }
    }

    pub(super) fn achievement(self) -> Achievement {
        match self {
            Job::Chop => Achievement::ChopTree,
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
