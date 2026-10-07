//! What agents carry and make: items, the recipes that combine them, and the
//! jobs tools do.

use crate::envs::common::Position;

use super::{Survival, SurvivalConfig, metrics::Achievement, tiles::SurvivalObs};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Item {
    Stick,
    Stone,
    Wood,
    Berry,
    CookedBerry,
    Carrot,
    CookedCarrot,
    Grass,
    Axe,
    Campfire,
    /// Lights the ground around the agent holding it, burning down a step
    /// for every step it is held, and gone once it has burnt
    /// `torch_burn_steps`. In the backpack or on the ground it neither lights
    /// nor burns, and keeps its wear.
    Torch {
        burnt: u32,
    },
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
            Item::CookedCarrot => SurvivalObs::ItemCookedCarrot,
            Item::Grass => SurvivalObs::ItemGrass,
            Item::Axe => SurvivalObs::ItemAxe,
            Item::Campfire => SurvivalObs::ItemCampfire,
            Item::Torch { .. } => SurvivalObs::ItemTorch,
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
            Item::CookedCarrot => Some((config.cooked_carrot_food, Achievement::EatCookedCarrot)),
            _ => None,
        }
    }

    /// What the item cooks into, held to a fire, if it cooks.
    pub(super) fn cooked(self) -> Option<(Item, Achievement)> {
        match self {
            Item::Berry => Some((Item::CookedBerry, Achievement::CookBerry)),
            Item::Carrot => Some((Item::CookedCarrot, Achievement::CookCarrot)),
            _ => None,
        }
    }
}

impl Survival {
    /// Sets `item` down on the ground at `position`; a torch keeps its wear
    /// there until it is taken up again.
    pub(super) fn lay(&mut self, position: Position, item: Item) {
        if let Item::Torch { burnt } = item {
            self.state.laid_torches.push((position, burnt));
        }
        self.set_ground(position, item.tile());
    }

    /// Takes up the item lying at `position`, leaving bare ground.
    pub(super) fn take(&mut self, position: Position) -> Item {
        let item = self.state.base_map[position.idx()]
            .item()
            .expect("only an item can be taken up");
        self.set_ground(position, SurvivalObs::TileEmpty);
        match item {
            Item::Torch { .. } => {
                let laid = &mut self.state.laid_torches;
                let i = laid
                    .iter()
                    .position(|&(at, _)| at == position)
                    .expect("every torch on the ground was laid there");
                Item::Torch {
                    burnt: laid.swap_remove(i).1,
                }
            }
            item => item,
        }
    }
}

/// Combining the item in hand with the backpack's makes the third, either
/// way round, into the hand.
pub(super) const RECIPES: &[(Item, Item, Item, Achievement)] = &[
    (Item::Stick, Item::Stone, Item::Axe, Achievement::MakeAxe),
    (
        Item::Wood,
        Item::Grass,
        Item::Campfire,
        Achievement::MakeCampfire,
    ),
    (
        Item::Stick,
        Item::Grass,
        Item::Torch { burnt: 0 },
        Achievement::MakeTorch,
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
/// leaves of it. Felling a tree and clearing a bush with an axe, and
/// harvesting tall grass by hand, are the first; digging walls and mining are
/// meant to join them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Job {
    Chop,
    ClearBush,
    Harvest,
}

impl Job {
    const ALL: [Job; 3] = [Job::Chop, Job::ClearBush, Job::Harvest];

    /// The job `tool` does on `tile`, if any; `None` is bare hands.
    pub(super) fn of(tool: Option<Item>, tile: SurvivalObs) -> Option<Job> {
        Self::ALL
            .into_iter()
            .find(|job| job.tool() == tool && job.works(tile))
    }

    fn tool(self) -> Option<Item> {
        match self {
            Job::Chop | Job::ClearBush => Some(Item::Axe),
            Job::Harvest => None,
        }
    }

    /// Whether the job works on `tile`.
    pub(super) fn works(self, tile: SurvivalObs) -> bool {
        use SurvivalObs::*;
        match self {
            Job::Chop => tile == TileTree,
            Job::ClearBush => matches!(tile, TileBerryBush | TileBush | TileDeadBush),
            Job::Harvest => tile == TileTallGrass,
        }
    }

    /// What the job leaves of its tile when it's done.
    pub(super) fn leaves(self) -> SurvivalObs {
        match self {
            Job::Chop => SurvivalObs::ItemWood,
            Job::ClearBush => SurvivalObs::ItemStick,
            Job::Harvest => SurvivalObs::ItemGrass,
        }
    }

    /// Steps the job takes, the action that starts it included.
    pub(super) fn steps(self, config: &SurvivalConfig) -> u32 {
        match self {
            Job::Chop => config.chop_steps,
            Job::ClearBush => config.clear_bush_steps,
            Job::Harvest => config.harvest_steps,
        }
    }

    pub(super) fn achievement(self) -> Achievement {
        match self {
            Job::Chop => Achievement::ChopTree,
            Job::ClearBush => Achievement::ClearBush,
            Job::Harvest => Achievement::HarvestGrass,
        }
    }
}

/// What using the item in hand, or bare hands, on the tile in front does.
#[derive(Debug, Clone, Copy)]
pub(super) enum Usage {
    /// Wood on a fire burning low builds it back up.
    Stoke,
    /// A campfire set down on open ground is lit there.
    Kindle,
    /// Food held to a fire cooks into the item, in hand.
    Cook(Item, Achievement),
    /// A tool starts its job, or bare hands one done by hand.
    Work(Job),
}

impl Usage {
    pub(super) fn of(held: Option<Item>, tile: SurvivalObs) -> Option<Usage> {
        match held {
            Some(Item::Wood) => (tile == SurvivalObs::TileFireLow).then_some(Usage::Stoke),
            Some(Item::Campfire) => tile.is_floor().then_some(Usage::Kindle),
            _ => match held.and_then(Item::cooked) {
                Some((cooked, achievement)) => {
                    tile.is_fire().then_some(Usage::Cook(cooked, achievement))
                }
                None => Job::of(held, tile).map(Usage::Work),
            },
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
