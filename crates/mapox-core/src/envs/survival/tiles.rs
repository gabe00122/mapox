//! The tile vocabulary, the rules every system reads off a tile, and the
//! geometry of the grid. Systems ask these rather than matching tiles
//! themselves, so a tile means the same thing to all of them.

use crate::{
    envs::common::Position,
    symbols::{
        AGENT_GENERIC_DOWN, AGENT_GENERIC_LEFT, AGENT_GENERIC_RIGHT, AGENT_GENERIC_UP,
        AGENT_SPIDER, ITEM_AXE, ITEM_BERRY, ITEM_CAMPFIRE, ITEM_CARROT, ITEM_COOKED_BERRY,
        ITEM_GRASS, ITEM_STICK, ITEM_STONE, ITEM_WOOD, TILE_BERRY_BUSH, TILE_BURIED_CARROT,
        TILE_BUSH, TILE_DEAD_BUSH, TILE_DECOR_1, TILE_DECOR_2, TILE_DECOR_3, TILE_DECOR_4,
        TILE_DESTRUCTIBLE_WALL, TILE_EMPTY, TILE_FIRE, TILE_FIRE_LOW, TILE_ICE, TILE_MASK,
        TILE_SPIDER_EGGS, TILE_TALL_GRASS, TILE_TREE, TILE_UI, TILE_WALL, TILE_WATER, UI_BACKPACK,
        UI_DAY, UI_DIGITS, UI_HANDS, UI_HEALTH, UI_HUNGER, UI_NIGHT, UI_TEMPERATURE, UI_WINTER,
    },
    vocab::{VocabId, Vocabulary},
    vocab_enum,
};

use super::items::Item;

vocab_enum!(pub(super) SurvivalObs {
    UI => TILE_UI,
    Mask => TILE_MASK,
    TileEmpty => TILE_EMPTY,
    TileWall => TILE_WALL,
    TileDestructibleWall => TILE_DESTRUCTIBLE_WALL,
    TileWater => TILE_WATER,
    TileIce => TILE_ICE,
    TileDecor1 => TILE_DECOR_1,
    TileDecor2 => TILE_DECOR_2,
    TileDecor3 => TILE_DECOR_3,
    TileDecor4 => TILE_DECOR_4,
    TileTree => TILE_TREE,
    TileBerryBush => TILE_BERRY_BUSH,
    TileBush => TILE_BUSH,
    TileDeadBush => TILE_DEAD_BUSH,
    TileTallGrass => TILE_TALL_GRASS,
    TileBuriedCarrot => TILE_BURIED_CARROT,
    TileFire => TILE_FIRE,
    TileFireLow => TILE_FIRE_LOW,
    TileSpiderEggs => TILE_SPIDER_EGGS,
    ItemStick => ITEM_STICK,
    ItemStone => ITEM_STONE,
    ItemWood => ITEM_WOOD,
    ItemBerry => ITEM_BERRY,
    ItemCookedBerry => ITEM_COOKED_BERRY,
    ItemCarrot => ITEM_CARROT,
    ItemGrass => ITEM_GRASS,
    ItemAxe => ITEM_AXE,
    ItemCampfire => ITEM_CAMPFIRE,
    AgentUp => AGENT_GENERIC_UP,
    AgentRight => AGENT_GENERIC_RIGHT,
    AgentDown => AGENT_GENERIC_DOWN,
    AgentLeft => AGENT_GENERIC_LEFT,
    Spider => AGENT_SPIDER,
    UiHealth => UI_HEALTH,
    UiHunger => UI_HUNGER,
    UiHands => UI_HANDS,
    UiBackpack => UI_BACKPACK,
    UiDay => UI_DAY,
    UiNight => UI_NIGHT,
    UiTemperature => UI_TEMPERATURE,
    UiWinter => UI_WINTER,
    Digit0 => UI_DIGITS[0],
    Digit1 => UI_DIGITS[1],
    Digit2 => UI_DIGITS[2],
    Digit3 => UI_DIGITS[3],
    Digit4 => UI_DIGITS[4],
    Digit5 => UI_DIGITS[5],
    Digit6 => UI_DIGITS[6],
    Digit7 => UI_DIGITS[7],
    Digit8 => UI_DIGITS[8],
    Digit9 => UI_DIGITS[9],
});

impl SurvivalObs {
    /// Bare open ground: where an item can be set down or an agent spawn.
    pub(super) fn is_floor(self) -> bool {
        use SurvivalObs::*;
        matches!(
            self,
            TileEmpty | TileDecor1 | TileDecor2 | TileDecor3 | TileDecor4
        )
    }

    /// Open ground, bare or with an item lying on it, low plants (bushes, tall
    /// grass, buried carrots) and ice. An agent standing on one hides it
    /// until it steps off; trees and fires stand in the way.
    pub(super) fn walkable(self) -> bool {
        use SurvivalObs::*;
        self.is_floor()
            || self.item().is_some()
            || matches!(
                self,
                TileBerryBush
                    | TileBush
                    | TileDeadBush
                    | TileTallGrass
                    | TileBuriedCarrot
                    | TileIce
            )
    }

    /// What winter's cold makes of the tile, if it changes it: bushes die,
    /// tall grass dies back to bare ground, and water freezes over. Buried
    /// carrots keep in the ground.
    pub(super) fn frozen(self) -> Option<SurvivalObs> {
        use SurvivalObs::*;
        match self {
            TileBerryBush | TileBush => Some(TileDeadBush),
            TileTallGrass => Some(TileEmpty),
            TileWater => Some(TileIce),
            _ => None,
        }
    }

    pub(super) fn is_fire(self) -> bool {
        matches!(self, SurvivalObs::TileFire | SurvivalObs::TileFireLow)
    }

    /// Agents and spiders: drawn over the ground they stand on.
    pub(super) fn is_creature(self) -> bool {
        AGENT_TILES.contains(&self) || self == SurvivalObs::Spider
    }

    pub(super) fn opaque(self) -> bool {
        matches!(
            self,
            SurvivalObs::TileWall | SurvivalObs::TileDestructibleWall
        )
    }

    /// The item lying here, if the tile is one.
    pub(super) fn item(self) -> Option<Item> {
        use SurvivalObs::*;
        Some(match self {
            ItemStick => Item::Stick,
            ItemStone => Item::Stone,
            ItemWood => Item::Wood,
            ItemBerry => Item::Berry,
            ItemCookedBerry => Item::CookedBerry,
            ItemCarrot => Item::Carrot,
            ItemGrass => Item::Grass,
            ItemAxe => Item::Axe,
            ItemCampfire => Item::Campfire,
            _ => return None,
        })
    }
}

/// The agent's tile by facing, in `DIRECTIONS` order.
pub(super) const AGENT_TILES: [SurvivalObs; 4] = [
    SurvivalObs::AgentUp,
    SurvivalObs::AgentRight,
    SurvivalObs::AgentDown,
    SurvivalObs::AgentLeft,
];

/// Facing offsets in `MOVES` order; env +y is up.
pub(super) const DIRECTIONS: [Position; 4] = [
    Position { x: 0, y: 1 },
    Position { x: 1, y: 0 },
    Position { x: 0, y: -1 },
    Position { x: -1, y: 0 },
];

/// The eight cells around one.
pub(super) const AROUND: [Position; 8] = [
    Position { x: -1, y: -1 },
    Position { x: 0, y: -1 },
    Position { x: 1, y: -1 },
    Position { x: -1, y: 0 },
    Position { x: 1, y: 0 },
    Position { x: -1, y: 1 },
    Position { x: 0, y: 1 },
    Position { x: 1, y: 1 },
];

/// Whether two cells share a side.
pub(super) fn beside(a: Position, b: Position) -> bool {
    (a.x - b.x).abs() + (a.y - b.y).abs() == 1
}

/// Whether `offset` lies inside a disc of `radius` cells; the `+ radius`
/// rounds the disc out so small ones aren't diamonds.
pub(super) fn within(offset: Position, radius: i32) -> bool {
    offset.x * offset.x + offset.y * offset.y <= radius * radius + radius
}
