use crate::{
    envs::common::fov::ViewTile,
    symbols::{
        AGENT_GENERIC, TILE_DECOR_1, TILE_DECOR_2, TILE_DECOR_3, TILE_DECOR_4,
        TILE_DESTRUCTIBLE_WALL, TILE_EMPTY, TILE_MASK, TILE_UI, TILE_WALL, TILE_WATER,
    },
    vocab_enum,
};

vocab_enum!(pub(super) SurvivalObs {
    UI => TILE_UI,
    Mask => TILE_MASK,
    TileEmpty => TILE_EMPTY,
    TileDestructibleWall => TILE_DESTRUCTIBLE_WALL,
    TileWall => TILE_WALL,
    TileWater => TILE_WATER,
    TileDecor1 => TILE_DECOR_1,
    TileDecor2 => TILE_DECOR_2,
    TileDecor3 => TILE_DECOR_3,
    TileDecor4 => TILE_DECOR_4,
    AgentGeneric => AGENT_GENERIC,
});

impl SurvivalObs {
    pub(super) fn move_blocked(self) -> bool {
        use SurvivalObs::*;
        matches!(
            self,
            TileWall | TileDestructibleWall | TileWater | AgentGeneric
        )
    }

    pub(super) fn spawnable(self) -> bool {
        use SurvivalObs::*;
        matches!(
            self,
            TileEmpty | TileDecor1 | TileDecor2 | TileDecor3 | TileDecor4
        )
    }
}

impl ViewTile for SurvivalObs {
    const MASK: Self = SurvivalObs::Mask;

    /// Water is the one blocking tile an agent can see straight over.
    fn opaque(self) -> bool {
        use SurvivalObs::*;
        matches!(self, TileWall | TileDestructibleWall)
    }
}
