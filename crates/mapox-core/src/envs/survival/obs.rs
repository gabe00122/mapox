use ndarray::{Array2, ArrayViewMut2};

use crate::{
    envs::common::{
        Position,
        fov::{self, ViewTile},
    },
    symbols::{
        AGENT_GENERIC, TILE_DECOR_1, TILE_DECOR_2, TILE_DECOR_3, TILE_DECOR_4,
        TILE_DESTRUCTIBLE_WALL, TILE_EMPTY, TILE_MASK, TILE_UI, TILE_WALL, TILE_WATER,
    },
    vocab::VocabId,
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

pub(super) fn encode_view(
    map: &Array2<SurvivalObs>,
    lighting: &Array2<bool>,
    viewer: Position,
    night_vision_radius: i32,
    view: &mut ArrayViewMut2<VocabId>,
) {
    let (width, height) = view.dim();
    // the sweep runs over the window itself, with the viewer at its centre
    let center = Position::new(width as i32 / 2, height as i32 / 2);
    let origin = viewer - center;

    view.fill(SurvivalObs::MASK.into());
    fov::shadowcast(
        center,
        (width, height),
        |cell| !map[(origin + cell).idx()].opaque(),
        |cell| {
            let position = origin + cell;
            if lighting[position.idx()] || fov::within(cell - center, night_vision_radius) {
                view[cell.idx()] = map[position.idx()].into();
            }
        },
    );
}
