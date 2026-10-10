use crate::{
    envs::common::Direction,
    symbols::{
        COMBINE, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, NOOP, PUT, REFINE, SWAP, TAKE, USE,
    },
    vocab_enum,
};

// pub const TAKE: &str = "item/take";
// pub const PUT: &str = "item/put";
// pub const SWAP: &str = "item/swap";
// pub const COMBINE: &str = "item/combine";
// pub const USE: &str = "item/use";

vocab_enum!(pub(super) SurvivalAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
    ItemTake => TAKE,
    ItemPut => PUT,
    ItemSwap => SWAP,
    ItemCombine => COMBINE,
    ItemUse => USE,
    ItemRefine => REFINE,
    Noop => NOOP,
});

impl SurvivalAction {
    pub(super) fn move_direction(self) -> Option<Direction> {
        use SurvivalAction::*;
        match self {
            MoveUp => Some(Direction::Up),
            MoveRight => Some(Direction::Right),
            MoveDown => Some(Direction::Down),
            MoveLeft => Some(Direction::Left),
            _ => None,
        }
    }
}
