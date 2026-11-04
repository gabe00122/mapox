use crate::{
    envs::common::Direction,
    symbols::{
        COMBINE, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, NOOP, PUT, REFINE, SWAP1, SWAP2, SWAP3,
        SWAP4, TAKE, USE,
    },
    vocab_enum,
};

vocab_enum!(pub(super) SurvivalAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
    ItemTake => TAKE,
    ItemPut => PUT,
    ItemSwap1 => SWAP1,
    ItemSwap2 => SWAP2,
    ItemSwap3 => SWAP3,
    ItemSwap4 => SWAP4,
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
