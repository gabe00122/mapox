use crate::{
    envs::common::Position,
    symbols::{MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, NOOP},
    vocab::{VocabId, Vocabulary},
    vocab_enum,
};

vocab_enum!(pub(super) SurvivalAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
    Noop => NOOP,
});

impl SurvivalAction {
    pub(super) fn direction(self) -> Position {
        use SurvivalAction::*;
        match self {
            MoveUp => Position::new(0, 1),
            MoveRight => Position::new(1, 0),
            MoveDown => Position::new(0, -1),
            MoveLeft => Position::new(-1, 0),
            Noop => Position::new(0, 0),
        }
    }

    pub(super) fn is_move(self) -> bool {
        !matches!(self, SurvivalAction::Noop)
    }
}
