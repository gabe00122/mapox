use crate::envs::common::Position;

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Up = 0,
    Right = 1,
    Down = 2,
    Left = 3,
}

impl Direction {
    pub fn to_pos(&self) -> Position {
        match self {
            Direction::Up => Position::new(0, 1),
            Direction::Right => Position::new(1, 0),
            Direction::Down => Position::new(0, -1),
            Direction::Left => Position::new(-1, 0),
        }
    }
}
