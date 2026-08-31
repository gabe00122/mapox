use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub fn idx(self) -> [usize; 2] {
        [self.x as usize, self.y as usize]
    }
}

impl Add for Position {
    type Output = Position;

    fn add(self, rhs: Self) -> Self::Output {
        Position {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for Position {
    type Output = Position;

    fn sub(self, rhs: Self) -> Self::Output {
        Position {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Neg for Position {
    type Output = Position;

    fn neg(self) -> Self::Output {
        Position {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl Mul<i32> for Position {
    type Output = Position;

    fn mul(self, rhs: i32) -> Self::Output {
        Position {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Mul<Position> for Position {
    type Output = Position;

    fn mul(self, rhs: Position) -> Self::Output {
        Position {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}

impl Div<i32> for Position {
    type Output = Position;

    fn div(self, rhs: i32) -> Self::Output {
        Position {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl AddAssign for Position {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl SubAssign for Position {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl MulAssign<i32> for Position {
    fn mul_assign(&mut self, rhs: i32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl MulAssign<Position> for Position {
    fn mul_assign(&mut self, rhs: Position) {
        self.x *= rhs.x;
        self.y *= rhs.y;
    }
}

impl DivAssign<i32> for Position {
    fn div_assign(&mut self, rhs: i32) {
        self.x /= rhs;
        self.y /= rhs;
    }
}
