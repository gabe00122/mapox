use ndarray::{Array2, s};

use super::Position;

pub fn stamp_circle<T: Clone>(grid: &mut Array2<T>, center: Position, radius: i32, value: T) {
    if radius < 0 {
        return;
    }
    let (width, height) = (grid.dim().0 as i32, grid.dim().1 as i32);

    let x_min = (center.x - radius).max(0);
    let x_max = (center.x + radius).min(width - 1);
    for x in x_min..=x_max {
        let dx = x - center.x;
        let half = (radius * radius - dx * dx).isqrt();
        let y_min = (center.y - half).max(0);
        let y_max = (center.y + half).min(height - 1);
        if y_min <= y_max {
            grid.slice_mut(s![x as usize, y_min as usize..=y_max as usize])
                .fill(value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation: test every cell.
    fn brute(dim: (usize, usize), center: Position, radius: i32) -> Array2<bool> {
        Array2::from_shape_fn(dim, |(x, y)| {
            let (dx, dy) = (x as i32 - center.x, y as i32 - center.y);
            dx * dx + dy * dy <= radius * radius
        })
    }

    #[test]
    fn matches_brute_force_including_clipping() {
        let dim = (13, 9);
        for radius in 0..8 {
            for center in [
                Position::new(6, 4),
                Position::new(0, 0),
                Position::new(12, 8),
                Position::new(-3, 4),
                Position::new(6, 15),
            ] {
                let mut grid = Array2::from_elem(dim, false);
                stamp_circle(&mut grid, center, radius, true);
                assert_eq!(grid, brute(dim, center, radius), "{center:?} r={radius}");
            }
        }
    }

    #[test]
    fn negative_radius_paints_nothing() {
        let mut grid = Array2::from_elem((5, 5), false);
        stamp_circle(&mut grid, Position::new(2, 2), -1, true);
        assert!(grid.iter().all(|&lit| !lit));
    }
}
