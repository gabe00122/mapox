//! Field of view: which cells a viewer standing on a grid can actually see.
//!
//! [`shadowcast`] is the algorithm on its own, over nothing but offsets and a
//! callback, so an env can point it at whatever it stores. [`encode_visible`]
//! is the wrapper every grid env with a centred observation window wants: copy
//! what is visible out of the map, leave the rest masked.

use ndarray::{Array2, ArrayViewMut2};

use super::Position;
use crate::vocab::VocabId;

/// The eight `(xx, xy, yx, yy)` transforms a shadowcast sweeps, one per
/// half-quadrant: a cell `depth` rows out and `lateral` columns off the centre
/// line sits at `(lateral * xx + depth * xy, lateral * yx + depth * yy)`.
/// Between them they tile the full circle, so the sweep only ever has to reason
/// about one wedge where `0 <= lateral <= depth`.
const OCTANTS: [(i32, i32, i32, i32); 8] = [
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (0, -1, 1, 0),
    (-1, 0, 0, 1),
    (-1, 0, 0, -1),
    (0, -1, -1, 0),
    (0, 1, -1, 0),
    (1, 0, 0, -1),
];

/// One wedge of a sweep: how to place a `(lateral, depth)` pair on the grid,
/// and how far the window reaches along each of those axes.
#[derive(Debug, Clone, Copy)]
struct Octant {
    xx: i32,
    xy: i32,
    yx: i32,
    yy: i32,
    max_depth: i32,
    max_lateral: i32,
}

impl Octant {
    fn new((xx, xy, yx, yy): (i32, i32, i32, i32), half_width: i32, half_height: i32) -> Self {
        // depth runs along x exactly when the depth term feeds the x offset
        let (max_depth, max_lateral) = if xy != 0 {
            (half_width, half_height)
        } else {
            (half_height, half_width)
        };

        Self {
            xx,
            xy,
            yx,
            yy,
            max_depth,
            max_lateral,
        }
    }

    fn offset(&self, lateral: i32, depth: i32) -> Position {
        Position::new(
            lateral * self.xx + depth * self.xy,
            lateral * self.yx + depth * self.yy,
        )
    }
}

/// Visits every cell a viewer can see inside the box reaching `half_width` and
/// `half_height` around it, calling `reveal` once per visible cell with that
/// cell's offset from the viewer. `reveal` answers whether the cell it was
/// handed blocks sight of what lies behind it.
///
/// This is recursive shadowcasting (Björn Bergström's algorithm): walk out row
/// by row carrying the slope range still lit, and every time a wall interrupts
/// that range, recurse on the slice above it and keep scanning below it. Each
/// visible cell is visited once, so a sweep costs O(cells in the box) rather
/// than tracing a ray per cell.
///
/// Two things the callers inherit. Cells whose corner is exactly tangent to a
/// wall's corner count as seen, so a lone wall never casts a perfectly clean
/// shadow at 45°. And the offsets handed out can reach `half_width` and
/// `half_height` in either direction, so a viewer near the edge of a map needs
/// that much padding around it (the envs get this from their wall border).
pub fn shadowcast(half_width: i32, half_height: i32, mut reveal: impl FnMut(Position) -> bool) {
    // the viewer always sees the cell it stands on; the sweep starts a ring out
    reveal(Position::new(0, 0));

    for &transform in &OCTANTS {
        let octant = Octant::new(transform, half_width, half_height);
        cast_light(&octant, 1, 1.0, 0.0, &mut reveal);
    }
}

/// Sweeps one octant over the slope range `end_slope..=start_slope`, starting
/// `first_depth` rows out.
fn cast_light(
    octant: &Octant,
    first_depth: i32,
    start_slope: f32,
    end_slope: f32,
    reveal: &mut impl FnMut(Position) -> bool,
) {
    if start_slope < end_slope {
        return;
    }
    // the beam narrows as walls eat into it, so this outlives the row loop
    let mut start_slope = start_slope;

    for depth in first_depth..=octant.max_depth {
        let mut blocked = false;
        let mut next_start = start_slope;

        // scan from the outer edge of the wedge inwards, i.e. from the steepest
        // slope down. Cells past the side of the window are left out: whatever
        // they would shadow is outside the window too.
        for lateral in (0..=depth.min(octant.max_lateral)).rev() {
            // slopes of this cell's outer and inner corners
            let outer_slope = (lateral as f32 + 0.5) / (depth as f32 - 0.5);
            let inner_slope = (lateral as f32 - 0.5) / (depth as f32 + 0.5);

            if start_slope < inner_slope {
                continue; // the beam has not reached this cell yet
            }
            if end_slope > outer_slope {
                break; // past the beam, and so is the rest of this row
            }

            let opaque = reveal(octant.offset(lateral, depth));

            if blocked {
                if opaque {
                    next_start = inner_slope;
                } else {
                    // the wall ended; the beam picks back up below it
                    blocked = false;
                    start_slope = next_start;
                }
            } else if opaque && depth < octant.max_depth {
                // split the beam: recurse on the slice above the wall and carry
                // on under it in this scan
                blocked = true;
                cast_light(octant, depth + 1, start_slope, outer_slope, reveal);
                next_start = inner_slope;
            }
        }

        if blocked {
            break; // the row ran out inside a wall, nothing deeper is lit
        }
    }
}

/// Copies the tiles a viewer at `center` can see out of `map` into `view`,
/// leaving everything hidden behind a wall as `mask`. `view` is the viewer's
/// observation window, centred on it, and `opaque` decides which tiles block
/// sight.
///
/// `map` must have at least half a window of padding around `center`, which is
/// the wall border the envs already keep.
pub fn encode_visible(
    map: &Array2<VocabId>,
    center: Position,
    view: &mut ArrayViewMut2<VocabId>,
    mask: VocabId,
    opaque: impl Fn(VocabId) -> bool,
) {
    view.fill(mask);

    let (width, height) = view.dim();
    let half_width = width as i32 / 2;
    let half_height = height as i32 / 2;

    shadowcast(half_width, half_height, |offset| {
        let tile = map[(center + offset).idx()];

        // an even-sized window runs [-half, half - 1], so the far row and
        // column fall outside it; those tiles still shadow what is behind them
        let x = (half_width + offset.x) as usize;
        let y = (half_height + offset.y) as usize;
        if let Some(cell) = view.get_mut([x, y]) {
            *cell = tile;
        }

        opaque(tile)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    const CLEAR: VocabId = 0;
    const WALL: VocabId = 1;
    const MASK: VocabId = 2;

    /// Reads a map out of ASCII art: `#` is a wall, `@` is the viewer, anything
    /// else is clear floor. Rows read top down, so the first line is the
    /// highest `y` — the way a map looks on screen rather than the way it is
    /// indexed.
    fn parse(rows: &[&str]) -> (Array2<VocabId>, Position) {
        let (width, height) = (rows[0].len(), rows.len());
        let mut map = Array2::from_elem((width, height), CLEAR);
        let mut viewer = None;

        for (row, line) in rows.iter().enumerate() {
            let y = height - 1 - row;
            for (x, glyph) in line.chars().enumerate() {
                match glyph {
                    '#' => map[[x, y]] = WALL,
                    '@' => viewer = Some(Position::new(x as i32, y as i32)),
                    _ => {}
                }
            }
        }

        (map, viewer.expect("the map marks the viewer with @"))
    }

    /// Sweeps a parsed map and draws back what the viewer sees: `#` wall, `.`
    /// floor, `?` hidden. The window is the whole map, so `rows` has to be odd
    /// sized with the viewer dead centre.
    fn seen(rows: &[&str]) -> String {
        let (map, viewer) = parse(rows);
        let (width, height) = map.dim();
        assert_eq!(
            (viewer.x, viewer.y),
            (width as i32 / 2, height as i32 / 2),
            "the viewer has to sit at the centre of the window"
        );

        let mut view = Array2::from_elem((width, height), MASK);
        encode_visible(&map, viewer, &mut view.view_mut(), MASK, |tile| {
            tile == WALL
        });

        (0..height)
            .map(|row| {
                let y = height - 1 - row;
                (0..width)
                    .map(|x| match view[[x, y]] {
                        WALL => '#',
                        MASK => '?',
                        _ => '.',
                    })
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn an_open_room_hides_nothing() {
        let view = seen(&[
            ".....", //
            ".....", "..@..", ".....", ".....",
        ]);

        assert_eq!(view, ".....\n.....\n.....\n.....\n.....");
    }

    /// A pillar casts a wedge that widens with distance — except right at the
    /// diagonals, where the beam grazes its corner and gets past.
    #[test]
    fn a_pillar_casts_a_widening_shadow() {
        let view = seen(&[
            ".......", //
            ".......", "...#...", "...@...", ".......", ".......", ".......",
        ]);

        assert_eq!(
            view,
            [
                "..???..", //
                "...?...", "...#...", ".......", ".......", ".......", ".......",
            ]
            .join("\n")
        );
    }

    /// The classic case: stood against a doorway you see a cone of the room
    /// beyond, widening with distance, and the wall hides the rest. The wall
    /// you are stood against is visible along its whole length, since every
    /// cell of it is grazed by a ray running down the face.
    #[test]
    fn a_doorway_admits_a_cone() {
        let view = seen(&[
            ".........", //
            ".........",
            ".........",
            "####.####",
            "....@....",
            ".........",
            ".........",
            ".........",
            ".........",
        ]);

        assert_eq!(
            view,
            [
                "??.....??", //
                "???...???",
                "???...???",
                "####.####",
                ".........",
                ".........",
                ".........",
                ".........",
                ".........",
            ]
            .join("\n")
        );
    }

    /// Standing in the corner of two walls: the one overhead hides the column
    /// behind it, the one alongside hides the row, and the two shadows meet.
    #[test]
    fn walls_shadow_independently() {
        let view = seen(&[
            "...#...", //
            "...#...", "...#...", "..#@...", ".......", ".......", ".......",
        ]);

        assert_eq!(
            view,
            [
                "..???..", //
                "...?...", "?..#...", "??#....", "?......", ".......", ".......",
            ]
            .join("\n")
        );
    }

    /// With nothing in the way the sweep has to hand back exactly the window a
    /// plain slice of the map would, whatever its shape — including an
    /// even-sized one, whose window runs `[-half, half - 1]` and so has no
    /// centre cell to sit on.
    #[test]
    fn an_open_view_matches_a_plain_window_copy() {
        // every cell distinct, so a misplaced one cannot pass unnoticed
        let map = Array2::from_shape_fn((32, 32), |(x, y)| (x * 32 + y) as VocabId);
        let center = Position::new(16, 16);

        for (width, height) in [(11, 11), (10, 8), (5, 9)] {
            let mut view = Array2::from_elem((width, height), MASK);
            encode_visible(&map, center, &mut view.view_mut(), MASK, |_| false);

            let x0 = center.x as usize - width / 2;
            let y0 = center.y as usize - height / 2;
            let window = map.slice(s![x0..x0 + width, y0..y0 + height]);

            assert_eq!(view, window, "{width}x{height} window");
        }
    }
}
