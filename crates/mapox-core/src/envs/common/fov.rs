//! Field of view: which cells a viewer standing on a grid can actually see.
//!
//! [`shadowcast`] is the algorithm on its own, over nothing but offsets and a
//! callback, so an env can point it at whatever it stores. [`cast_visible`] is
//! what every grid env with a centred observation window wants: from the
//! [`window`] of transparency around the viewer, mark which cells it sees, then
//! [`apply_mask`] to whatever was copied into its observation.

use ndarray::{Array2, ArrayView2, ArrayViewMut2, s};

use super::Position;

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

/// The `width` x `height` window of `map` centred on `center`, the same cells
/// a viewer there has in its observation window: slice tiles out of the map to
/// copy into the observation, or transparency to hand to [`cast_visible`].
///
/// An even-sized window runs `[-half, half - 1]` around `center`. `map` must
/// have at least half a window of padding around `center`, which is the wall
/// border the envs already keep.
pub fn window<T>(
    map: &Array2<T>,
    center: Position,
    (width, height): (usize, usize),
) -> ArrayView2<'_, T> {
    let x0 = center.x as usize - width / 2;
    let y0 = center.y as usize - height / 2;
    map.slice(s![x0..x0 + width, y0..y0 + height])
}

/// Marks in `visible` which cells of a window the viewer at its centre can
/// see, judging line of sight from `transparent`, the window's cells that let
/// sight through (see [`window`]). Both are window-sized; every cell of
/// `visible` is written, so it can be reused from one viewer to the next.
///
/// Nothing is masked here, so the caller owns every buffer and decides what a
/// hidden cell turns into: hand `visible` to [`apply_mask`] for each layer of
/// the observation.
pub fn cast_visible(transparent: ArrayView2<bool>, visible: &mut ArrayViewMut2<bool>) {
    debug_assert_eq!(transparent.dim(), visible.dim());
    visible.fill(false);

    let (width, height) = visible.dim();
    let half_width = width as i32 / 2;
    let half_height = height as i32 / 2;

    shadowcast(half_width, half_height, |offset| {
        let x = (half_width + offset.x) as usize;
        let y = (half_height + offset.y) as usize;

        // an even-sized window runs [-half, half - 1], so the sweep reaches one
        // row and column past it. Those cells could only shadow what lies
        // further out still, so treating them as clear changes nothing inside.
        match transparent.get([x, y]) {
            Some(&clear) => {
                visible[[x, y]] = true;
                !clear
            }
            None => false,
        }
    });
}

/// Overwrites with `mask` every cell of `view` that `visible` marks hidden.
pub fn apply_mask<T: Copy>(view: &mut ArrayViewMut2<T>, visible: ArrayView2<bool>, mask: T) {
    view.zip_mut_with(&visible, |cell, &seen| {
        if !seen {
            *cell = mask;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::VocabId;

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

        let transparent = map.mapv(|tile| tile != WALL);
        let mut visible = Array2::from_elem((width, height), false);
        cast_visible(
            window(&transparent, viewer, (width, height)),
            &mut visible.view_mut(),
        );

        let mut view = window(&map, viewer, (width, height)).to_owned();
        apply_mask(&mut view.view_mut(), visible.view(), MASK);

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
        let transparent = Array2::from_elem(map.dim(), true);
        let center = Position::new(16, 16);

        for (width, height) in [(11, 11), (10, 8), (5, 9)] {
            let mut visible = Array2::from_elem((width, height), false);
            cast_visible(
                window(&transparent, center, (width, height)),
                &mut visible.view_mut(),
            );

            let window = window(&map, center, (width, height));
            let mut view = window.to_owned();
            apply_mask(&mut view.view_mut(), visible.view(), MASK);

            assert_eq!(view, window, "{width}x{height} window");
        }
    }

    /// The visibility buffer is scratch shared between viewers, so whatever the
    /// last one saw must not leak into the next.
    #[test]
    fn a_reused_visibility_buffer_starts_clean() {
        let (map, viewer) = parse(&[
            ".......", //
            ".......", "...#...", "...@...", ".......", ".......", ".......",
        ]);
        let transparent = map.mapv(|tile| tile != WALL);

        let mut fresh = Array2::from_elem(map.dim(), false);
        cast_visible(
            window(&transparent, viewer, map.dim()),
            &mut fresh.view_mut(),
        );

        let mut reused = Array2::from_elem(map.dim(), true);
        cast_visible(
            window(&transparent, viewer, map.dim()),
            &mut reused.view_mut(),
        );

        assert_eq!(reused, fresh);
    }
}
