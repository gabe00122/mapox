//! Field of view: which cells a viewer standing on a grid can actually see.
//!
//! [`shadowcast`] sweeps a grid from a source, asking one callback whether a
//! cell lets sight through and handing every cell it reaches to another, so an
//! env reads opacity straight off whatever it stores and does whatever it
//! likes with what is seen: fill an observation, light the
//! map, remember what was explored. [`observe`] is the common case built on
//! it: fill an agent's observation window with what it can see.
//!
//! Neither takes a radius. A consumer that wants one sweeps with
//! [`shadowcast`] and checks it in both callbacks (with [`within`], say),
//! answering that a cell past it blocks sight and leaving it out when it is
//! revealed: everything behind it is left out too, and the sweep stops there.

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
/// and how far the grid reaches along each of those axes in this wedge's
/// direction.
#[derive(Debug, Clone, Copy)]
struct Octant {
    source: Position,
    xx: i32,
    xy: i32,
    yx: i32,
    yy: i32,
    max_depth: i32,
    max_lateral: i32,
    /// Whether this wedge reveals the cells on its two edges, the centre line
    /// (`lateral == 0`) and the diagonal (`lateral == depth`). Each edge is
    /// shared with a neighbouring wedge that reaches exactly the same cells
    /// along it, so only one of the two reveals them.
    owns_edges: bool,
}

impl Octant {
    fn new(
        (xx, xy, yx, yy): (i32, i32, i32, i32),
        owns_edges: bool,
        source: Position,
        (width, height): (i32, i32),
    ) -> Self {
        // how far the grid runs from the source along an axis, given which way
        let reach_x = |sign: i32| {
            if sign > 0 {
                width - 1 - source.x
            } else {
                source.x
            }
        };
        let reach_y = |sign: i32| {
            if sign > 0 {
                height - 1 - source.y
            } else {
                source.y
            }
        };

        // depth runs along x exactly when the depth term feeds the x offset,
        // and lateral along x exactly when the lateral term does
        let max_depth = if xy != 0 { reach_x(xy) } else { reach_y(yy) };
        let max_lateral = if xx != 0 { reach_x(xx) } else { reach_y(yx) };

        Self {
            source,
            xx,
            xy,
            yx,
            yy,
            max_depth,
            max_lateral,
            owns_edges,
        }
    }

    fn reveals(&self, lateral: i32, depth: i32) -> bool {
        self.owns_edges || (lateral != 0 && lateral != depth)
    }

    fn cell(&self, lateral: i32, depth: i32) -> Position {
        self.source
            + Position::new(
                lateral * self.xx + depth * self.xy,
                lateral * self.yx + depth * self.yy,
            )
    }
}

/// Visits every cell of a `width` x `height` grid that a viewer at `source`
/// can see, calling `reveal` with each one exactly once. `is_transparent`
/// answers whether a cell lets sight through to what lies behind it, and may
/// be asked about the same cell more than once. Cells handed to either are
/// always on the grid.
///
/// To stop at a radius, have `is_transparent` answer that a cell past it
/// blocks, and `reveal` leave such a cell alone: nothing behind a cell is any
/// nearer the source than it, so the sweep reaches exactly the disk and goes
/// no further. Any disk works, the one [`within`] draws or the one
/// [`stamp_circle`] fills.
///
/// What a revealed cell means is up to the caller: copy a tile into an
/// observation, light a cell, remember it. For a viewer's observation window,
/// sweep the window itself with the viewer at its centre, `(width / 2,
/// height / 2)`, and offset into the map from there.
///
/// This is recursive shadowcasting (Björn Bergström's algorithm): walk out row
/// by row carrying the slope range still lit, and every time a wall interrupts
/// that range, recurse on the slice above it and keep scanning below it. A
/// sweep costs O(cells in range) rather than tracing a ray per cell. Cells on
/// the boundary between two octants (the axes and diagonals) are reached by
/// both, always alike, so only every other octant reveals them; within an
/// octant the beams split around a wall never overlap. So each cell is
/// revealed once without keeping track of which have been, and a sweep
/// allocates nothing.
///
/// Cells whose corner is exactly tangent to a wall's corner count as seen, so a
/// lone wall never casts a perfectly clean shadow at 45°.
///
/// [`stamp_circle`]: super::stamp::stamp_circle
pub fn shadowcast(
    source: Position,
    (width, height): (usize, usize),
    mut is_transparent: impl FnMut(Position) -> bool,
    mut reveal: impl FnMut(Position),
) {
    let dim = (width as i32, height as i32);
    debug_assert!(
        (0..dim.0).contains(&source.x) && (0..dim.1).contains(&source.y),
        "the source {source:?} has to be on the {width}x{height} grid"
    );
    // the viewer always sees the cell it stands on; the sweep starts a ring out
    reveal(source);

    // neighbouring octants share an edge, so every other one reveals them all
    for (i, &transform) in OCTANTS.iter().enumerate() {
        let octant = Octant::new(transform, i % 2 == 0, source, dim);
        cast_light(&octant, 1, 1.0, 0.0, &mut is_transparent, &mut reveal);
    }
}

/// Whether `offset` lies within `radius` of the origin, on the disk
/// `x² + y² <= r² + r`. That is the disk of cells whose centres lie within
/// `r + ½`, which reads rounder on a grid than `x² + y² <= r²` and its lone
/// cells poking out at the four tips. A negative radius takes in nothing.
pub fn within(offset: Position, radius: i32) -> bool {
    radius >= 0 && offset.x * offset.x + offset.y * offset.y <= radius * radius + radius
}

/// A tile an agent's view is drawn in: whether it blocks sight, and what an
/// observation shows in place of a tile out of sight.
pub trait ViewTile: Copy {
    /// What a hidden cell of an observation reads as.
    const MASK: Self;

    /// Whether the tile hides what lies behind it.
    fn opaque(self) -> bool;
}

/// Fills `view`, an observation window centred on `viewer`, with the tiles of
/// `map` the viewer can see, and [`ViewTile::MASK`] everywhere else.
///
/// An even-sized window runs `[-half, half - 1]` around `viewer`. `map` must
/// have at least half a window of padding around `viewer`, which is the wall
/// border the envs already keep.
pub fn observe<T: ViewTile + Into<V>, V: Copy>(
    map: &Array2<T>,
    viewer: Position,
    view: &mut ArrayViewMut2<V>,
) {
    let (width, height) = view.dim();
    // the sweep runs over the window itself, with the viewer at its centre
    let center = Position::new(width as i32 / 2, height as i32 / 2);
    let origin = viewer - center;

    view.fill(T::MASK.into());
    shadowcast(
        center,
        (width, height),
        |cell| !map[(origin + cell).idx()].opaque(),
        |cell| view[cell.idx()] = map[(origin + cell).idx()].into(),
    );
}

pub fn window<T>(
    map: &Array2<T>,
    center: Position,
    (width, height): (usize, usize),
) -> ArrayView2<'_, T> {
    let x0 = center.x as usize - width / 2;
    let y0 = center.y as usize - height / 2;
    map.slice(s![x0..x0 + width, y0..y0 + height])
}

/// Sweeps one octant over the slope range `end_slope..=start_slope`, starting
/// `first_depth` rows out.
fn cast_light(
    octant: &Octant,
    first_depth: i32,
    start_slope: f32,
    end_slope: f32,
    is_transparent: &mut impl FnMut(Position) -> bool,
    reveal: &mut impl FnMut(Position),
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
        // slope down. Cells past the side of the grid are left out: whatever
        // they would shadow is off the grid too.
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
            let cell = octant.cell(lateral, depth);
            if octant.reveals(lateral, depth) {
                reveal(cell);
            }
            let opaque = !is_transparent(cell);

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
                cast_light(
                    octant,
                    depth + 1,
                    start_slope,
                    outer_slope,
                    is_transparent,
                    reveal,
                );
                next_start = inner_slope;
            }
        }

        if blocked {
            break; // the row ran out inside a wall, nothing deeper is lit
        }
    }
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
    /// floor, `?` hidden.
    fn seen(rows: &[&str]) -> String {
        let (map, viewer) = parse(rows);
        let (width, height) = map.dim();

        let mut view = Array2::from_elem(map.dim(), MASK);
        shadowcast(
            viewer,
            map.dim(),
            |cell| map[cell.idx()] != WALL,
            |cell| view[cell.idx()] = map[cell.idx()],
        );

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

    /// Floor that never blocks sight, each cell telling its id.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Floor(VocabId);

    impl ViewTile for Floor {
        const MASK: Self = Floor(MASK);

        fn opaque(self) -> bool {
            false
        }
    }

    impl From<Floor> for VocabId {
        fn from(Floor(id): Floor) -> Self {
            id
        }
    }

    /// With nothing in the way an observation has to be exactly what a plain
    /// slice of the map would be, whatever its shape — including an even-sized
    /// one, whose window runs `[-half, half - 1]` and so has no centre cell to
    /// sit on.
    #[test]
    fn an_open_view_matches_a_plain_window_copy() {
        // every cell distinct, so a misplaced one cannot pass unnoticed
        let map = Array2::from_shape_fn((32, 32), |(x, y)| Floor((x * 32 + y) as VocabId));
        let viewer = Position::new(16, 16);

        for (width, height) in [(11, 11), (10, 8), (5, 9)] {
            let mut view = Array2::from_elem((width, height), MASK);
            observe(&map, viewer, &mut view.view_mut());

            let window = window(&map, viewer, (width, height)).mapv(|Floor(id)| id);
            assert_eq!(view, window, "{width}x{height} window");
        }
    }

    /// The viewer does not have to be central: off centre, the sweep still
    /// reaches every corner of the grid and never steps off it.
    #[test]
    fn an_off_centre_viewer_sees_the_whole_open_grid() {
        for source in [
            Position::new(0, 0),
            Position::new(8, 1),
            Position::new(3, 4),
        ] {
            let mut seen = Array2::from_elem((9, 5), false);
            let dim = seen.dim();
            shadowcast(source, dim, |_| true, |cell| seen[cell.idx()] = true);
            assert!(seen.iter().all(|&seen| seen), "from {source:?}");
        }
    }

    /// Lights `lit` with what a light at `source` reaches over `transparent`,
    /// out to the disk `in_range` draws around it, enforced the way any
    /// consumer would: a cell past it is left dark and blocks.
    fn light(
        transparent: &Array2<bool>,
        source: Position,
        in_range: impl Fn(Position) -> bool,
        lit: &mut Array2<bool>,
    ) {
        shadowcast(
            source,
            transparent.dim(),
            |cell| in_range(cell - source) && transparent[cell.idx()],
            |cell| {
                if in_range(cell - source) {
                    lit[cell.idx()] = true;
                }
            },
        );
    }

    /// However the octants meet on the axes and diagonals and however walls
    /// split the beam, every cell the sweep reaches is revealed exactly once
    /// and nothing else is.
    #[test]
    fn each_cell_reached_is_revealed_once() {
        use rand::{RngExt, SeedableRng, rngs::SmallRng};

        let mut rng = SmallRng::seed_from_u64(0);
        for _ in 0..2000 {
            let dim = (rng.random_range(1..24), rng.random_range(1..24));
            let density = rng.random_range(0.0..0.6);
            let opaque = Array2::from_shape_fn(dim, |_| rng.random_bool(density));
            let source = Position::new(
                rng.random_range(0..dim.0 as i32),
                rng.random_range(0..dim.1 as i32),
            );

            let mut reached = Array2::from_elem(dim, false);
            reached[source.idx()] = true;
            let mut reveals = Array2::from_elem(dim, 0);
            shadowcast(
                source,
                dim,
                |cell| {
                    reached[cell.idx()] = true;
                    !opaque[cell.idx()]
                },
                |cell| reveals[cell.idx()] += 1,
            );

            assert_eq!(
                reveals,
                reached.mapv(u32::from),
                "from {source:?} over {opaque:?}"
            );
        }
    }

    /// A light against a wall lights the wall but nothing behind it.
    #[test]
    fn a_wall_stops_a_light() {
        let (map, source) = parse(&[
            "...#...", //
            "...#...", "...#...", ".@.#...", "...#...", "...#...", "...#...",
        ]);
        let transparent = map.mapv(|tile| tile != WALL);

        let mut lit = Array2::from_elem(map.dim(), false);
        light(&transparent, source, |offset| within(offset, 3), &mut lit);

        assert!(lit[[3, 3]], "the wall itself is lit");
        assert!(
            (4..7).all(|x| (0..7).all(|y| !lit[[x, y]])),
            "behind the wall is dark"
        );
    }
}
