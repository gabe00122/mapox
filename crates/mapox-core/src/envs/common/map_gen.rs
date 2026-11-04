use std::{cmp::Reverse, collections::BinaryHeap};

use fastnoise_lite::{DomainWarpType, FastNoiseLite, FractalType, NoiseType};
use ndarray::{Array2, ArrayViewMut2};
use rand::RngExt;

use super::Position;

/// Per-cell probability of each decor tile, matching
/// `map_generator.generate_decor_tiles` (the remaining 0.90 stays empty).
const DECOR_PROBS: [f64; 4] = [0.04, 0.04, 0.015, 0.005];

/// The 4-neighbourhood, which is how agents walk.
const STEPS: [(isize, isize); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];

fn fractal(width: usize, height: usize, rng: &mut impl RngExt) -> FastNoiseLite {
    let mut noise = FastNoiseLite::with_seed(rng.random());
    noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    noise.set_fractal_type(Some(FractalType::FBm));
    noise.set_fractal_octaves(Some(5));
    // lowest octave spans the map about twice over, like the jax res = 2
    noise.set_frequency(Some(2.0 / width.min(height) as f32));
    noise.set_fractal_gain(Some(rng.random_range(0.4..0.7)));
    noise
}

pub fn fractal_noise<F>(width: usize, height: usize, rng: &mut impl RngExt, mut f: F)
where
    F: FnMut(usize, usize, f32),
{
    let noise = fractal(width, height, rng);

    for y in 0..height {
        for x in 0..width {
            let sample = noise.get_noise_2d(x as f32, y as f32);
            f(x, y, sample);
        }
    }
}

/// The same fractal noise as [`fractal_noise`], as a `[x, y]` field, with
/// every sample taken up to `warp` cells away from its own cell along a
/// second noise field. Warping bends the field's contours, so thresholds cut
/// winding coasts and ridges out of it instead of round blobs; a `warp` of 0
/// leaves it unbent.
pub fn noise_field(width: usize, height: usize, warp: f32, rng: &mut impl RngExt) -> Array2<f32> {
    let noise = fractal(width, height, rng);

    let mut warper = FastNoiseLite::with_seed(rng.random());
    warper.set_domain_warp_type(Some(DomainWarpType::OpenSimplex2));
    warper.set_domain_warp_amp(Some(warp));
    warper.set_fractal_type(Some(FractalType::DomainWarpProgressive));
    warper.set_fractal_octaves(Some(3));
    // bends a little finer than the field's own largest features
    warper.set_frequency(Some(4.0 / width.min(height) as f32));

    Array2::from_shape_fn((width, height), |(x, y)| {
        let (x, y) = warper.domain_warp_2d(x as f32, y as f32);
        noise.get_noise_2d(x, y)
    })
}

/// The value `fraction` of `values` lie below: 0 is the least of them, 1
/// the greatest. Thresholding a noise field at its own quantiles gets the
/// same share of each terrain on every map, whatever the field's spread.
/// Empty input gives 0.
pub fn quantile(values: impl IntoIterator<Item = f32>, fraction: f64) -> f32 {
    let mut values: Vec<f32> = values.into_iter().collect();
    if values.is_empty() {
        return 0.0;
    }
    let rank = ((values.len() - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize;
    *values.select_nth_unstable_by(rank, f32::total_cmp).1
}

pub fn sprinkle_decor<T>(map: ArrayViewMut2<T>, empty: T, decor: &[T], rng: &mut impl RngExt)
where
    T: Copy + PartialEq,
{
    let table: Vec<(T, f64)> = decor.iter().copied().zip(DECOR_PROBS).collect();
    scatter(map, empty, &table, rng);
}

/// Turns each `empty` cell into at most one of `table`'s tiles, per [`roll`].
pub fn scatter<T>(mut map: ArrayViewMut2<T>, empty: T, table: &[(T, f64)], rng: &mut impl RngExt)
where
    T: Copy + PartialEq,
{
    for tile in map.iter_mut() {
        if *tile == empty
            && let Some(id) = roll(table, rng)
        {
            *tile = id;
        }
    }
}

/// One of `table`'s tiles, each with its own probability, or none with
/// whatever probability is left over; so the probabilities must sum to at
/// most one.
pub fn roll<T: Copy>(table: &[(T, f64)], rng: &mut impl RngExt) -> Option<T> {
    let mut roll: f64 = rng.random();
    for &(id, prob) in table {
        roll -= prob;
        if roll < 0.0 {
            return Some(id);
        }
    }
    None
}

/// Numbers the 4-connected regions of `passable` cells from 1, leaving
/// impassable cells 0. Returns the labels and each region's size, indexed by
/// label (index 0 is always 0).
pub fn label_regions(passable: &Array2<bool>) -> (Array2<u32>, Vec<usize>) {
    let (width, height) = passable.dim();
    let mut labels = Array2::zeros((width, height));
    let mut sizes = vec![0];
    let mut stack = Vec::new();

    for ((x, y), &open) in passable.indexed_iter() {
        if !open || labels[[x, y]] != 0 {
            continue;
        }
        let label = sizes.len() as u32;
        let mut size = 0;
        labels[[x, y]] = label;
        stack.push((x, y));
        while let Some(cell) = stack.pop() {
            size += 1;
            for next in neighbours(cell, (width, height)) {
                if passable[next] && labels[next] == 0 {
                    labels[next] = label;
                    stack.push(next);
                }
            }
        }
        sizes.push(size);
    }

    (labels, sizes)
}

/// Joins all the open ground of `map` into one region an agent can walk
/// across. `cost(tile)` is 0 for open ground, and otherwise what it costs to
/// dig a way through the tile; it must be 0 for `carve`.
///
/// Regions smaller than `min_region` cells are filled in rather than joined,
/// each with the tile most common around its edge: a pinhole in rock becomes
/// rock, a clearing hemmed in by trees becomes trees. Every other region is
/// joined to the largest by the cheapest path through what lies between, and
/// every tile on that path becomes `carve`.
pub fn connect_regions<T>(
    mut map: ArrayViewMut2<T>,
    cost: impl Fn(T) -> u32,
    min_region: usize,
    carve: T,
) where
    T: Copy + PartialEq,
{
    debug_assert_eq!(cost(carve), 0, "a carved path has to be open ground");
    let dim = map.dim();

    let (labels, sizes) = label_regions(&map.map(|&tile| cost(tile) == 0));
    let largest = largest_region(&sizes);
    let small = |label: u32| label != 0 && label != largest && sizes[label as usize] < min_region;

    // what borders each small region, and how often
    let mut borders: Vec<Vec<(T, usize)>> = vec![Vec::new(); sizes.len()];
    for (cell, &label) in labels.indexed_iter() {
        if !small(label) {
            continue;
        }
        for next in neighbours(cell, dim) {
            if labels[next] != 0 {
                continue;
            }
            let tally = &mut borders[label as usize];
            match tally.iter_mut().find(|(tile, _)| *tile == map[next]) {
                Some((_, count)) => *count += 1,
                None => tally.push((map[next], 1)),
            }
        }
    }
    for (cell, &label) in labels.indexed_iter() {
        if let Some(&(fill, _)) = borders[label as usize].iter().max_by_key(|(_, n)| *n) {
            map[cell] = fill;
        }
    }

    // Join the rest one at a time, nearest first, relabelling after each.
    loop {
        let (labels, sizes) = label_regions(&map.map(|&tile| cost(tile) == 0));
        if sizes.iter().filter(|&&size| size > 0).count() <= 1 {
            return;
        }
        let main = largest_region(&sizes);

        // Dijkstra out of the main region; open ground is free to cross
        let mut dist = Array2::from_elem(dim, u32::MAX);
        let mut parent = Array2::from_elem(dim, (usize::MAX, usize::MAX));
        let mut frontier = BinaryHeap::new();
        for (cell, &label) in labels.indexed_iter() {
            if label == main {
                dist[cell] = 0;
                frontier.push(Reverse((0, cell)));
            }
        }

        let mut reached = None;
        while let Some(Reverse((d, cell))) = frontier.pop() {
            if d > dist[cell] {
                continue;
            }
            if labels[cell] != 0 && labels[cell] != main {
                reached = Some(cell);
                break;
            }
            for next in neighbours(cell, dim) {
                let step = if labels[next] != 0 {
                    0
                } else {
                    cost(map[next])
                };
                let d = d.saturating_add(step);
                if d < dist[next] {
                    dist[next] = d;
                    parent[next] = cell;
                    frontier.push(Reverse((d, next)));
                }
            }
        }

        let Some(mut cell) = reached else {
            return;
        };
        while labels[cell] != main {
            if labels[cell] == 0 {
                map[cell] = carve;
            }
            cell = parent[cell];
        }
    }
}

/// Up to `n` of `candidates`, taken in order but skipping any closer than
/// `min_distance` cells (along either axis) to one already taken. If too few
/// are that far apart, the rest are the skipped ones, in order.
pub fn spread_out(candidates: &[Position], n: usize, min_distance: i32) -> Vec<Position> {
    let mut picked: Vec<Position> = Vec::with_capacity(n);
    let mut skipped = Vec::new();
    for &candidate in candidates {
        if picked.len() == n {
            break;
        }
        let crowded = picked.iter().any(|p| {
            (p.x - candidate.x).abs() < min_distance && (p.y - candidate.y).abs() < min_distance
        });
        if crowded {
            skipped.push(candidate);
        } else {
            picked.push(candidate);
        }
    }
    let missing = n - picked.len();
    picked.extend(skipped.into_iter().take(missing));
    picked
}

fn largest_region(sizes: &[usize]) -> u32 {
    (0..sizes.len())
        .max_by_key(|&label| sizes[label])
        .unwrap_or(0) as u32
}

fn neighbours(
    (x, y): (usize, usize),
    (width, height): (usize, usize),
) -> impl Iterator<Item = (usize, usize)> {
    STEPS.iter().filter_map(move |&(dx, dy)| {
        let x = x.checked_add_signed(dx).filter(|&x| x < width)?;
        let y = y.checked_add_signed(dy).filter(|&y| y < height)?;
        Some((x, y))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Reads a map out of ASCII art, row by row; `#` is wall, `~` water, and
    /// anything else open ground.
    fn grid(art: &[&str]) -> Array2<char> {
        let height = art.len();
        let width = art[0].len();
        Array2::from_shape_fn((width, height), |(x, y)| art[y].as_bytes()[x] as char)
    }

    fn cost(tile: char) -> u32 {
        match tile {
            '#' => 2,
            '~' => 5,
            _ => 0,
        }
    }

    fn regions(map: &Array2<char>) -> usize {
        let (_, sizes) = label_regions(&map.map(|&tile| cost(tile) == 0));
        sizes.iter().filter(|&&size| size > 0).count()
    }

    #[test]
    fn regions_are_four_connected() {
        let map = grid(&[
            "..#..", //
            "..#..", "##.##", "..#..",
        ]);
        let (labels, sizes) = label_regions(&map.map(|&tile| cost(tile) == 0));
        // the middle cell touches the others only at their corners
        assert_eq!(sizes.len(), 6);
        assert_eq!(sizes[1..].iter().sum::<usize>(), 13);
        assert_eq!(labels[[0, 0]], labels[[1, 1]]);
        assert_ne!(labels[[0, 0]], labels[[2, 2]]);
    }

    /// A pocket below the size limit fills with what surrounds it, and the
    /// big regions are joined through the cheaper wall rather than the water.
    #[test]
    fn small_pockets_fill_and_big_regions_join() {
        let mut map = grid(&[
            "....#....", //
            "....#....",
            "....#....",
            "~~~~#~~~~",
            "#########",
            "##.######",
            "#########",
        ]);
        connect_regions(map.view_mut(), cost, 3, '.');

        assert_eq!(
            map[[2, 5]],
            '#',
            "the pinhole fills with the rock around it"
        );
        assert_eq!(regions(&map), 1);
        let dug = map.iter().filter(|&&tile| tile == '.').count() - 24;
        assert_eq!(dug, 1, "one wall cell joins the two halves");
        assert_eq!(map.iter().filter(|&&tile| tile == '~').count(), 8);
    }

    #[test]
    fn a_connected_map_is_left_alone() {
        let before = grid(&[
            "..#..", //
            ".....", "~~#..",
        ]);
        let mut map = before.clone();
        connect_regions(map.view_mut(), cost, 3, '.');
        assert_eq!(map, before);
    }

    #[test]
    fn quantiles_pick_by_rank() {
        let values = [5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(quantile(values, 0.0), 1.0);
        assert_eq!(quantile(values, 0.5), 3.0);
        assert_eq!(quantile(values, 1.0), 5.0);
        assert_eq!(quantile([], 0.5), 0.0);
    }

    #[test]
    fn spread_out_keeps_its_distance_until_it_cannot() {
        let candidates: Vec<Position> = (0..10).map(|x| Position::new(x, 0)).collect();
        assert_eq!(
            spread_out(&candidates, 3, 4),
            vec![
                Position::new(0, 0),
                Position::new(4, 0),
                Position::new(8, 0)
            ]
        );
        // only three fit four apart; the fourth is the first one skipped
        assert_eq!(spread_out(&candidates, 4, 4)[3], Position::new(1, 0));
    }

    #[test]
    fn a_warped_field_has_the_requested_shape_and_range() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        let field = noise_field(30, 20, 6.0, &mut rng);
        assert_eq!(field.dim(), (30, 20));
        assert!(field.iter().all(|v| (-1.0..=1.0).contains(v)));
        assert!(quantile(field.iter().copied(), 0.9) > quantile(field.iter().copied(), 0.1));
    }
}
