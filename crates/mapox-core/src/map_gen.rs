//! Noise-driven map generation, the rust counterpart of
//! `python/mapox/map_generator.py`. The jax side sums hand-rolled perlin
//! octaves (res `[2, 4, 5, 8, 10]`) with dirichlet-random amplitudes; here
//! [`fastnoise_lite`]'s FBm fractal plays the same role — five octaves at
//! lacunarity 2 cover the same frequency band, and a per-map random gain
//! stands in for the random amplitude mix. The maps are not bit-identical to
//! the jax ones, only statistically similar.

use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use ndarray::{Array2, ArrayViewMut2};
use rand::RngExt;

use crate::{envs::common::Position, vocab::VocabId};

/// Per-cell probability of each decor tile, matching
/// `map_generator.generate_decor_tiles` (the remaining 0.90 stays empty).
const DECOR_PROBS: [f64; 4] = [0.04, 0.04, 0.015, 0.005];

/// Fractal noise over a `width × height` grid, roughly in `[-1, 1]`.
/// Thresholding it yields connected blobby wall regions rather than salt
/// and pepper.
pub fn fractal_noise(width: usize, height: usize, rng: &mut impl RngExt) -> Array2<f32> {
    let mut noise = FastNoiseLite::with_seed(rng.random());
    noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    noise.set_fractal_type(Some(FractalType::FBm));
    noise.set_fractal_octaves(Some(5));
    // lowest octave spans the map about twice over, like the jax res = 2
    noise.set_frequency(Some(2.0 / width.min(height) as f32));
    noise.set_fractal_gain(Some(rng.random_range(0.4..0.7)));

    Array2::from_shape_fn((width, height), |(x, y)| {
        noise.get_noise_2d(x as f32, y as f32)
    })
}

/// Replaces a random sprinkle of `empty` cells with decor tiles, drawn with
/// [`DECOR_PROBS`]. Decor is cosmetic: envs treat it as walkable but don't
/// spawn anything on it.
pub fn sprinkle_decor(
    mut map: ArrayViewMut2<VocabId>,
    empty: VocabId,
    decor: &[VocabId; 4],
    rng: &mut impl RngExt,
) {
    for tile in map.iter_mut() {
        if *tile != empty {
            continue;
        }
        let mut roll: f64 = rng.random();
        for (&id, &prob) in decor.iter().zip(&DECOR_PROBS) {
            roll -= prob;
            if roll < 0.0 {
                *tile = id;
                break;
            }
        }
    }
}

/// Chooses `n` distinct cells whose tile equals `empty`, uniformly at random
/// (partial Fisher–Yates). Panics when the map has fewer than `n` such cells
/// — reachable only with an extreme wall threshold, and better loud than as
/// agents silently stacked inside walls.
pub fn choose_positions(
    map: &Array2<VocabId>,
    empty: VocabId,
    n: usize,
    rng: &mut impl RngExt,
) -> Vec<Position> {
    let mut cells: Vec<Position> = map
        .indexed_iter()
        .filter(|&(_, &tile)| tile == empty)
        .map(|((x, y), _)| Position {
            x: x as i32,
            y: y as i32,
        })
        .collect();
    assert!(
        cells.len() >= n,
        "need {n} spawn cells but the map only has {} empty ones; \
         is mapgen_threshold too low?",
        cells.len()
    );

    for i in 0..n {
        let j = rng.random_range(i..cells.len());
        cells.swap(i, j);
    }
    cells.truncate(n);
    cells
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    /// The wall threshold in env configs assumes noise stays around
    /// `[-1, 1]` and that a mid-range cutoff carves out a playable share of
    /// walls; a fastnoise config change that rescales the output would
    /// silently turn maps solid or empty otherwise.
    #[test]
    fn noise_range_and_wall_fraction_stay_sane() {
        for seed in 0..8 {
            let mut rng = StdRng::seed_from_u64(seed);
            let noise = fractal_noise(40, 40, &mut rng);

            for &value in &noise {
                assert!((-1.5..=1.5).contains(&value), "noise out of range: {value}");
            }

            let walls = noise.iter().filter(|&&v| v > 0.3).count();
            let fraction = walls as f64 / noise.len() as f64;
            assert!(
                (0.01..0.6).contains(&fraction),
                "wall fraction {fraction} at seed {seed}"
            );
        }
    }

    #[test]
    fn same_seed_same_map() {
        let a = fractal_noise(40, 40, &mut StdRng::seed_from_u64(7));
        let b = fractal_noise(40, 40, &mut StdRng::seed_from_u64(7));
        assert_eq!(a, b);
    }

    #[test]
    fn chosen_positions_are_distinct_and_empty() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut map = Array2::zeros((10, 10));
        map.row_mut(0).fill(1); // one wall stripe to dodge

        let positions = choose_positions(&map, 0, 50, &mut rng);
        assert_eq!(positions.len(), 50);
        let mut seen: Vec<_> = positions.iter().map(|p| (p.x, p.y)).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 50, "positions repeat");
        assert!(positions.iter().all(|p| p.x != 0), "spawned on a wall");
    }
}
