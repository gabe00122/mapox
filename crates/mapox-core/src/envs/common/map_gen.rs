use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use ndarray::ArrayViewMut2;
use rand::RngExt;

/// Per-cell probability of each decor tile, matching
/// `map_generator.generate_decor_tiles` (the remaining 0.90 stays empty).
const DECOR_PROBS: [f64; 4] = [0.04, 0.04, 0.015, 0.005];

pub fn fractal_noise<F>(width: usize, height: usize, rng: &mut impl RngExt, mut f: F)
where
    F: FnMut(usize, usize, f32),
{
    let mut noise = FastNoiseLite::with_seed(rng.random());
    noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    noise.set_fractal_type(Some(FractalType::FBm));
    noise.set_fractal_octaves(Some(5));
    // lowest octave spans the map about twice over, like the jax res = 2
    noise.set_frequency(Some(2.0 / width.min(height) as f32));
    noise.set_fractal_gain(Some(rng.random_range(0.4..0.7)));

    for y in 0..height {
        for x in 0..width {
            let sample = noise.get_noise_2d(x as f32, y as f32);
            f(x, y, sample);
        }
    }
}

pub fn sprinkle_decor<T>(mut map: ArrayViewMut2<T>, empty: T, decor: &[T], rng: &mut impl RngExt)
where
    T: Copy + PartialEq,
{
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
