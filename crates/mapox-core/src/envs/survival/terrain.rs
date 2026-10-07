//! Map generation: biomes from noise, joined up into one walkable region,
//! with spider nests set in the forest.

use ndarray::{Array2, s};
use rand::seq::SliceRandom;

use crate::envs::common::{
    Position,
    map_gen::{connect_regions, noise_field, quantile, roll},
};

use super::{
    Survival,
    tiles::{AROUND, SurvivalObs},
};

/// How far map generation bends its noise fields, in cells.
const TERRAIN_WARP: f32 = 8.0;
/// Pockets of open ground smaller than this are filled in rather than joined
/// up to the rest.
const MIN_REGION: usize = 24;

/// The kinds of land between the water and the rock.
#[derive(Debug, Clone, Copy)]
pub(super) enum Biome {
    Forest,
    Meadow,
    Scrub,
}

impl Biome {
    /// What the biome's ground grows, one roll per cell: dense trees with
    /// sticks under them in forest, berry bushes, tall grass and buried
    /// carrots in meadow, stones in scrub. Each also has its own decor, so
    /// bare ground shows which biome it is. Carrots are only ever what reset
    /// buries: once dug up they are gone.
    pub(super) fn growth(self) -> &'static [(SurvivalObs, f64)] {
        use SurvivalObs::*;
        match self {
            Biome::Forest => &[
                (TileTree, 0.30),
                (ItemStick, 0.04),
                (TileBerryBush, 0.004),
                (TileTallGrass, 0.02),
                (TileDecor4, 0.08),
            ],
            Biome::Meadow => &[
                (TileBerryBush, 0.02),
                (TileTallGrass, 0.08),
                (TileBuriedCarrot, 0.006),
                (TileTree, 0.015),
                (ItemStick, 0.008),
                (ItemStone, 0.004),
                (TileDecor1, 0.12),
            ],
            Biome::Scrub => &[
                (ItemStone, 0.05),
                (TileTallGrass, 0.03),
                (TileBuriedCarrot, 0.002),
                (TileTree, 0.008),
                (ItemStick, 0.008),
                (TileDecor2, 0.08),
                (TileDecor3, 0.03),
            ],
        }
    }
}

/// What map generation pays to dig a path through `tile` when it joins up
/// open ground; 0 for ground already open, bushes and grass included. Paths
/// would rather cut through forest than tunnel rock or bridge water.
fn dig_cost(tile: SurvivalObs) -> u32 {
    use SurvivalObs::*;
    match tile {
        _ if tile.walkable() => 0,
        TileTree => 1,
        TileWall | TileDestructibleWall => 2,
        TileWater => 3,
        _ => 4,
    }
}

impl Survival {
    /// Lays out the interior: an elevation field puts water in the low ground
    /// and rock on the heights, a moisture field splits the land between into
    /// biomes, each biome grows its own things, and the open ground is joined
    /// up.
    pub(super) fn generate_map(&mut self) {
        let (width, height) = (self.config.width as usize, self.config.height as usize);
        let rng = &mut self.state.rng;
        let elevation = noise_field(width, height, TERRAIN_WARP, rng);
        let moisture = noise_field(width, height, TERRAIN_WARP, rng);

        let water = quantile(elevation.iter().copied(), self.config.water_fraction);
        let rock = quantile(elevation.iter().copied(), 1.0 - self.config.rock_fraction);
        let land: Vec<f32> = elevation
            .iter()
            .zip(&moisture)
            .filter(|&(&e, _)| water <= e && e <= rock)
            .map(|(_, &m)| m)
            .collect();
        let dry = quantile(land.iter().copied(), self.config.scrub_fraction);
        let wet = quantile(land.iter().copied(), 1.0 - self.config.forest_fraction);

        let mut forest = Array2::from_elem((width, height), false);
        let mut interior = self.state.base_map.slice_mut(s![
            self.pad_width as usize..(self.width - self.pad_width) as usize,
            self.pad_height as usize..(self.height - self.pad_height) as usize,
        ]);
        for (cell, tile) in interior.indexed_iter_mut() {
            let (e, m) = (elevation[cell], moisture[cell]);
            *tile = if e < water {
                SurvivalObs::TileWater
            } else if e > rock {
                SurvivalObs::TileDestructibleWall
            } else {
                forest[cell] = m > wet;
                let biome = if m > wet {
                    Biome::Forest
                } else if m < dry {
                    Biome::Scrub
                } else {
                    Biome::Meadow
                };
                roll(biome.growth(), rng).unwrap_or(SurvivalObs::TileEmpty)
            };
        }

        connect_regions(interior, dig_cost, MIN_REGION, SurvivalObs::TileEmpty);
        self.place_nests(&forest);
    }

    /// Sets the spider nests on open ground, in the forest while it has room.
    /// A nest only goes where all eight cells around it are open, so it
    /// never cuts a path in two: there is always a way round it.
    fn place_nests(&mut self, forest: &Array2<bool>) {
        let corner = Position::new(self.pad_width, self.pad_height);
        let mut sites: Vec<Position> = forest
            .indexed_iter()
            .map(|((x, y), _)| corner + Position::new(x as i32, y as i32))
            .filter(|site| self.state.base_map[site.idx()].is_floor())
            .collect();
        sites.shuffle(&mut self.state.rng);
        // a stable sort, so each group stays shuffled
        sites.sort_by_key(|&site| !forest[(site - corner).idx()]);

        self.state.eggs.clear();
        for site in sites {
            if self.state.eggs.len() == self.config.num_spider_eggs {
                break;
            }
            let base_map = &self.state.base_map;
            if AROUND
                .iter()
                .all(|&d| base_map[(site + d).idx()].walkable())
            {
                self.state.base_map[site.idx()] = SurvivalObs::TileSpiderEggs;
                self.state.eggs.push(site);
            }
        }
    }
}
