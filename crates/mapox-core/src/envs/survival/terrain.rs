//! Map generation: biomes from noise, joined up into one walkable region.

use ndarray::s;
use rand::seq::SliceRandom;

use crate::envs::{
    common::{
        Position,
        map_gen::{connect_regions, noise_field, quantile, roll, spread_out},
    },
    survival::{Survival, SurvivalObs, items::ItemType, prototypes::Prototype},
};

/// How far map generation bends its noise fields, in cells.
const TERRAIN_WARP: f32 = 8.0;
/// Pockets of open ground smaller than this are filled in rather than joined
/// up to the rest.
const MIN_REGION: usize = 24;

/// The kinds of land between the water and the rock.
#[derive(Debug, Clone, Copy)]
enum Biome {
    Forest,
    Meadow,
    Scrub,
}

impl Biome {
    /// What the biome's ground grows, one roll per cell: dense trees with
    /// sticks under them in forest, berry bushes and tall grass in meadow,
    /// rocks in scrub. Each also has its own decor, so bare ground shows
    /// which biome it is.
    fn growth(self) -> &'static [(SurvivalObs, f64)] {
        use SurvivalObs::*;
        match self {
            Biome::Forest => &[
                (TileTree, 0.25),
                (ItemStick, 0.02),
                (TileBerryBush, 0.004),
                (TileTallGrass, 0.02),
                (TileDecor4, 0.08),
            ],
            Biome::Meadow => &[
                (TileBerryBush, 0.02),
                (TileTallGrass, 0.08),
                (TileTree, 0.012),
                (ItemStick, 0.004),
                (ItemRock, 0.002),
                (TileDecor1, 0.12),
            ],
            Biome::Scrub => &[
                (ItemRock, 0.025),
                (TileTallGrass, 0.03),
                (TileTree, 0.006),
                (ItemStick, 0.004),
                (TileDecor2, 0.08),
                (TileDecor3, 0.03),
            ],
        }
    }
}

/// The entity map generation's `tile` stands for, if it isn't terrain.
fn prototype(tile: SurvivalObs) -> Option<Prototype> {
    use SurvivalObs::*;
    Some(match tile {
        TileTree => Prototype::Tree,
        TileBerryBush => Prototype::BerryBush,
        TileTallGrass => Prototype::TallGrass,
        ItemStick => Prototype::Item(ItemType::Stick),
        ItemRock => Prototype::Item(ItemType::Rock),
        _ => return None,
    })
}

/// What map generation pays to dig a path through `tile` when it joins up
/// open ground; 0 for ground already open, bushes and grass included. Paths
/// would rather cut through forest than tunnel rock or bridge water.
fn dig_cost(tile: SurvivalObs) -> u32 {
    use SurvivalObs::*;
    match tile {
        // trees block by filling both slots, not by their tile
        TileTree => 1,
        _ if !tile.move_blocked() => 0,
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
        let rng = &mut self.state.rngs;
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

        let mut interior = self.state.tiles.slice_mut(s![
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
        self.lift_entities();
    }

    /// Map generation lays everything out on the terrain layer; this lifts
    /// what isn't terrain off it into entities.
    fn lift_entities(&mut self) {
        for x in self.pad_width..self.width - self.pad_width {
            for y in self.pad_height..self.height - self.pad_height {
                let position = Position::new(x, y);
                if let Some(prototype) = prototype(self.state.tiles[position.idx()]) {
                    self.state.tiles[position.idx()] = SurvivalObs::TileEmpty;
                    self.state.spawn_prototype(prototype, position);
                }
            }
        }
    }

    /// Where the agents start: on open ground with nothing on it, out of each
    /// other's sight where the map has room for it.
    pub(super) fn spawn_positions(&mut self) -> Vec<Position> {
        let mut free = Vec::new();
        for x in self.pad_width..self.width - self.pad_width {
            for y in self.pad_height..self.height - self.pad_height {
                let position = Position::new(x, y);
                if self.state.spatial_index[position.idx()] == [None; 2]
                    && !self.state.tiles[position.idx()].move_blocked()
                {
                    free.push(position);
                }
            }
        }
        free.shuffle(&mut self.state.rngs);

        let apart = self.pad_width.max(self.pad_height) + 1;
        let spawns = spread_out(&free, self.config.num_agents, apart);
        assert_eq!(
            spawns.len(),
            self.config.num_agents,
            "no open ground left to spawn on"
        );
        spawns
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::{
        env::Environment,
        envs::{common::map_gen::label_regions, survival::config::SurvivalConfig},
        timestep::TimeStepBuffers,
    };

    fn generated(seed: u64) -> Survival {
        let mut env = Survival::new(&SurvivalConfig::default(), 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(seed, &mut buffers.view_mut());
        env
    }

    #[test]
    fn a_seed_always_generates_the_same_map() {
        let (a, b) = (generated(7), generated(7));
        assert_eq!(a.state.render_map, b.state.render_map);
        assert_ne!(a.state.render_map, generated(8).state.render_map);
    }

    #[test]
    fn the_open_ground_is_one_region_with_every_agent_on_it() {
        for seed in 0..20 {
            let env = generated(seed);
            // trees fill both slots, so they block like walls do
            let open = Array2::from_shape_fn(env.state.tiles.dim(), |cell| {
                !env.state.tiles[cell].move_blocked()
                    && env.state.render_map[cell] != SurvivalObs::TileTree
            });
            let (labels, sizes) = label_regions(&open);
            assert_eq!(sizes.len(), 2, "seed {seed}: open ground is split up");

            for &agent in &env.state.agents {
                let position = *env.state.world.get::<&Position>(agent).unwrap();
                assert_eq!(labels[position.idx()], 1, "seed {seed}: agent off the map");
            }
        }
    }

    #[test]
    fn map_gen_grows_every_biome_entity() {
        let env = generated(0);
        for tile in [
            SurvivalObs::TileTree,
            SurvivalObs::TileBerryBush,
            SurvivalObs::TileTallGrass,
            SurvivalObs::ItemStick,
            SurvivalObs::ItemRock,
        ] {
            assert!(
                env.state.render_map.iter().any(|&t| t == tile),
                "no {tile:?} on the map"
            );
            assert!(
                env.state.tiles.iter().all(|&t| t != tile),
                "{tile:?} left on the terrain layer"
            );
        }
    }
}
