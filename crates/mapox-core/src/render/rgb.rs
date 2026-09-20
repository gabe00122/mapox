use egui::{Pos2, Rect, vec2};

use crate::error::{MapoxError, MapoxResult};

use super::{
    env::{GridRenderSettings, GridRenderState},
    grid::GridLayout,
    resolve_art,
    tileset::{TILE_SIZE, Tileset},
};

const TILE_PIXELS: usize = (TILE_SIZE * TILE_SIZE) as usize;

pub(crate) struct RgbRenderer {
    sprites: Vec<[[u8; 3]; TILE_PIXELS]>,
    cols: usize,
    rows: usize,
    // Pixel centres mapped to (grid cell, sprite texel); None is letterboxing.
    xs: Vec<Option<(usize, u32)>>,
    ys: Vec<Option<(usize, u32)>>,
    pixels: Vec<u8>,
}

impl RgbRenderer {
    pub(crate) fn new(settings: &GridRenderSettings, width: u32, height: u32) -> MapoxResult<Self> {
        if settings.tile_width == 0 || settings.tile_height == 0 || width == 0 || height == 0 {
            return Err(MapoxError::InvalidConfig {
                reason: "empty video or map dimensions".into(),
            });
        }
        let len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| MapoxError::InvalidConfig {
                reason: "video dimensions overflow".into(),
            })?;
        let layout = GridLayout::fit(
            Rect::from_min_size(Pos2::ZERO, vec2(width as f32, height as f32)),
            settings.tile_width,
            settings.tile_height,
            1.0,
        );
        let axis = |length: u32, origin: f32, cells: usize, flip: bool| {
            (0..length)
                .map(|pixel| {
                    let position = (pixel as f32 + 0.5 - origin) / layout.tile;
                    if position < 0.0 || position >= cells as f32 {
                        return None;
                    }
                    let cell = position.floor() as usize;
                    let texel = ((position.fract() * TILE_SIZE) as u32).min(TILE_SIZE as u32 - 1);
                    Some((if flip { cells - 1 - cell } else { cell }, texel))
                })
                .collect()
        };
        let rgba = Tileset::decode();
        // Retain only used sprites, composited onto black once per clip.
        let sprites = resolve_art(&settings.obs_vocab)
            .into_iter()
            .map(|(col, row)| {
                let source = Tileset::source(col, row);
                std::array::from_fn(|i| {
                    let x = source.min.x as u32 + i as u32 % TILE_SIZE as u32;
                    let y = source.min.y as u32 + i as u32 / TILE_SIZE as u32;
                    let p = rgba.get_pixel(x, y).0;
                    std::array::from_fn(|c| ((u16::from(p[c]) * u16::from(p[3])) / 255) as u8)
                })
            })
            .collect();
        Ok(Self {
            sprites,
            cols: settings.tile_width,
            rows: settings.tile_height,
            xs: axis(width, layout.origin.x, settings.tile_width, false),
            ys: axis(height, layout.origin.y, settings.tile_height, true),
            pixels: vec![0; len],
        })
    }

    pub(crate) fn render(&mut self, state: &GridRenderState) -> MapoxResult<&[u8]> {
        if state.tilemap.dim() != (self.cols, self.rows)
            || state
                .tilemap
                .iter()
                .any(|&id| usize::from(id) >= self.sprites.len())
        {
            return Err(MapoxError::RenderSettingsMismatch);
        }
        let stride = self.xs.len() * 3;
        for (row, y) in self.pixels.chunks_exact_mut(stride).zip(&self.ys) {
            let Some((cell_y, texel_y)) = *y else {
                continue;
            };
            for (pixel, x) in row.chunks_exact_mut(3).zip(&self.xs) {
                let Some((cell_x, texel_x)) = *x else {
                    continue;
                };
                let sprite = &self.sprites[usize::from(state.tilemap[[cell_x, cell_y]])];
                pixel.copy_from_slice(&sprite[(texel_y * TILE_SIZE as u32 + texel_x) as usize]);
            }
        }
        Ok(&self.pixels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::{AGENT_SNAKE_BLUE, TILE_WALL};
    use ndarray::Array2;

    #[test]
    fn map_y_points_up_and_non_square_output_is_letterboxed() {
        let settings = GridRenderSettings {
            tile_width: 1,
            tile_height: 1,
            obs_vocab: [TILE_WALL, AGENT_SNAKE_BLUE].into_iter().collect(),
            ..Default::default()
        };
        let mut tile = RgbRenderer::new(&settings, 12, 12).unwrap();
        let mut state = GridRenderState {
            tilemap: Array2::from_elem((1, 1), 0),
            ..Default::default()
        };
        let wall = tile.render(&state).unwrap().to_vec();
        state.tilemap.fill(1);
        let blue = tile.render(&state).unwrap().to_vec();
        assert_ne!(wall, blue);

        let mut renderer = RgbRenderer::new(
            &GridRenderSettings {
                tile_height: 2,
                ..settings
            },
            24,
            24,
        )
        .unwrap();
        state.tilemap = Array2::from_shape_vec((1, 2), vec![0, 1]).unwrap();
        let frame = renderer.render(&state).unwrap();
        for y in 0..24 {
            let row = &frame[y * 72..(y + 1) * 72];
            assert!(row[..18].iter().chain(&row[54..]).all(|&p| p == 0));
            let expected = if y < 12 { &blue } else { &wall };
            assert_eq!(&row[18..54], &expected[(y % 12) * 36..(y % 12 + 1) * 36]);
        }
        state.tilemap.fill(2);
        assert!(matches!(
            renderer.render(&state).unwrap_err(),
            MapoxError::RenderSettingsMismatch
        ));
    }
}
