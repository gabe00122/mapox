//! The Urizen sprite sheet: where each tile lives in it, and how to draw one.

use egui::{Color32, ColorImage, Rect, TextureHandle, TextureOptions, Vec2, pos2};

/// Urizen Onebit Tileset by Vurmux — <https://vurmux.itch.io/urizen-onebit-tileset>
///
/// The sheet is baked into the binary rather than loaded from disk, so one
/// code path serves every host: the python extension has no asset directory to
/// point at, and wasm has no filesystem at all. The cost is ~250 KiB of
/// `.rodata` in every artifact.
const TILESET_PNG: &[u8] = include_bytes!("../../assets/urizen_onebit_tileset__v2d0.png");

/// Side of one tile, in sheet pixels.
const TILE_SIZE: f32 = 12.0;
/// Separator between tiles. It is also a border, so tile (0, 0) starts at (1, 1).
const TILE_PAD: f32 = 1.0;

/// Sheet is 2679x651 px = `COLS * 13 + 1` by `ROWS * 13 + 1`.
pub const TILESET_COLS: u32 = 206;
pub const TILESET_ROWS: u32 = 50;

/// The Urizen sheet, uploaded once and sampled per tile.
pub struct Tileset {
    texture: TextureHandle,
}

impl Tileset {
    /// Decodes and uploads the embedded sheet.
    ///
    /// Call from inside a frame (i.e. within [`eframe::App::ui`]): before the
    /// backend delivers input, the context still reports a placeholder max
    /// texture side of 2048 and debug builds assert the 2679px sheet against it.
    pub fn embedded(ctx: &egui::Context) -> Self {
        let rgba = image::load_from_memory_with_format(TILESET_PNG, image::ImageFormat::Png)
            .expect("embedded tileset is a valid png")
            .into_rgba8();
        let size = [rgba.width() as usize, rgba.height() as usize];
        let pixels = ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
        // Pixel art, and neighbouring tiles sit one pixel away: interpolating
        // would bleed the separator and the next sprite into every edge.
        let texture = ctx.load_texture("urizen-tileset", pixels, TextureOptions::NEAREST);
        Self { texture }
    }

    pub fn texture(&self) -> &TextureHandle {
        &self.texture
    }

    /// Sheet size in texels, which is also its size in points at 100% zoom.
    pub fn sheet_size(&self) -> Vec2 {
        self.texture.size_vec2()
    }

    /// Source rect of tile `(col, row)` in sheet pixels.
    fn source(col: u32, row: u32) -> Rect {
        Rect::from_min_size(
            pos2(
                col as f32 * (TILE_SIZE + TILE_PAD) + TILE_PAD,
                row as f32 * (TILE_SIZE + TILE_PAD) + TILE_PAD,
            ),
            Vec2::splat(TILE_SIZE),
        )
    }

    /// The same rect in the 0..=1 texture coordinates the painter samples with.
    fn uv(&self, col: u32, row: u32) -> Rect {
        let sheet = self.sheet_size();
        let src = Self::source(col, row);
        Rect::from_min_max(
            pos2(src.min.x / sheet.x, src.min.y / sheet.y),
            pos2(src.max.x / sheet.x, src.max.y / sheet.y),
        )
    }

    /// Draws tile `(col, row)` filling `rect`.
    pub fn draw(&self, painter: &egui::Painter, col: u32, row: u32, rect: Rect, tint: Color32) {
        painter.image(self.texture.id(), rect, self.uv(col, row), tint);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads width/height straight out of the PNG's IHDR, which is always the
    /// first chunk: 8 byte signature, 4 byte length, 4 byte type, then w/h as
    /// big-endian u32s.
    fn png_dimensions(bytes: &[u8]) -> (u32, u32) {
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "not a png");
        assert_eq!(&bytes[12..16], b"IHDR", "first chunk is not IHDR");
        let read = |at: usize| u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap());
        (read(16), read(20))
    }

    /// The tile counts are hardcoded, so pin them to the sheet we ship: a
    /// tileset update that changes the grid should fail here, not silently
    /// hand out source rects that run off the texture.
    ///
    /// This is also what lets callers treat `col < TILESET_COLS` as proof that
    /// a tile is on the sheet, instead of re-measuring the PNG themselves.
    #[test]
    fn tileset_grid_matches_the_embedded_sheet() {
        let (width, height) = png_dimensions(TILESET_PNG);
        let stride = (TILE_SIZE + TILE_PAD) as u32;

        assert_eq!(width, TILESET_COLS * stride + TILE_PAD as u32);
        assert_eq!(height, TILESET_ROWS * stride + TILE_PAD as u32);
    }

    /// Same formula as `SpriteSheet.image_at_tile` in the pygame renderer:
    /// `x * (tile_size + tile_pad) + tile_pad`.
    #[test]
    fn source_rects_match_the_python_layout() {
        let first = Tileset::source(0, 0);
        assert_eq!(
            (first.min.x, first.min.y, first.width(), first.height()),
            (1.0, 1.0, 12.0, 12.0)
        );

        // tile/wall, as used by both renderers.
        let wall = Tileset::source(20, 3);
        assert_eq!((wall.min.x, wall.min.y), (20.0 * 13.0 + 1.0, 3.0 * 13.0 + 1.0));
    }

    /// The last row and column have to land inside the texture, or every tile
    /// index the grid advertises is not actually addressable.
    #[test]
    fn the_last_tile_is_inside_the_sheet() {
        let (width, height) = png_dimensions(TILESET_PNG);
        let last = Tileset::source(TILESET_COLS - 1, TILESET_ROWS - 1);

        assert!(last.max.x <= width as f32);
        assert!(last.max.y <= height as f32);
    }
}
