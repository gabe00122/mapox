//! The Urizen sprite sheet: where each tile lives in it, and how to draw one.

use macroquad::prelude::*;

/// Urizen Onebit Tileset by Vurmux — <https://vurmux.itch.io/urizen-onebit-tileset>
///
/// The sheet is baked into the binary rather than loaded through
/// `load_texture`, so one code path serves every host: the python extension
/// has no asset directory to point at, and wasm has no filesystem at all
/// (macroquad turns a load there into an HTTP fetch the page must serve).
/// The cost is ~250 KiB of `.rodata` in every artifact.
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
    texture: Texture2D,
}

impl Tileset {
    /// Decodes and uploads the embedded sheet.
    ///
    /// Needs a live graphics context, so call it from inside the macroquad
    /// window (i.e. within [`super::run`]), not before [`super::open_window`].
    pub fn embedded() -> Self {
        let texture = Texture2D::from_file_with_format(TILESET_PNG, Some(ImageFormat::Png));
        // Pixel art, and neighbouring tiles sit one pixel away: interpolating
        // would bleed the separator and the next sprite into every edge.
        texture.set_filter(FilterMode::Nearest);
        Self { texture }
    }

    pub fn texture(&self) -> &Texture2D {
        &self.texture
    }

    /// Source rect of tile `(col, row)` in sheet pixels.
    fn source(col: u32, row: u32) -> Rect {
        Rect::new(
            col as f32 * (TILE_SIZE + TILE_PAD) + TILE_PAD,
            row as f32 * (TILE_SIZE + TILE_PAD) + TILE_PAD,
            TILE_SIZE,
            TILE_SIZE,
        )
    }

    /// Draws tile `(col, row)` as a `size`x`size` square with its top-left at
    /// `(x, y)`.
    pub fn draw(&self, col: u32, row: u32, x: f32, y: f32, size: f32, tint: Color) {
        draw_texture_ex(
            &self.texture,
            x,
            y,
            tint,
            DrawTextureParams {
                dest_size: Some(Vec2::splat(size)),
                source: Some(Self::source(col, row)),
                ..Default::default()
            },
        );
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
        assert_eq!((first.x, first.y, first.w, first.h), (1.0, 1.0, 12.0, 12.0));

        // tile/wall, as used by both renderers.
        let wall = Tileset::source(20, 3);
        assert_eq!((wall.x, wall.y), (20.0 * 13.0 + 1.0, 3.0 * 13.0 + 1.0));
    }

    /// The last row and column have to land inside the texture, or every tile
    /// index the grid advertises is not actually addressable.
    #[test]
    fn the_last_tile_is_inside_the_sheet() {
        let (width, height) = png_dimensions(TILESET_PNG);
        let last = Tileset::source(TILESET_COLS - 1, TILESET_ROWS - 1);

        assert!(last.right() <= width as f32);
        assert!(last.bottom() <= height as f32);
    }
}
