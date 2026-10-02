//! Grid-to-screen layout shared by both view modes. The env's y axis
//! points up and screen y points down; `cell_rect` applies that flip to
//! overlays, and `pos_to_cell` is its inverse.

use crate::render::tileset::TILE_SIZE;
use egui::{Pos2, Rect, pos2, vec2};

/// Maps a `cols x rows` grid (env coords: x right, y up) onto a screen
/// rect, centred: [`fill`](Self::fill) for the live view, [`fit`](Self::fit)
/// for video frames, where only nearest sampling is available.
pub(crate) struct GridLayout {
    /// Screen position of the grid's top-left corner.
    pub origin: Pos2,
    /// Side of one tile in points.
    pub tile: f32,
    pub cols: usize,
    pub rows: usize,
}

impl GridLayout {
    /// Scales the grid to the largest size that fits `area`, at whatever
    /// fractional tile size that takes. The tilemap shader filters texel
    /// edges by pixel coverage, so the art stays even without snapping.
    pub fn fill(area: Rect, cols: usize, rows: usize) -> Self {
        let tile = if cols == 0 || rows == 0 {
            0.0
        } else {
            (area.width() / cols as f32).min(area.height() / rows as f32)
        };
        let grid = vec2(cols as f32, rows as f32) * tile;
        Self {
            origin: area.center() - grid / 2.0,
            tile,
            cols,
            rows,
        }
    }

    /// Fits the grid into an `area` measured in pixels so every sprite
    /// texel covers a whole number of them: the art stays crisp at equal
    /// pixel widths instead of NEAREST sampling alternating between two
    /// sizes. The leftover space becomes symmetric letterbox padding.
    ///
    /// Below 1 sheet pixel per pixel (an area smaller than the grid at 1×)
    /// an integer scale cannot fit at all; the grid then takes the
    /// fractional fit and accepts the wobble over overflowing the area.
    pub fn fit(area: Rect, cols: usize, rows: usize) -> Self {
        if cols == 0 || rows == 0 {
            return Self {
                origin: area.center(),
                tile: 0.0,
                cols,
                rows,
            };
        }
        let fill = (area.width() / cols as f32).min(area.height() / rows as f32);
        let tile = if fill >= TILE_SIZE {
            (fill / TILE_SIZE).floor() * TILE_SIZE
        } else {
            fill
        };
        let grid = vec2(cols as f32, rows as f32) * tile;
        // With the tile already a whole number of pixels, a whole-pixel
        // origin puts every tile edge on the pixel grid; only the padding can
        // be a pixel lopsided, when the leftover space is odd.
        Self {
            origin: (area.center() - grid / 2.0).round(),
            tile,
            cols,
            rows,
        }
    }

    /// Rect of a `w x h` cell block whose min corner is grid cell `(x, y)`.
    pub fn cell_rect(&self, x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::from_min_size(
            pos2(
                self.origin.x + x * self.tile,
                self.origin.y + (self.rows as f32 - y - h) * self.tile,
            ),
            vec2(w, h) * self.tile,
        )
    }

    /// Grid cell under a screen position, `None` outside the grid.
    pub fn pos_to_cell(&self, pos: Pos2) -> Option<(usize, usize)> {
        if self.cols == 0 || self.rows == 0 || self.tile <= 0.0 {
            return None;
        }
        let col = ((pos.x - self.origin.x) / self.tile).floor();
        let row = ((pos.y - self.origin.y) / self.tile).floor();
        if col < 0.0 || row < 0.0 || col >= self.cols as f32 || row >= self.rows as f32 {
            return None;
        }
        Some((col as usize, self.rows - 1 - row as usize))
    }

    pub fn grid_rect(&self) -> Rect {
        self.cell_rect(0.0, 0.0, self.cols as f32, self.rows as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A click on the centre of a drawn cell must resolve back to that cell,
    /// or click-to-focus selects the wrong agent.
    #[test]
    fn pos_to_cell_inverts_cell_rect() {
        let area = Rect::from_min_size(pos2(13.0, 7.0), vec2(800.0, 600.0));
        let layout = GridLayout::fill(area, 40, 30);

        for x in [0, 1, 20, 39] {
            for y in [0, 1, 15, 29] {
                let center = layout.cell_rect(x as f32, y as f32, 1.0, 1.0).center();
                assert_eq!(layout.pos_to_cell(center), Some((x, y)), "cell ({x}, {y})");
            }
        }
    }

    #[test]
    fn positions_outside_the_grid_are_rejected() {
        let area = Rect::from_min_size(pos2(0.0, 0.0), vec2(600.0, 600.0));
        let layout = GridLayout::fill(area, 10, 10);

        let rect = layout.grid_rect();
        assert_eq!(layout.pos_to_cell(rect.min - vec2(1.0, 1.0)), None);
        assert_eq!(layout.pos_to_cell(rect.max + vec2(1.0, 1.0)), None);
    }

    /// Filling spends all of the limiting axis and centres the other.
    #[test]
    fn fill_uses_the_limiting_axis() {
        let area = Rect::from_min_size(pos2(5.0, 0.0), vec2(1000.0, 500.0));
        let layout = GridLayout::fill(area, 10, 12);
        assert_eq!(layout.grid_rect().height(), 500.0);
        assert_eq!(layout.grid_rect().center(), area.center());
    }

    /// The letterboxed grid stays centred, at a whole-pixel tile size:
    /// 50 px of available width snaps down to 4 sheet pixels per tile.
    #[test]
    fn fit_centers_the_grid() {
        let area = Rect::from_min_size(pos2(0.0, 0.0), vec2(1000.0, 500.0));
        let layout = GridLayout::fit(area, 10, 10);
        assert_eq!(layout.tile, 48.0);
        assert_eq!(layout.grid_rect().center(), area.center());
    }
}
