//! Grid-to-screen layout and the one tile-drawing loop both view modes
//! share. The env's y axis points up and screen y points down; `cell_rect`
//! is the single place that flip lives, and `pos_to_cell` is its inverse.

use egui::{Color32, Pos2, Rect, pos2, vec2};

use crate::render::tileset::Tileset;

/// Maps a `cols x rows` grid (env coords: x right, y up) onto a screen
/// rect, scaled to fit and centred.
pub(crate) struct GridLayout {
    /// Screen position of the grid's top-left corner.
    pub origin: Pos2,
    /// Side of one tile in points.
    pub tile: f32,
    pub cols: usize,
    pub rows: usize,
}

impl GridLayout {
    pub fn fit(area: Rect, cols: usize, rows: usize) -> Self {
        let tile = (area.width() / cols as f32).min(area.height() / rows as f32);
        let origin = area.center() - vec2(cols as f32, rows as f32) * tile / 2.0;
        Self {
            origin,
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

/// Draws every cell of the grid; `id_at(x, y)` returns the obs-vocab id of
/// the cell and `art` maps that id to sheet coordinates.
pub(crate) fn draw_tile_grid(
    painter: &egui::Painter,
    tileset: &Tileset,
    art: &[(u32, u32)],
    layout: &GridLayout,
    mut id_at: impl FnMut(usize, usize) -> u8,
) {
    for x in 0..layout.cols {
        for y in 0..layout.rows {
            let (col, row) = art[id_at(x, y) as usize];
            tileset.draw(
                painter,
                col,
                row,
                layout.cell_rect(x as f32, y as f32, 1.0, 1.0),
                Color32::WHITE,
            );
        }
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
        let layout = GridLayout::fit(area, 40, 30);

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
        let layout = GridLayout::fit(area, 10, 10);

        let rect = layout.grid_rect();
        assert_eq!(layout.pos_to_cell(rect.min - vec2(1.0, 1.0)), None);
        assert_eq!(layout.pos_to_cell(rect.max + vec2(1.0, 1.0)), None);
    }

    /// The letterboxed grid stays centred in the area it was fit to.
    #[test]
    fn fit_centers_the_grid() {
        let area = Rect::from_min_size(pos2(0.0, 0.0), vec2(1000.0, 500.0));
        let layout = GridLayout::fit(area, 10, 10);
        assert_eq!(layout.tile, 50.0);
        assert_eq!(layout.grid_rect().center(), area.center());
    }
}
