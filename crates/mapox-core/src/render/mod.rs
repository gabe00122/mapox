//! Window setup and, for now, a demo that draws the tileset so the atlas math
//! can be checked by eye.

pub mod tileset;

use macroquad::prelude::*;

use tileset::{TILESET_COLS, TILESET_ROWS, Tileset};

/// Tiles worth eyeballing, lifted from the coordinates the pygame renderer
/// uses (`python/mapox/renderer.py::tilemap`). If the atlas math here drifts
/// from that table, this panel is where it shows up first.
const SHOWCASE: &[(&str, u32, u32)] = &[
    ("tile/empty", 17, 0),
    ("tile/wall", 20, 3),
    ("tile/grass", 6, 9),
    ("tile/food", 8, 18),
    ("tile/flag", 29, 23),
    ("tile/arrow", 80, 21),
    ("agent/generic", 104, 0),
    ("agent/scout", 3, 16),
    ("agent/harvester", 13, 14),
    ("knight/red", 35, 31),
    ("knight/blue", 35, 32),
    ("archer/red", 39, 31),
    ("archer/blue", 39, 32),
    ("decor_1", 15, 5),
    ("decor_2", 16, 5),
    ("decor_3", 17, 5),
];

pub fn window_conf() -> Conf {
    Conf {
        window_title: "mapox".to_owned(),
        window_width: 800,
        window_height: 600,
        high_dpi: true,
        ..Default::default()
    }
}

pub fn open_window() {
    macroquad::Window::from_config(window_conf(), run());
}

/// Height of a text overlay strip, and so the space the key hint reserves at
/// the bottom of the window.
const HINT_BAR_H: f32 = 28.0;

/// Which half of the tileset demo is on screen.
enum View {
    /// Named tiles at a readable size.
    Showcase,
    /// The raw sheet, pannable and zoomable.
    Atlas,
}

pub async fn run() {
    prevent_quit();

    let tileset = Tileset::embedded();
    let mut view = View::Showcase;
    let mut zoom = 2.0f32;
    let mut pan = Vec2::ZERO;

    while !is_quit_requested() && !is_key_pressed(KeyCode::Escape) {
        clear_background(BLACK);

        if is_key_pressed(KeyCode::Space) {
            view = match view {
                View::Showcase => View::Atlas,
                View::Atlas => View::Showcase,
            };
        }

        let hint = match view {
            View::Showcase => {
                draw_showcase(&tileset);
                "space: full atlas   esc: quit"
            }
            View::Atlas => {
                draw_atlas(&tileset, &mut zoom, &mut pan);
                "wasd/arrows: pan   wheel: zoom   space: showcase   esc: quit"
            }
        };
        overlay_text(hint, 12.0, screen_height() - HINT_BAR_H, 18.0);

        next_frame().await
    }
}

/// Named tiles in a centred grid, each with its sheet coordinates underneath.
fn draw_showcase(tileset: &Tileset) {
    /// Room under each tile for the name and the coordinate line.
    const LABEL_H: f32 = 36.0;
    /// Horizontal breathing room between columns.
    const GUTTER: f32 = 16.0;
    /// Past this the sprites are just blurry, so stop growing.
    const MAX_TILE: f32 = 72.0;

    let area_w = screen_width();
    let area_h = (screen_height() - HINT_BAR_H).max(1.0);

    // The window is whatever the host gives us — a python script's default, a
    // browser canvas, a resized frame — so pick the column count that makes
    // the tiles largest rather than assuming one.
    let (cols, tile) = (1..=SHOWCASE.len())
        .map(|cols| {
            let rows = SHOWCASE.len().div_ceil(cols);
            let tile = (area_w / cols as f32 - GUTTER)
                .min(area_h / rows as f32 - LABEL_H)
                .min(MAX_TILE);
            (cols, tile)
        })
        // Strictly-greater keeps the first winner, so once the tile size caps
        // out the fewest columns win and the last row stays as full as it can.
        .reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .unwrap();
    let tile = tile.max(4.0);

    let rows = SHOWCASE.len().div_ceil(cols);
    let cell_w = area_w / cols as f32;
    let cell_h = tile + LABEL_H;
    let origin_y = (area_h - rows as f32 * cell_h).max(0.0) / 2.0;

    for (i, (label, col, row)) in SHOWCASE.iter().enumerate() {
        let centre_x = ((i % cols) as f32 + 0.5) * cell_w;
        let cell_y = origin_y + (i / cols) as f32 * cell_h;

        tileset.draw(*col, *row, centre_x - tile / 2.0, cell_y, tile, WHITE);

        let text_w = cell_w - 6.0;
        centred_text(label, centre_x, cell_y + tile + 16.0, 15.0, text_w, WHITE);
        centred_text(
            &format!("{col},{row}"),
            centre_x,
            cell_y + tile + 31.0,
            14.0,
            text_w,
            DARKGRAY,
        );
    }
}

/// The whole sheet, so a missing or misaligned tile is visible at a glance.
fn draw_atlas(tileset: &Tileset, zoom: &mut f32, pan: &mut Vec2) {
    const PAN_SPEED: f32 = 600.0;

    let (_, wheel_y) = mouse_wheel();
    if wheel_y != 0.0 {
        *zoom = (*zoom * if wheel_y > 0.0 { 1.1 } else { 1.0 / 1.1 }).clamp(0.25, 12.0);
    }

    let dt = get_frame_time();
    let mut delta = Vec2::ZERO;
    if is_key_down(KeyCode::A) || is_key_down(KeyCode::Left) {
        delta.x += 1.0;
    }
    if is_key_down(KeyCode::D) || is_key_down(KeyCode::Right) {
        delta.x -= 1.0;
    }
    if is_key_down(KeyCode::W) || is_key_down(KeyCode::Up) {
        delta.y += 1.0;
    }
    if is_key_down(KeyCode::S) || is_key_down(KeyCode::Down) {
        delta.y -= 1.0;
    }
    *pan += delta * PAN_SPEED * dt;

    let texture = tileset.texture();
    let size = vec2(texture.width(), texture.height()) * *zoom;

    // Keep at least a corner of the sheet on screen no matter how far you pan.
    let limit = size + vec2(screen_width(), screen_height()) / 2.0;
    *pan = pan.clamp(-limit, limit);

    draw_texture_ex(
        texture,
        pan.x,
        pan.y,
        WHITE,
        DrawTextureParams {
            dest_size: Some(size),
            ..Default::default()
        },
    );

    overlay_text(
        &format!(
            "{TILESET_COLS}x{TILESET_ROWS} tiles @ {:.0}%",
            *zoom * 100.0
        ),
        12.0,
        0.0,
        18.0,
    );
}

/// Draws a line of text on a dimmed strip spanning the window, so it stays
/// legible on top of the sheet.
fn overlay_text(text: &str, x: f32, strip_top: f32, size: f32) {
    draw_rectangle(
        0.0,
        strip_top,
        screen_width(),
        HINT_BAR_H,
        Color::new(0.0, 0.0, 0.0, 0.75),
    );
    draw_text(text, x, strip_top + HINT_BAR_H - 8.0, size, LIGHTGRAY);
}

/// Centres `text` on `centre_x`, shrinking it until it fits `max_width` so
/// long symbol names in narrow cells stay separate words.
fn centred_text(text: &str, centre_x: f32, y: f32, size: f32, max_width: f32, color: Color) {
    let mut size = size;
    let mut width = measure_text(text, None, size as u16, 1.0).width;
    while width > max_width && size > 7.0 {
        size -= 1.0;
        width = measure_text(text, None, size as u16, 1.0).width;
    }
    draw_text(text, centre_x - width / 2.0, y, size, color);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off-grid coordinates would draw a source rect from past the edge of the
    /// texture, which macroquad happily clamps into something wrong-looking
    /// rather than reporting. The grid test in [`tileset`] pins the tile counts
    /// to the sheet, so being in range here is enough to be on the sheet.
    #[test]
    fn every_showcase_tile_is_on_the_grid() {
        for (label, col, row) in SHOWCASE {
            assert!(
                *col < TILESET_COLS && *row < TILESET_ROWS,
                "{label} off grid"
            );
        }
    }
}
