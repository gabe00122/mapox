//! The egui demo app that draws the tileset so the atlas math can be checked
//! by eye, plus the native window that hosts it. On the web the same
//! [`DemoApp`] is hosted by `mapox-web`'s wasm-bindgen entry point instead.

pub mod env;
pub mod tileset;

use egui::{Align2, Color32, FontId, Rect, RichText, Vec2, pos2, vec2};

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

/// Which half of the tileset demo is on screen.
enum View {
    /// Named tiles at a readable size.
    Showcase,
    /// The raw sheet, pannable and zoomable.
    Atlas,
}

pub struct DemoApp {
    /// Uploaded on the first frame, not in [`DemoApp::new`]: until the
    /// backend delivers input, the context reports a placeholder 2048 max
    /// texture side and `load_texture` debug-asserts the 2679px sheet against
    /// it. The real wgpu device allows 8192.
    tileset: Option<Tileset>,
    view: View,
    /// Scene-space region of the atlas in view. Starts as
    /// [`Rect::NOTHING`] so the first atlas frame auto-fits the whole sheet;
    /// after that it persists across view toggles like the old pan/zoom did.
    scene_rect: Rect,
}

impl DemoApp {
    pub fn new() -> Self {
        Self {
            tileset: None,
            view: View::Showcase,
            scene_rect: Rect::NOTHING,
        }
    }

    /// The sheet texture, valid any time after the top of [`eframe::App::ui`].
    fn tileset(&self) -> &Tileset {
        self.tileset.as_ref().expect("uploaded at the top of ui()")
    }

    /// Named tiles in a centred grid, each with its sheet coordinates underneath.
    fn showcase_ui(&self, ui: &mut egui::Ui) {
        /// Room under each tile for the name and the coordinate line.
        const LABEL_H: f32 = 36.0;
        /// Horizontal breathing room between columns.
        const GUTTER: f32 = 16.0;
        /// Past this the sprites are just blurry, so stop growing.
        const MAX_TILE: f32 = 72.0;

        let area = ui.max_rect();

        // The window is whatever the host gives us — a python script's default,
        // a browser canvas, a resized frame — so pick the column count that
        // makes the tiles largest rather than assuming one.
        let (cols, tile) = (1..=SHOWCASE.len())
            .map(|cols| {
                let rows = SHOWCASE.len().div_ceil(cols);
                let tile = (area.width() / cols as f32 - GUTTER)
                    .min(area.height() / rows as f32 - LABEL_H)
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
        let cell_w = area.width() / cols as f32;
        let cell_h = tile + LABEL_H;
        let origin_y = area.top() + (area.height() - rows as f32 * cell_h).max(0.0) / 2.0;
        let painter = ui.painter();

        for (i, (label, col, row)) in SHOWCASE.iter().enumerate() {
            let centre_x = area.left() + ((i % cols) as f32 + 0.5) * cell_w;
            let cell_y = origin_y + (i / cols) as f32 * cell_h;

            let rect = Rect::from_min_size(pos2(centre_x - tile / 2.0, cell_y), Vec2::splat(tile));
            self.tileset()
                .draw(painter, *col, *row, rect, Color32::WHITE);

            let text_w = cell_w - 6.0;
            centred_text(
                painter,
                label,
                centre_x,
                cell_y + tile + 4.0,
                15.0,
                text_w,
                Color32::WHITE,
            );
            centred_text(
                painter,
                &format!("{col},{row}"),
                centre_x,
                cell_y + tile + 20.0,
                14.0,
                text_w,
                Color32::DARK_GRAY,
            );
        }
    }

    /// The whole sheet, so a missing or misaligned tile is visible at a glance.
    fn atlas_ui(&mut self, ui: &mut egui::Ui) {
        const PAN_SPEED: f32 = 600.0;

        let viewport = ui.max_rect();

        let (dt, delta) = ui.input(|i| {
            use egui::Key::*;
            let axis = |neg1, neg2, pos1, pos2| {
                (i.key_down(pos1) || i.key_down(pos2)) as i8 as f32
                    - (i.key_down(neg1) || i.key_down(neg2)) as i8 as f32
            };
            (
                i.stable_dt,
                vec2(
                    axis(A, ArrowLeft, D, ArrowRight),
                    axis(W, ArrowUp, S, ArrowDown),
                ),
            )
        });
        // Keys move the camera in screen pixels per second; the scene rect is
        // in sheet texels, so divide by the zoom to keep the speed constant.
        if self.scene_rect.is_positive() {
            let zoom = viewport.width() / self.scene_rect.width();
            self.scene_rect = self.scene_rect.translate(delta * PAN_SPEED * dt / zoom);
        }

        let tileset = self.tileset.as_ref().expect("uploaded at the top of ui()");
        egui::Scene::new()
            .zoom_range(0.25..=12.0)
            .show(ui, &mut self.scene_rect, |ui| {
                ui.add(egui::Image::from_texture(tileset.texture()));
            });

        // Zoom readout on a dimmed strip, painted after the scene so it stays
        // on top of the sheet.
        if self.scene_rect.is_positive() {
            let zoom = viewport.width() / self.scene_rect.width();
            let strip = Rect::from_min_size(viewport.min, vec2(viewport.width(), 24.0));
            let painter = ui.painter();
            painter.rect_filled(
                strip,
                egui::CornerRadius::ZERO,
                Color32::from_black_alpha(191),
            );
            painter.text(
                strip.left_center() + vec2(12.0, 0.0),
                Align2::LEFT_CENTER,
                format!("{TILESET_COLS}x{TILESET_ROWS} tiles @ {:.0}%", zoom * 100.0),
                FontId::proportional(14.0),
                Color32::LIGHT_GRAY,
            );
        }
    }
}

impl eframe::App for DemoApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.tileset.is_none() {
            self.tileset = Some(Tileset::embedded(ui.ctx()));
        }

        if ui.input(|i| i.key_pressed(egui::Key::Space)) {
            self.view = match self.view {
                View::Showcase => View::Atlas,
                View::Atlas => View::Showcase,
            };
        }
        #[cfg(not(target_arch = "wasm32"))]
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }

        let hint = match self.view {
            View::Showcase => "space: full atlas",
            View::Atlas => "drag/wasd: pan   ctrl+wheel: zoom   space: showcase",
        };
        let hint = if cfg!(target_arch = "wasm32") {
            hint.to_owned()
        } else {
            format!("{hint}   esc: quit")
        };

        egui::Panel::bottom("hint").show(ui, |ui| ui.label(RichText::new(hint).weak()));
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(Color32::BLACK))
            .show(ui, |ui| match self.view {
                View::Showcase => self.showcase_ui(ui),
                View::Atlas => self.atlas_ui(ui),
            });
    }
}

/// Opens the native demo window and blocks until it closes. The web build
/// instead starts [`DemoApp`] through `mapox-web`'s wasm-bindgen entry point.
#[cfg(not(target_arch = "wasm32"))]
pub fn open_window() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("mapox")
            .with_inner_size([800.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "mapox",
        options,
        Box::new(|_cc| Ok(Box::new(DemoApp::new()))),
    )
}

/// Centres `text` on `centre_x`, shrinking it until it fits `max_width` so
/// long symbol names in narrow cells stay separate words.
fn centred_text(
    painter: &egui::Painter,
    text: &str,
    centre_x: f32,
    top: f32,
    size: f32,
    max_width: f32,
    color: Color32,
) {
    let mut size = size;
    let mut galley = painter.layout_no_wrap(text.to_owned(), FontId::proportional(size), color);
    while galley.size().x > max_width && size > 7.0 {
        size -= 1.0;
        galley = painter.layout_no_wrap(text.to_owned(), FontId::proportional(size), color);
    }
    painter.galley(pos2(centre_x - galley.size().x / 2.0, top), galley, color);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off-grid coordinates would sample a uv rect from past the edge of the
    /// texture, which the sampler happily clamps into something wrong-looking
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
