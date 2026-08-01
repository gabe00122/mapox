//! The egui demo app: a playable [`FindReturn`] episode, driven by the
//! keyboard and drawn from the tileset. On the web the same [`DemoApp`] is
//! hosted by `mapox-web`'s wasm-bindgen entry point instead of
//! [`open_window`].

pub mod env;
pub mod tileset;

use egui::{Color32, Rect, RichText, Stroke, StrokeKind, pos2, vec2};
use ndarray::{Array1, Array2, Array4};

use crate::{
    env::Environment,
    envs::find_return::{FindReturn, FindReturnConfig, FindReturnState},
    render::env::{GridRenderSettings, GridRenderState},
    symbols,
    timestep::{OBS_CHANNELS, TimeStepMut},
};
use tileset::Tileset;

/// Sheet coordinates per symbol, lifted from the table the pygame renderer
/// uses (`python/mapox/renderer.py::tilemap`). If the atlas math here drifts
/// from that table, this demo is where it shows up first.
const TILE_ART: &[(&str, u32, u32)] = &[
    (symbols::TILE_EMPTY, 17, 0),
    (symbols::TILE_WALL, 20, 3),
    (symbols::TILE_DESTRUCTIBLE_WALL, 20, 3),
    (symbols::TILE_FLAG, 29, 23),
    (symbols::AGENT_GENERIC, 104, 0),
];

/// Owns the arrays a [`TimeStepMut`] borrows, sized once from the env's specs.
struct TimeStepBuffers {
    obs: Array4<i8>,
    time: Array1<i32>,
    terminated: Array1<u8>,
    last_action: Array1<i32>,
    reward: Array1<f32>,
    action_mask: Array2<u8>,
    task_ids: Array1<i32>,
}

impl TimeStepBuffers {
    fn new(env: &impl Environment) -> Self {
        let num_agents = env.num_agents();
        let obs_spec = env.observation_spec();

        Self {
            obs: Array4::zeros((
                num_agents,
                obs_spec.width as usize,
                obs_spec.height as usize,
                OBS_CHANNELS,
            )),
            time: Array1::zeros(num_agents),
            terminated: Array1::zeros(num_agents),
            last_action: Array1::zeros(num_agents),
            reward: Array1::zeros(num_agents),
            action_mask: Array2::zeros((num_agents, env.action_spec().num_actions)),
            task_ids: Array1::zeros(num_agents),
        }
    }

    fn as_mut(&mut self) -> TimeStepMut<'_> {
        TimeStepMut {
            obs: self.obs.view_mut(),
            time: self.time.view_mut(),
            terminated: self.terminated.view_mut(),
            last_action: self.last_action.view_mut(),
            reward: self.reward.view_mut(),
            action_mask: self.action_mask.view_mut(),
            task_ids: self.task_ids.view_mut(),
        }
    }
}

pub struct DemoApp {
    /// Uploaded on the first frame, not in [`DemoApp::new`]: until the
    /// backend delivers input, the context reports a placeholder 2048 max
    /// texture side and `load_texture` debug-asserts the 2679px sheet against
    /// it. The real wgpu device allows 8192.
    tileset: Option<Tileset>,

    env: FindReturn,
    state: FindReturnState,
    buffers: TimeStepBuffers,

    settings: GridRenderSettings,
    render_state: GridRenderState,
    /// Sheet coordinates indexed by obs vocab id.
    art: Vec<(u32, u32)>,
    /// Action ids for up/right/down/left, in that order.
    move_actions: [i32; 4],

    seed: u64,
}

impl DemoApp {
    pub fn new() -> Self {
        let env = FindReturn::new(&FindReturnConfig::default());
        let mut state = env.init_state();
        let mut buffers = TimeStepBuffers::new(&env);

        let settings = env.get_render_settings();
        let art = settings
            .obs_vocab
            .symbols()
            .iter()
            .map(|symbol| {
                TILE_ART
                    .iter()
                    .find(|(name, _, _)| name == symbol)
                    .map(|&(_, col, row)| (col, row))
                    .expect("every obs symbol has tile art")
            })
            .collect();

        let action_id = |symbol| env.action_vocab().get(symbol).expect("move action") as i32;
        let move_actions = [
            action_id(symbols::MOVE_UP),
            action_id(symbols::MOVE_RIGHT),
            action_id(symbols::MOVE_DOWN),
            action_id(symbols::MOVE_LEFT),
        ];

        let seed = 0;
        env.reset(&mut state, seed, &mut buffers.as_mut());

        Self {
            tileset: None,
            env,
            state,
            buffers,
            settings,
            render_state: GridRenderState::default(),
            art,
            move_actions,
            seed,
        }
    }

    /// One key press (or OS key repeat, so holding a key walks) is one step.
    fn handle_input(&mut self, ui: &egui::Ui) {
        use egui::Key;

        let action = ui.input(|i| {
            let pressed = |a: Key, b: Key| i.key_pressed(a) || i.key_pressed(b);
            [
                (Key::ArrowUp, Key::W),
                (Key::ArrowRight, Key::D),
                (Key::ArrowDown, Key::S),
                (Key::ArrowLeft, Key::A),
            ]
            .into_iter()
            .position(|(a, b)| pressed(a, b))
        });
        if let Some(direction) = action {
            let actions = vec![self.move_actions[direction]; self.env.num_agents()];
            self.env
                .step(&mut self.state, &actions, &mut self.buffers.as_mut());
        }

        if ui.input(|i| i.key_pressed(Key::R)) {
            self.seed += 1;
            self.env
                .reset(&mut self.state, self.seed, &mut self.buffers.as_mut());
        }
    }

    /// The whole padded map, scaled to fit and centred. The env's y axis
    /// points up, screen y points down, so rows are drawn flipped.
    fn map_ui(&mut self, ui: &mut egui::Ui) {
        let width = self.settings.tile_width;
        let height = self.settings.tile_height;

        let area = ui.max_rect();
        let tile = (area.width() / width as f32).min(area.height() / height as f32);
        let origin = area.center() - vec2(width as f32, height as f32) * tile / 2.0;

        let screen_rect = |x: f32, y: f32, w: f32, h: f32| {
            Rect::from_min_size(
                pos2(origin.x + x * tile, origin.y + (height as f32 - y - h) * tile),
                vec2(w, h) * tile,
            )
        };

        let tileset = self.tileset.as_ref().expect("uploaded at the top of ui()");
        let painter = ui.painter();

        for x in 0..width {
            for y in 0..height {
                let id = self.render_state.tilemap[x * height + y] as usize;
                let (col, row) = self.art[id];
                tileset.draw(
                    painter,
                    col,
                    row,
                    screen_rect(x as f32, y as f32, 1.0, 1.0),
                    Color32::WHITE,
                );
            }
        }

        // outline each agent's egocentric view, so the partial observability
        // the env actually exposes is visible
        let view_width = self.settings.view_width as f32;
        let view_height = self.settings.view_height as f32;
        for position in &self.render_state.agent_positions {
            painter.rect_stroke(
                screen_rect(
                    position.x as f32 - view_width / 2.0 + 0.5,
                    position.y as f32 - view_height / 2.0 + 0.5,
                    view_width,
                    view_height,
                ),
                0.0,
                Stroke::new(1.0, Color32::YELLOW),
                StrokeKind::Outside,
            );
        }
    }
}

impl eframe::App for DemoApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.tileset.is_none() {
            self.tileset = Some(Tileset::embedded(ui.ctx()));
        }

        #[cfg(not(target_arch = "wasm32"))]
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }

        self.handle_input(ui);
        self.env
            .render_state_into(&self.state, &mut self.render_state);

        let hint = format!(
            "t={}   arrows/wasd: move   r: reset{}",
            self.state.time,
            if cfg!(target_arch = "wasm32") {
                ""
            } else {
                "   esc: quit"
            }
        );
        egui::Panel::bottom("hint").show(ui, |ui| ui.label(RichText::new(hint).weak()));
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(Color32::BLACK))
            .show(ui, |ui| self.map_ui(ui));
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

#[cfg(test)]
mod tests {
    use super::*;
    use tileset::{TILESET_COLS, TILESET_ROWS};

    /// Off-grid coordinates would sample a uv rect from past the edge of the
    /// texture, which the sampler happily clamps into something wrong-looking
    /// rather than reporting.
    #[test]
    fn every_art_tile_is_on_the_grid() {
        for (label, col, row) in TILE_ART {
            assert!(
                *col < TILESET_COLS && *row < TILESET_ROWS,
                "{label} off grid"
            );
        }
    }

    /// [`DemoApp::new`] panics on a symbol without art; catch that here, where
    /// the failure names the symbol, instead of at first launch.
    #[test]
    fn every_env_symbol_has_art() {
        let env = FindReturn::new(&FindReturnConfig::default());
        for symbol in env.obs_vocab().symbols() {
            assert!(
                TILE_ART.iter().any(|(name, _, _)| name == symbol),
                "no art for {symbol}"
            );
        }
    }
}
