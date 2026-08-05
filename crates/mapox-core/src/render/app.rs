//! The interactive render app: eframe owns the loop, a [`Policy`] supplies
//! actions once per env step, and the keyboard can override the focused
//! agent. Training keeps the inverse control model and never touches this.

use egui::{Color32, RichText, Stroke, StrokeKind};
use ndarray::{Array1, Array2, Array4};

use crate::{
    env::Environment,
    envs::find_return::FindReturnConfig,
    make::{EnvConfig, make},
    policy::{Policy, PolicyInputs, RandomPolicy},
    render::{
        KEY_HINTS,
        env::{GridRenderSettings, GridRenderState},
        grid::{GridLayout, draw_tile_grid},
        resolve_art,
        tileset::Tileset,
    },
    symbols,
    timestep::{OBS_CHANNELS, TimeStepMut},
};

/// Owns the arrays a [`TimeStepMut`] borrows, sized once from the env's specs.
struct TimeStepBuffers {
    obs: Array4<u8>,
    time: Array1<i32>,
    terminated: Array1<bool>,
    last_action: Array1<i32>,
    reward: Array1<f32>,
    action_mask: Array2<bool>,
    task_ids: Array1<i32>,
}

impl TimeStepBuffers {
    fn new(env: &dyn Environment) -> Self {
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
            terminated: Array1::default(num_agents),
            last_action: Array1::zeros(num_agents),
            reward: Array1::zeros(num_agents),
            action_mask: Array2::default((num_agents, env.action_spec().num_actions)),
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    /// The map from the render state, padding cropped, every agent visible.
    BirdsEye,
    /// The focused agent's actual observations, not the render state.
    AgentPov,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PacingMode {
    /// The env steps at [`RenderApp::target_fps`]; a held movement key
    /// overrides the focused agent's action.
    FreeRun,
    /// The env only steps when a movement key press supplies the focused
    /// agent's action.
    StepOnInput,
}

/// What the per-frame input reading found; one struct so `ui()` reads input
/// exactly once.
struct FrameInput {
    tab: bool,
    pacing: bool,
    next_agent: bool,
    reset: bool,
    /// Direction index (up/right/down/left) that was newly pressed.
    dir_pressed: Option<usize>,
    /// Direction index currently held down.
    dir_down: Option<usize>,
}

pub struct RenderApp {
    /// Uploaded on the first frame, not in [`RenderApp::new`]: until the
    /// backend delivers input, the context reports a placeholder 2048 max
    /// texture side and `load_texture` debug-asserts the 2679px sheet against
    /// it. The real wgpu device allows 8192.
    tileset: Option<Tileset>,

    env: Box<dyn Environment + Send + Sync>,
    policy: Box<dyn Policy>,
    /// Set once the policy errors; the hint bar reports the random fallback.
    policy_failed: bool,
    buffers: TimeStepBuffers,
    actions: Vec<i32>,
    /// Whether `actions` already holds the policy's picks for the next step.
    /// The policy runs ahead of input, on the frame after a step, so a slow
    /// policy stalls a visually idle frame instead of adding its inference
    /// time to the keypress-to-screen latency.
    actions_ready: bool,

    settings: GridRenderSettings,
    /// Derived as `view / 2`, the invariant the obs-encoding scheme already
    /// relies on; the birds-eye view crops this border off the padded map.
    pad_w: usize,
    pad_h: usize,
    render_state: GridRenderState,
    /// Sheet coordinates indexed by obs vocab id.
    art: Vec<(u32, u32)>,
    /// Action ids for up/right/down/left; `None` when the env lacks the move.
    move_actions: [Option<i32>; 4],

    view_mode: ViewMode,
    pacing: PacingMode,
    focused_agent: usize,
    target_fps: f32,
    /// Deadline for the next free-run step on egui's `input.time` clock,
    /// `None` until free-run schedules one. An absolute deadline instead of
    /// a dt accumulator because `stable_dt` only reports real elapsed time
    /// after an *immediate* repaint request; under `request_repaint_after`
    /// it is a constant `predicted_dt`, which ran the env slow when idle and
    /// fast whenever mouse motion raised the frame rate.
    next_step_time: Option<f64>,
    /// A key pressed between free-run steps, latched until the next step.
    pending_override: Option<usize>,

    seed: u64,
}

impl RenderApp {
    pub fn new(mut env: Box<dyn Environment + Send + Sync>, policy: Box<dyn Policy>) -> Self {
        let mut buffers = TimeStepBuffers::new(env.as_ref());

        let settings = env.get_render_settings();
        let art = resolve_art(&settings.obs_vocab);

        let action_vocab = env.action_vocab();
        let action_id = |symbol| action_vocab.get(symbol).map(|id| id as i32);
        let move_actions = [
            action_id(symbols::MOVE_UP),
            action_id(symbols::MOVE_RIGHT),
            action_id(symbols::MOVE_DOWN),
            action_id(symbols::MOVE_LEFT),
        ];

        let seed = 0;
        env.reset(seed, &mut buffers.as_mut());

        let num_agents = env.num_agents();
        Self {
            tileset: None,
            actions: vec![0; num_agents],
            actions_ready: false,
            env,
            policy,
            policy_failed: false,
            buffers,
            pad_w: settings.view_width / 2,
            pad_h: settings.view_height / 2,
            settings,
            render_state: GridRenderState::default(),
            art,
            move_actions,
            view_mode: ViewMode::BirdsEye,
            // the native demo keeps the play-by-keypress feel; the web demo
            // free-runs so the page doesn't look frozen
            pacing: if cfg!(target_arch = "wasm32") {
                PacingMode::FreeRun
            } else {
                PacingMode::StepOnInput
            },
            focused_agent: 0,
            target_fps: 10.0,
            next_step_time: None,
            pending_override: None,
            seed,
        }
    }

    /// The default [`FindReturn`](crate::envs::find_return::FindReturn)
    /// episode under a random policy, used by the native example and the web
    /// build.
    pub fn demo() -> Self {
        Self::new(
            make(&EnvConfig::FindReturn(FindReturnConfig::default())),
            Box::new(RandomPolicy::new(0)),
        )
    }

    fn read_input(ui: &egui::Ui) -> FrameInput {
        use egui::Key;

        const DIRECTION_KEYS: [(Key, Key); 4] = [
            (Key::ArrowUp, Key::W),
            (Key::ArrowRight, Key::D),
            (Key::ArrowDown, Key::S),
            (Key::ArrowLeft, Key::A),
        ];

        ui.input(|i| FrameInput {
            tab: i.key_pressed(Key::Tab),
            pacing: i.key_pressed(Key::P),
            next_agent: i.key_pressed(Key::N),
            reset: i.key_pressed(Key::R),
            dir_pressed: DIRECTION_KEYS
                .into_iter()
                .position(|(a, b)| i.key_pressed(a) || i.key_pressed(b)),
            dir_down: DIRECTION_KEYS
                .into_iter()
                .position(|(a, b)| i.key_down(a) || i.key_down(b)),
        })
    }

    fn reset(&mut self) {
        self.seed += 1;
        self.env.reset(self.seed, &mut self.buffers.as_mut());
        self.next_step_time = None;
        self.pending_override = None;
        // any precomputed actions were for the old episode's observations
        self.actions_ready = false;
    }

    /// Runs the policy over the current observations into `actions`. Split
    /// from [`Self::step_env`] so `ui()` can run it on the frame *after* a
    /// step: immediate mode presents a frame only when `ui()` returns, so a
    /// slow policy called on the keypress frame would hold back the very
    /// frame that shows the step's result.
    fn compute_actions(&mut self) {
        // field accesses stay inline so the borrows of `buffers`, `policy`
        // and `actions` split; a `&self` helper for the inputs would not
        let inputs = PolicyInputs {
            obs: self.buffers.obs.view(),
            reward: self.buffers.reward.view(),
            terminated: self.buffers.terminated.view(),
            action_mask: self.buffers.action_mask.view(),
        };
        if let Err(err) = self.policy.act(&inputs, &mut self.actions) {
            eprintln!("policy failed, falling back to random: {err}");
            self.policy = Box::new(RandomPolicy::new(self.seed));
            self.policy_failed = true;
            self.policy
                .act(&inputs, &mut self.actions)
                .expect("random policy is infallible");
        }
        self.actions_ready = true;
    }

    /// One env step: the precomputed policy actions, with the keyboard
    /// override replacing the focused agent's. The inline fallback only runs
    /// when free-run catch-up takes several steps in a single frame.
    fn step_env(&mut self, override_dir: Option<usize>) {
        if !self.actions_ready {
            self.compute_actions();
        }

        if let Some(action) = override_dir.and_then(|dir| self.move_actions[dir]) {
            self.actions[self.focused_agent] = action;
        }

        self.env.step(&self.actions, &mut self.buffers.as_mut());
        self.actions_ready = false;

        // keep the window alive across episode ends without user action
        // (FindReturn never terminates; this is for envs that do)
        if !self.buffers.terminated.is_empty() && self.buffers.terminated.iter().all(|&t| t) {
            self.reset();
        }
    }

    fn free_run(&mut self, ui: &egui::Ui, input: &FrameInput) {
        let now = ui.input(|i| i.time);
        let interval = f64::from(1.0 / self.target_fps);
        // first free-run frame steps immediately
        let mut next = self.next_step_time.unwrap_or(now);

        // cap the catch-up after a stall (window drag, slow policy)
        let mut steps = 0;
        while now >= next && steps < 4 {
            let override_dir = input.dir_down.or(self.pending_override.take());
            self.step_env(override_dir);
            next += interval;
            steps += 1;
        }
        // whatever debt remains after the cap is forgiven, not replayed
        if now >= next {
            next = now + interval;
        }
        self.next_step_time = Some(next);

        let wait = std::time::Duration::from_secs_f64((next - now).max(0.001));
        ui.ctx().request_repaint_after(wait);
    }

    fn hint_text(&self) -> String {
        let pacing = match self.pacing {
            PacingMode::FreeRun => format!("free-run {}fps", self.target_fps),
            PacingMode::StepOnInput => "step-on-input".to_owned(),
        };
        let mut hint = format!(
            "t={}   agent {}/{}   {}",
            self.buffers.time.first().copied().unwrap_or(0),
            self.focused_agent,
            self.env.num_agents(),
            pacing,
        );
        for (keys, action) in KEY_HINTS {
            if *keys == "esc" && cfg!(target_arch = "wasm32") {
                continue;
            }
            hint.push_str(&format!("   {keys}: {action}"));
        }
        if self.policy_failed {
            hint.push_str("   [policy error (see console) - random fallback]");
        }
        hint
    }

    /// The map with the wall padding cropped off; click an agent to focus it.
    fn birds_eye_ui(&mut self, ui: &mut egui::Ui) {
        let crop_cols = self.settings.tile_width - 2 * self.pad_w;
        let crop_rows = self.settings.tile_height - 2 * self.pad_h;
        let layout = GridLayout::fit(ui.max_rect(), crop_cols, crop_rows);

        let response = ui.allocate_rect(layout.grid_rect(), egui::Sense::click());
        if response.clicked() {
            if let Some((x, y)) = response
                .interact_pointer_pos()
                .and_then(|pos| layout.pos_to_cell(pos))
            {
                // cropped grid coords back to the padded frame positions use
                let clicked = (x + self.pad_w, y + self.pad_h);
                // agents can share a tile; the lowest index wins
                if let Some(agent) = self
                    .render_state
                    .agent_positions
                    .iter()
                    .position(|p| (p.x as usize, p.y as usize) == clicked)
                {
                    self.focused_agent = agent;
                }
            }
        }

        // clip so the focused view rect can't overhang into the letterbox
        let painter = ui.painter_at(layout.grid_rect());
        let tileset = self.tileset.as_ref().expect("uploaded at the top of ui()");
        let (pad_w, pad_h) = (self.pad_w, self.pad_h);
        let tilemap = &self.render_state.tilemap;
        draw_tile_grid(&painter, tileset, &self.art, &layout, |x, y| {
            tilemap[[x + pad_w, y + pad_h]]
        });

        // outline the focused agent and its egocentric view, so the partial
        // observability the env actually exposes is visible
        if let Some(position) = self.render_state.agent_positions.get(self.focused_agent) {
            let x = position.x as f32 - pad_w as f32;
            let y = position.y as f32 - pad_h as f32;
            let view_width = self.settings.view_width as f32;
            let view_height = self.settings.view_height as f32;
            painter.rect_stroke(
                layout.cell_rect(
                    x - view_width / 2.0 + 0.5,
                    y - view_height / 2.0 + 0.5,
                    view_width,
                    view_height,
                ),
                0.0,
                Stroke::new(1.0, Color32::YELLOW),
                StrokeKind::Outside,
            );
            painter.rect_stroke(
                layout.cell_rect(x, y, 1.0, 1.0),
                0.0,
                Stroke::new(2.0, Color32::YELLOW),
                StrokeKind::Inside,
            );
        }
    }

    /// The focused agent's observations as the env encoded them, which is
    /// what a policy sees; the agent itself sits in the centre cell.
    fn pov_ui(&mut self, ui: &mut egui::Ui) {
        let layout = GridLayout::fit(
            ui.max_rect(),
            self.settings.view_width,
            self.settings.view_height,
        );
        let tileset = self.tileset.as_ref().expect("uploaded at the top of ui()");
        let obs = &self.buffers.obs;
        let focused = self.focused_agent;
        draw_tile_grid(ui.painter(), tileset, &self.art, &layout, |x, y| {
            obs[[focused, x, y, 0]]
        });
    }
}

impl eframe::App for RenderApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.tileset.is_none() {
            self.tileset = Some(Tileset::embedded(ui.ctx()));
        }

        // run the policy ahead of input, so a keypress finds its step's
        // actions already in hand instead of waiting on inference
        if !self.actions_ready {
            self.compute_actions();
        }

        #[cfg(not(target_arch = "wasm32"))]
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }

        let input = Self::read_input(ui);

        if input.tab {
            self.view_mode = match self.view_mode {
                ViewMode::BirdsEye => ViewMode::AgentPov,
                ViewMode::AgentPov => ViewMode::BirdsEye,
            };
        }
        if input.pacing {
            self.pacing = match self.pacing {
                PacingMode::FreeRun => PacingMode::StepOnInput,
                PacingMode::StepOnInput => PacingMode::FreeRun,
            };
            self.next_step_time = None;
        }
        if input.next_agent && self.env.num_agents() > 0 {
            self.focused_agent = (self.focused_agent + 1) % self.env.num_agents();
        }
        if input.reset {
            self.reset();
        }

        match self.pacing {
            PacingMode::StepOnInput => {
                if let Some(dir) = input.dir_pressed {
                    self.step_env(Some(dir));
                    self.pending_override = None;
                }
            }
            PacingMode::FreeRun => {
                if input.dir_pressed.is_some() {
                    self.pending_override = input.dir_pressed;
                }
                self.free_run(ui, &input);
            }
        }

        self.env.render_state_into(&mut self.render_state);

        egui::Panel::bottom("hint").show(ui, |ui| ui.label(RichText::new(self.hint_text()).weak()));
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(Color32::BLACK))
            .show(ui, |ui| match self.view_mode {
                ViewMode::BirdsEye => self.birds_eye_ui(ui),
                ViewMode::AgentPov => self.pov_ui(ui),
            });

        // a step consumed the precomputed actions; come straight back on an
        // idle frame to run the policy for the next one
        if !self.actions_ready {
            ui.ctx().request_repaint();
        }
    }
}

/// Opens the native window and blocks until it closes. The web build starts
/// [`RenderApp`] through `mapox-web`'s wasm-bindgen entry point instead.
#[cfg(not(target_arch = "wasm32"))]
pub fn open_window(app: RenderApp) -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("mapox")
            .with_inner_size([800.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native("mapox", options, Box::new(move |_cc| Ok(Box::new(app))))
}
