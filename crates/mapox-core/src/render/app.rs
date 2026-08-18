//! The interactive render app: eframe owns the loop, a [`Policy`] supplies
//! actions once per env step, and in step-on-input pacing the keyboard
//! overrides the focused agent. Training keeps the inverse control model and
//! never touches this.

use crate::{
    env::Environment,
    policy::Policy,
    render::{
        env::{GridRenderSettings, GridRenderState},
        grid::{GridLayout, draw_tile_grid},
        keys::{self, Command, Input},
        resolve_art,
        tileset::Tileset,
    },
    timestep::TimeStepBuffers,
    vocab::VocabId,
};
use egui::{Color32, RichText, Stroke, StrokeKind};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    /// The map from the render state, padding cropped, every agent visible.
    BirdsEye,
    /// The focused agent's actual observations, not the render state.
    AgentPov,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PacingMode {
    /// The env steps at [`RenderApp::target_fps`] under the policy alone;
    /// movement keys are ignored.
    FreeRun,
    /// The env only steps when a movement key press supplies the focused
    /// agent's action.
    StepOnInput,
}

pub struct RenderApp {
    /// Uploaded on the first frame, not in [`RenderApp::new`]: until the
    /// backend delivers input, the context reports a placeholder 2048 max
    /// texture side and `load_texture` debug-asserts the 2679px sheet against
    /// it. The real wgpu device allows 8192.
    tileset: Option<Tileset>,

    env: Box<dyn Environment>,
    policy: Box<dyn Policy>,
    /// Total step count before a reset
    length: usize,
    step_count: usize,
    /// Set once the policy errors; the hint bar reports the random fallback.
    buffers: TimeStepBuffers,
    actions: Vec<VocabId>,
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

    seed: u64,
}

impl RenderApp {
    pub fn new(
        mut env: Box<dyn Environment>,
        length: usize,
        seed: u64,
        mut policy: Box<dyn Policy>,
    ) -> Self {
        let mut buffers = TimeStepBuffers::new(env.as_ref());

        let settings = env.get_render_settings();
        let art = resolve_art(&settings.obs_vocab);

        let mut rng = SmallRng::seed_from_u64(seed);
        policy
            .reset(env.num_agents(), rng.random())
            .expect("Policy reset failed");
        env.reset(rng.random(), &mut buffers.view_mut());

        let num_agents = env.num_agents();
        Self {
            tileset: None,
            actions: vec![0; num_agents],
            actions_ready: false,
            env,
            policy,
            length,
            step_count: 0,
            buffers,
            pad_w: settings.view_width / 2,
            pad_h: settings.view_height / 2,
            settings,
            render_state: GridRenderState::default(),
            art,
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
            seed,
        }
    }

    fn reset(&mut self) {
        self.seed += 1;
        let mut rng = SmallRng::seed_from_u64(self.seed);

        self.step_count = 0;
        self.env.reset(rng.random(), &mut self.buffers.view_mut());
        self.policy
            .reset(self.env.num_agents(), rng.random())
            .expect("policy failed");
        self.next_step_time = None;
        // any precomputed actions were for the old episode's observations
        self.actions_ready = false;
    }

    fn compute_actions(&mut self) {
        if !self.episode_done() {
            let timestep = self.buffers.view();
            self.policy
                .act(&timestep, &mut self.actions)
                .expect("policy failed");
        }
        self.actions_ready = true;
    }

    fn episode_done(&self) -> bool {
        self.step_count >= self.length
    }

    /// `override_action` replaces the focused agent's policy action, which is
    /// how the keyboard plays a single agent while the policy drives the rest.
    fn step_env(&mut self, override_action: Option<VocabId>) {
        if self.episode_done() {
            self.reset();
            return;
        }

        if !self.actions_ready {
            self.compute_actions();
        }

        if let Some(action) = override_action {
            self.actions[self.focused_agent] = action;
        }

        self.env.step(&self.actions, &mut self.buffers.view_mut());
        self.step_count += 1;
        self.actions_ready = false;
    }

    fn run_command(&mut self, command: Command, ui: &egui::Ui) {
        match command {
            Command::ToggleView => {
                self.view_mode = match self.view_mode {
                    ViewMode::BirdsEye => ViewMode::AgentPov,
                    ViewMode::AgentPov => ViewMode::BirdsEye,
                }
            }
            Command::TogglePacing => {
                self.pacing = match self.pacing {
                    PacingMode::FreeRun => PacingMode::StepOnInput,
                    PacingMode::StepOnInput => PacingMode::FreeRun,
                };
                self.next_step_time = None;
            }
            Command::NextAgent => {
                if self.env.num_agents() > 0 {
                    self.focused_agent = (self.focused_agent + 1) % self.env.num_agents();
                }
            }
            Command::Reset => self.reset(),
            Command::Quit => ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close),
        }
    }

    fn free_run(&mut self, ui: &egui::Ui) {
        let now = ui.input(|i| i.time);
        let interval = f64::from(1.0 / self.target_fps);
        // first free-run frame steps immediately
        let mut next = self.next_step_time.unwrap_or(now);

        // cap the catch-up after a stall (window drag, slow policy)
        let mut steps = 0;
        while now >= next && steps < 4 {
            self.step_env(None);
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
        let hint = format!(
            "t={}   agent {}/{}   {}",
            self.step_count,
            self.focused_agent,
            self.env.num_agents(),
            pacing,
        );
        hint
    }

    /// The map with the wall padding cropped off; click an agent to focus it.
    fn birds_eye_ui(&mut self, ui: &mut egui::Ui) {
        let crop_cols = self.settings.tile_width - 2 * self.pad_w;
        let crop_rows = self.settings.tile_height - 2 * self.pad_h;
        let layout = GridLayout::fit(ui.max_rect(), crop_cols, crop_rows);

        let response = ui.allocate_rect(layout.grid_rect(), egui::Sense::click());
        if response.clicked()
            && let Some((x, y)) = response
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

        match ui.input(|state| keys::read(state, self.env.action_vocab())) {
            Some(Input::Command(command)) => self.run_command(command, ui),
            // free-run ignores the action keys: the policy drives every agent
            Some(Input::Action(action)) if self.pacing == PacingMode::StepOnInput => {
                self.step_env(Some(action));
            }
            _ => {}
        }

        if self.pacing == PacingMode::FreeRun {
            self.free_run(ui);
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

#[cfg(not(target_arch = "wasm32"))]
pub fn open_window(app: RenderApp) -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("mapox")
            .with_inner_size([800.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native("mapox", options, Box::new(move |_cc| Ok(Box::new(app))))
}
