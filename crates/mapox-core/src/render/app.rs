//! The interactive render app: eframe owns the loop, a [`Policy`] supplies
//! actions once per env step, and in step-on-input pacing the keyboard
//! overrides the focused agent. Training keeps the inverse control model and
//! never touches this.

use crate::{
    env::Environment,
    policy::Policy,
    render::{
        env::{GridRenderSettings, GridRenderState, visible_tiles},
        gpu::TilemapRenderer,
        grid::GridLayout,
        keys::{self, Command, Input},
        resolve_art,
    },
    symbols,
    timestep::TimeStepBuffers,
    vocab::VocabId,
};
use egui::{Color32, RichText, Stroke, StrokeKind};
use ndarray::s;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

/// How many steps the clock may replay in one frame to catch up after a
/// stall; the rest of the debt is forgiven.
const MAX_CATCH_UP_STEPS: usize = 4;

/// Width of the episode / agent / mode panel, in points.
const SIDE_PANEL_WIDTH: f32 = 220.0;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    /// The map from the render state, every agent visible.
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
    /// Created on the first frame, when the backend's wgpu device is available.
    tilemap_renderer: Option<TilemapRenderer>,

    env: Box<dyn Environment>,
    policy: Box<dyn Policy>,
    /// Total step count before a reset
    length: usize,
    step_count: usize,
    /// Rewritten by every env reset and step; the policy reads it through
    /// [`RenderApp::compute_actions`], and the side panel reads reward and
    /// last action out of it.
    buffers: TimeStepBuffers,
    actions: Vec<VocabId>,
    /// Whether `actions` already holds the policy's picks for the next step.
    /// The policy runs ahead of input, on the frame after a step, so a slow
    /// policy stalls a visually idle frame instead of adding its inference
    /// time to the keypress-to-screen latency.
    actions_ready: bool,

    settings: GridRenderSettings,
    render_state: GridRenderState,
    /// The env changed since the last render-state snapshot.
    render_state_dirty: bool,
    /// The displayed map changed since the last GPU upload.
    tilemap_dirty: bool,
    /// Sheet coordinates indexed by obs vocab id.
    art: Vec<(u32, u32)>,

    /// Per-agent reward summed since the last reset, so the side panel can show
    /// where the focused agent's episode stands and not just the last step.
    returns: Vec<f32>,

    view_mode: ViewMode,
    /// Whether the bird's-eye view fogs what no agent can see.
    show_fov: bool,
    pacing: PacingMode,
    focused_agent: usize,
    /// Task names in task-id order, empty for a single-task env.
    task_names: Vec<String>,
    /// The task being played, `Some` exactly when `task_names` is non-empty.
    task: Option<usize>,
    /// Whether the controls reference is open over the map. The next key
    /// press or click closes it and does nothing else.
    show_controls: bool,
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
        // A multitask env plays one task at a time: the whole batch would
        // step every task's agents while only the first env is drawn. A task
        // the caller already picked is kept.
        let task_names = env.task_names();
        let mut task = env.enjoy_task();
        if task.is_none() && !task_names.is_empty() {
            task = Some(0);
            env.set_enjoy_mode(task);
        }

        let mut buffers = TimeStepBuffers::new(env.as_ref());

        let settings = env.get_render_settings();
        let art = resolve_art(&settings.obs_vocab);

        // the same draw order as `reset`, so the seed the side panel shows
        // replays its episode when passed back in
        let mut rng = SmallRng::seed_from_u64(seed);
        env.reset(rng.random(), &mut buffers.view_mut());
        policy
            .reset(env.num_agents(), rng.random())
            .expect("Policy reset failed");

        let num_agents = env.num_agents();
        Self {
            tilemap_renderer: None,
            actions: vec![0; num_agents],
            actions_ready: false,
            env,
            policy,
            length,
            step_count: 0,
            buffers,
            settings,
            render_state: GridRenderState::default(),
            render_state_dirty: true,
            tilemap_dirty: true,
            art,
            returns: vec![0.0; num_agents],
            view_mode: ViewMode::BirdsEye,
            show_fov: true,
            // the native demo keeps the play-by-keypress feel; the web demo
            // free-runs so the page doesn't look frozen
            pacing: if cfg!(target_arch = "wasm32") {
                PacingMode::FreeRun
            } else {
                PacingMode::StepOnInput
            },
            focused_agent: 0,
            task_names,
            task,
            show_controls: false,
            target_fps: 10.0,
            next_step_time: None,
            seed,
        }
    }

    fn reset(&mut self) {
        self.reseed(self.seed.wrapping_add(1));
    }

    /// Starts a fresh episode from `seed`; the ones after it count up from
    /// there.
    fn reseed(&mut self, seed: u64) {
        self.seed = seed;
        let mut rng = SmallRng::seed_from_u64(seed);

        self.step_count = 0;
        self.env.reset(rng.random(), &mut self.buffers.view_mut());
        self.policy
            .reset(self.env.num_agents(), rng.random())
            .expect("policy failed");
        self.next_step_time = None;
        self.returns.fill(0.0);
        // any precomputed actions were for the old episode's observations
        self.actions_ready = false;
        self.render_state_dirty = true;
        self.tilemap_dirty = true;
    }

    /// Switches a multitask env to `task` and starts a fresh episode of it.
    /// The task can have a different agent count, map size and view, so
    /// everything sized from the env is rebuilt.
    fn select_task(&mut self, task: usize) {
        if self.task == Some(task) || task >= self.task_names.len() {
            return;
        }
        self.env.set_enjoy_mode(Some(task));
        self.task = Some(task);

        self.buffers = TimeStepBuffers::new(self.env.as_ref());
        self.settings = self.env.get_render_settings();
        self.art = resolve_art(&self.settings.obs_vocab);
        let num_agents = self.env.num_agents();
        self.actions = vec![0; num_agents];
        self.returns = vec![0.0; num_agents];
        self.focused_agent = 0;
        // also resets the policy, which resizes itself to the new agent count
        self.reset();
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
    /// An exhausted episode spends the call on the reset instead: the
    /// override is dropped, so a step-on-input keypress at the end restarts
    /// the episode rather than stepping.
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
        for (total, reward) in self.returns.iter_mut().zip(self.buffers.reward.iter()) {
            *total += reward;
        }
        self.step_count += 1;
        self.actions_ready = false;
        self.render_state_dirty = true;
        self.tilemap_dirty = true;
    }

    fn run_command(&mut self, command: Command, ui: &egui::Ui) {
        match command {
            Command::ToggleView => {
                self.view_mode = match self.view_mode {
                    ViewMode::BirdsEye => ViewMode::AgentPov,
                    ViewMode::AgentPov => ViewMode::BirdsEye,
                };
                self.tilemap_dirty = true;
            }
            Command::ToggleFov => {
                self.show_fov = !self.show_fov;
                if self.view_mode == ViewMode::BirdsEye {
                    self.tilemap_dirty = true;
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
                if self.env.num_agents() > 1 {
                    self.focused_agent = (self.focused_agent + 1) % self.env.num_agents();
                    if self.view_mode == ViewMode::AgentPov {
                        self.tilemap_dirty = true;
                    }
                }
            }
            Command::NextTask => {
                if let Some(task) = self.task {
                    self.select_task((task + 1) % self.task_names.len());
                }
            }
            Command::Reset => self.reset(),
            Command::ShowControls => self.show_controls = true,
            Command::Quit => ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close),
        }
    }

    /// The focused agent's only legal action, when its mask leaves it exactly
    /// one. There is nothing for manual control to ask about in that case.
    fn forced_action(&self) -> Option<VocabId> {
        let mut legal = self
            .buffers
            .action_mask
            .row(self.focused_agent)
            .into_iter()
            .enumerate()
            .filter(|&(_, &legal)| legal)
            .map(|(id, _)| VocabId::try_from(id).expect("action mask fits VocabId"));

        let only = legal.next()?;
        legal.next().is_none().then_some(only)
    }

    /// Whether the clock drives the env this frame instead of waiting on a
    /// keypress. Free-run always does. Step-on-input self-plays only while
    /// the focused agent's mask leaves it a single action, so a frozen agent
    /// (mid-dig, or a resting harvester) runs itself out and the user is
    /// asked for a key only when there is a choice to make.
    fn clock_runs(&self) -> bool {
        match self.pacing {
            PacingMode::FreeRun => true,
            PacingMode::StepOnInput => !self.episode_done() && self.forced_action().is_some(),
        }
    }

    /// The focused agent's action for a clock step: under step-on-input the
    /// one action its mask allows, under free-run the policy's own pick.
    fn clock_action(&self) -> Option<VocabId> {
        match self.pacing {
            PacingMode::FreeRun => None,
            PacingMode::StepOnInput => self.forced_action(),
        }
    }

    /// Steps the env on the [`RenderApp::target_fps`] clock for whichever
    /// pacing is running it, and parks the clock when neither is.
    fn run_clock(&mut self, ui: &egui::Ui) {
        if !self.clock_runs() {
            self.next_step_time = None;
            return;
        }

        let now = ui.input(|i| i.time);
        let interval = f64::from(1.0 / self.target_fps);
        // the first step after the clock starts plays immediately
        let mut next = self.next_step_time.unwrap_or(now);

        // cap the catch-up after a stall (window drag, slow policy)
        let mut steps = 0;
        while now >= next && steps < MAX_CATCH_UP_STEPS {
            self.step_env(self.clock_action());
            next += interval;
            steps += 1;
            // a step can hand the choice back: stop before overrunning it
            if !self.clock_runs() {
                break;
            }
        }
        // whatever debt remains after the cap is forgiven, not replayed
        if now >= next {
            next = now + interval;
        }
        self.next_step_time = Some(next);

        let wait = std::time::Duration::from_secs_f64((next - now).max(0.001));
        ui.ctx().request_repaint_after(wait);
    }

    /// Episode, focused agent, modes and action keys, down the right edge. A landscape window spends no map on it: the
    /// map runs out of height first and would leave this space black.
    fn side_panel_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        section_heading(ui, "episode");
        egui::Grid::new("episode").num_columns(2).show(ui, |ui| {
            ui.label("step");
            ui.label(format!("{} / {}", self.step_count, self.length));
            ui.end_row();

            ui.label("seed");
            let mut seed = self.seed;
            ui.add(
                egui::DragValue::new(&mut seed)
                    .speed(0.1)
                    .update_while_editing(false),
            )
            .on_hover_text("edit to restart on that seed; pass it back in to replay this episode");
            if seed != self.seed {
                self.reseed(seed);
            }
            ui.end_row();

            if let Some(current) = self.task {
                ui.label("task");
                let mut selected = current;
                egui::ComboBox::from_id_salt("task")
                    .selected_text(&self.task_names[current])
                    .show_ui(ui, |ui| {
                        for (task, name) in self.task_names.iter().enumerate() {
                            ui.selectable_value(&mut selected, task, name);
                        }
                    })
                    .response
                    .on_hover_text(command_hint(Command::NextTask));
                self.select_task(selected);
                ui.end_row();
            }
        });

        ui.add_space(12.0);
        ui.horizontal(|ui| {
            // one-based for people; the index stays zero-based everywhere else
            section_heading(ui, &format!("agent {}", self.focused_agent + 1));
            ui.label(RichText::new(format!("({} total)", self.env.num_agents())).weak());
            if self.env.num_agents() > 1 {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("next")
                        .on_hover_text(command_hint(Command::NextAgent))
                        .clicked()
                    {
                        self.run_command(Command::NextAgent, ui);
                    }
                });
            }
        });
        let agent = self.focused_agent;
        let stepped = self.step_count > 0 && agent < self.buffers.num_agents();
        egui::Grid::new("agent").num_columns(2).show(ui, |ui| {
            ui.label("action");
            let action = stepped
                .then(|| {
                    let id = self.buffers.last_action[agent] as usize;
                    self.env.action_vocab().symbols().get(id).copied()
                })
                .flatten()
                .map_or("–", keys::action_label);
            ui.label(action);
            ui.end_row();

            ui.label("reward");
            ui.label(signed(ui, stepped.then(|| self.buffers.reward[agent])));
            ui.end_row();

            ui.label("return");
            ui.label(signed(ui, stepped.then(|| self.returns[agent])));
            ui.end_row();
        });

        ui.add_space(12.0);
        section_heading(ui, "mode");
        if toggle(
            ui,
            self.pacing == PacingMode::StepOnInput,
            ("step-on-input", "free-run"),
            command_hint(Command::TogglePacing),
        ) {
            self.run_command(Command::TogglePacing, ui);
        }
        if self.pacing == PacingMode::StepOnInput && self.clock_runs() {
            ui.label(RichText::new("auto: one legal action").weak());
        }
        if toggle(
            ui,
            self.view_mode == ViewMode::BirdsEye,
            ("bird's-eye", "agent view"),
            command_hint(Command::ToggleView),
        ) {
            self.run_command(Command::ToggleView, ui);
        }
        // the agent view draws the observations, fog and all, as encoded
        let mut show_fov = self.show_fov;
        let fov = ui
            .add_enabled(
                self.view_mode == ViewMode::BirdsEye,
                egui::Checkbox::new(&mut show_fov, "field of view"),
            )
            .on_hover_text(command_hint(Command::ToggleFov));
        if fov.changed() {
            self.run_command(Command::ToggleFov, ui);
        }

        self.keys_ui(ui);
    }

    /// The action keys, dimming those the focused agent's mask rules out
    /// right now. The rest of the controls are one `?` away.
    fn keys_ui(&self, ui: &mut egui::Ui) {
        let mask = self.buffers.action_mask.row(self.focused_agent);
        let key_names = |keys: &[egui::Key]| {
            keys.iter()
                .map(|&key| keys::key_label(key))
                .collect::<Vec<_>>()
                .join(" ")
        };

        ui.add_space(12.0);
        section_heading(ui, "actions");
        egui::Grid::new("action keys").show(ui, |ui| {
            for (id, label, keys) in keys::action_bindings(self.env.action_vocab()) {
                let legal = mask.get(usize::from(id)).copied().unwrap_or(false);
                let text = |text: String| {
                    let text = RichText::new(text);
                    if legal { text } else { text.weak() }
                };
                ui.label(text(key_names(keys)).monospace());
                ui.label(text(label.to_owned()));
                ui.end_row();
            }
        });

        ui.add_space(4.0);
        ui.label(RichText::new(format!(
            "{} all controls",
            command_hint(Command::ShowControls)
        )));
    }

    /// The command keys, over the map until the next input.
    fn controls_ui(&self, ctx: &egui::Context) {
        egui::Modal::new(egui::Id::new("controls")).show(ctx, |ui| {
            section_heading(ui, "controls");
            egui::Grid::new("command keys").show(ui, |ui| {
                for (key, command) in keys::command_bindings() {
                    if command == Command::NextTask && self.task.is_none() {
                        continue;
                    }
                    ui.label(RichText::new(keys::key_label(key)).monospace());
                    ui.label(command.label());
                    ui.end_row();
                }
                ui.label(RichText::new("click").monospace());
                ui.label("focus an agent");
                ui.end_row();
            });
            ui.add_space(4.0);
            ui.label(RichText::new("any key or click closes this").weak());
        });
    }

    /// The entire map; click an agent to focus it.
    fn birds_eye_ui(&mut self, ui: &mut egui::Ui) {
        let layout = GridLayout::fill(
            ui.max_rect(),
            self.settings.tile_width,
            self.settings.tile_height,
        );

        let response = ui.allocate_rect(layout.grid_rect(), egui::Sense::click());
        if response.clicked()
            && let Some((x, y)) = response
                .interact_pointer_pos()
                .and_then(|pos| layout.pos_to_cell(pos))
        {
            // agents can share a tile; the lowest index wins
            if let Some(agent) = self
                .render_state
                .agent_positions
                .iter()
                .position(|p| (p.x as usize, p.y as usize) == (x, y))
            {
                self.focused_agent = agent;
            }
        }

        // clip so the focused view rect can't overhang into the letterbox
        let painter = ui.painter_at(layout.grid_rect());
        let renderer = self
            .tilemap_renderer
            .as_mut()
            .expect("initialized at the top of ui()");
        if self.tilemap_dirty {
            // Keep the union of every agent's encoded line of sight bright;
            // a vocabulary without a mask makes the whole window visible.
            let seen = self.show_fov.then(|| {
                visible_tiles(
                    &self.settings,
                    &self.render_state.agent_positions,
                    self.buffers.obs.slice(s![.., .., .., 0]),
                    self.settings.obs_vocab.get(symbols::TILE_MASK),
                )
            });
            let tilemap = &self.render_state.tilemap;
            let art = &self.art;
            renderer.update(layout.cols, layout.rows, |x, y| {
                let visible = seen.as_ref().is_none_or(|seen| seen[[x, y]]);
                (art[usize::from(tilemap[[x, y]])], visible)
            });
            self.tilemap_dirty = false;
        }
        renderer.paint(&painter, layout.grid_rect());

        // outline the focused agent and its field of view, so the partial
        // observability the env exposes is visible.
        if let Some(position) = self.render_state.agent_positions.get(self.focused_agent) {
            let (x, y) = (position.x as f32, position.y as f32);

            painter.rect_stroke(
                layout.cell_rect(x, y, 1.0, 1.0),
                0.0,
                Stroke::new(2.0, Color32::YELLOW),
                StrokeKind::Inside,
            );
        }
    }

    /// The focused agent's observations as the env encoded them, which is
    /// what a policy sees. The agent sits at the centre of the FOV band and
    /// the UI band fills the rows above it, leaving the agent a little below
    /// the centre of the grid as drawn.
    fn pov_ui(&mut self, ui: &mut egui::Ui) {
        let layout = GridLayout::fill(
            ui.max_rect(),
            self.settings.view_width,
            self.settings.view_height,
        );
        let renderer = self
            .tilemap_renderer
            .as_mut()
            .expect("initialized at the top of ui()");
        if self.tilemap_dirty {
            let obs = &self.buffers.obs;
            let focused = self.focused_agent;
            let art = &self.art;
            renderer.update(layout.cols, layout.rows, |x, y| {
                (art[usize::from(obs[[focused, x, y, 0]])], true)
            });
            self.tilemap_dirty = false;
        }
        renderer.paint(ui.painter(), layout.grid_rect());
    }
}

fn section_heading(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).strong());
}

/// "P" style hover text naming the key bound to `command`.
fn command_hint(command: Command) -> String {
    keys::command_bindings()
        .find(|&(_, bound)| bound == command)
        .map(|(key, _)| keys::key_label(key).to_owned())
        .unwrap_or_default()
}

/// Two-way switch as a pair of selectable labels; true when the user clicked
/// the unselected side.
fn toggle(ui: &mut egui::Ui, first: bool, labels: (&str, &str), hint: String) -> bool {
    ui.horizontal(|ui| {
        let a = ui.selectable_label(first, labels.0).on_hover_text(&hint);
        let b = ui.selectable_label(!first, labels.1).on_hover_text(&hint);
        (a.clicked() && !first) || (b.clicked() && first)
    })
    .inner
}

/// A reward or return, green above zero and red below; a dash before the
/// first step.
fn signed(ui: &egui::Ui, value: Option<f32>) -> RichText {
    let Some(value) = value else {
        return RichText::new("–").monospace();
    };
    let text = RichText::new(format!("{value:+.2}")).monospace();
    if value > 0.0 {
        text.color(Color32::from_rgb(120, 200, 120))
    } else if value < 0.0 {
        text.color(ui.visuals().error_fg_color)
    } else {
        text
    }
}

impl eframe::App for RenderApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        if self.tilemap_renderer.is_none() {
            let state = frame
                .wgpu_render_state()
                .expect("the tilemap renderer requires the wgpu backend");
            self.tilemap_renderer = Some(TilemapRenderer::new(state));
        }

        // run the policy ahead of input, so a keypress finds its step's
        // actions already in hand instead of waiting on inference
        if !self.actions_ready {
            self.compute_actions();
        }

        // an open overlay owns the keyboard: Escape closes it rather than
        // quitting, and no keypress steps the env behind it
        let input = if self.show_controls {
            // any press closes the reference, and is spent on closing it
            let dismissed = ui.input(|state| {
                state.events.iter().any(|event| {
                    matches!(
                        event,
                        egui::Event::Key { pressed: true, .. }
                            | egui::Event::PointerButton { pressed: true, .. }
                    )
                })
            });
            self.show_controls = !dismissed;
            None
        } else if egui::Popup::is_any_open(ui.ctx()) || ui.ctx().egui_wants_keyboard_input() {
            // typing into the seed field is not playing, and a control Tab
            // moved focus onto keeps the keys until a click or Esc drops it
            None
        } else {
            ui.input(|state| keys::read(state, self.env.action_vocab()))
        };
        match input {
            Some(Input::Command(command)) => self.run_command(command, ui),
            // the keyboard steps the env exactly when the clock does not:
            // free-run leaves every agent to the policy, and so does
            // step-on-input while the focused agent has no choice to make
            Some(Input::Action(action)) if !self.clock_runs() => self.step_env(Some(action)),
            _ => {}
        }

        self.run_clock(ui);

        egui::Panel::right("side")
            .resizable(false)
            .exact_size(SIDE_PANEL_WIDTH)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.side_panel_ui(ui));
            });

        // after the side panel: its task picker can swap in a different map
        // size, and the settings and the render state must agree when drawn
        if self.render_state_dirty {
            self.env.render_state_into(&mut self.render_state);
            self.render_state_dirty = false;
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(Color32::BLACK))
            .show(ui, |ui| match self.view_mode {
                ViewMode::BirdsEye => self.birds_eye_ui(ui),
                ViewMode::AgentPov => self.pov_ui(ui),
            });

        if self.show_controls {
            self.controls_ui(ui.ctx());
        }

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
            .with_inner_size([1140.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native("mapox", options, Box::new(move |_cc| Ok(Box::new(app))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::find_return::{FindReturn, FindReturnConfig};
    use crate::envs::scouts::{Scouts, ScoutsConfig};
    use crate::policy::RandomPolicy;
    use crate::render::env::visible_tiles;
    use crate::wrappers::multitask::{MultitaskWrapper, tests::mixed_specs};
    use std::sync::{Arc, Mutex};

    /// What a policy was handed over a run: every reset, and every timestep
    /// it acted on together with the actions it returned.
    #[derive(Default)]
    struct Transcript {
        resets: Vec<usize>,
        acted: Vec<(TimeStepBuffers, Vec<VocabId>)>,
    }

    struct RecordingPolicy {
        inner: RandomPolicy,
        transcript: Arc<Mutex<Transcript>>,
    }

    impl Policy for RecordingPolicy {
        fn act(
            &mut self,
            timestep: &crate::timestep::TimeStepRef<'_>,
            actions: &mut [VocabId],
        ) -> Result<(), crate::policy::PolicyError> {
            self.inner.act(timestep, actions)?;
            let mut transcript = self.transcript.lock().unwrap();
            transcript
                .acted
                .push((timestep.to_buffers(), actions.to_vec()));
            Ok(())
        }

        fn reset(
            &mut self,
            num_agents: usize,
            seed: u64,
        ) -> Result<(), crate::policy::PolicyError> {
            self.transcript.lock().unwrap().resets.push(num_agents);
            self.inner.reset(num_agents, seed)
        }
    }

    /// An app driving `env` under a [`RecordingPolicy`], and the transcript
    /// it records into. Drop the app before unwrapping the transcript.
    fn recording_app(
        env: impl Environment + 'static,
        length: usize,
        seed: u64,
    ) -> (RenderApp, Arc<Mutex<Transcript>>) {
        let transcript = Arc::new(Mutex::new(Transcript::default()));
        let policy = RecordingPolicy {
            inner: RandomPolicy::new(),
            transcript: transcript.clone(),
        };
        let app = RenderApp::new(Box::new(env), length, seed, Box::new(policy));
        (app, transcript)
    }

    /// The env seed of the app's `episode`th episode: each one bumps the
    /// app seed, and both `new` and `reset` draw the env's seed first.
    fn app_episode_seed(app_seed: u64, episode: usize) -> u64 {
        SmallRng::seed_from_u64(app_seed + episode as u64).random()
    }

    /// The play loop hands the policy exactly the timesteps an enjoy-mode
    /// env emits for the actions the policy picks, task ids included; the
    /// batch test in the multitask wrapper ties those to the training rows.
    /// The one timestep it withholds is the terminal one: at `length` the
    /// next step resets the episode instead of asking the policy to act.
    #[test]
    fn the_policy_sees_the_enjoy_mode_timesteps() {
        const LENGTH: usize = 6;
        const EPISODES: usize = 2;
        const SEED: u64 = 3;
        let specs = mixed_specs();

        for task in 0..specs.len() {
            let mut env = MultitaskWrapper::new(&specs, LENGTH).unwrap();
            env.set_enjoy_mode(Some(task));
            let num_agents = env.num_agents();

            let (mut app, transcript) = recording_app(env, LENGTH, SEED);
            // each episode is LENGTH steps plus the call that resets it
            for _ in 0..EPISODES * (LENGTH + 1) - 1 {
                app.step_env(None);
            }
            drop(app);
            let transcript = Arc::into_inner(transcript).unwrap().into_inner().unwrap();

            assert_eq!(transcript.resets, vec![num_agents; EPISODES]);
            assert_eq!(transcript.acted.len(), EPISODES * LENGTH);

            let mut replay = MultitaskWrapper::new(&specs, LENGTH).unwrap();
            replay.set_enjoy_mode(Some(task));
            let mut buffers = TimeStepBuffers::new(&replay);

            for (episode, acted) in transcript.acted.chunks(LENGTH).enumerate() {
                replay.reset(app_episode_seed(SEED, episode), &mut buffers.view_mut());
                for (step, (seen, actions)) in acted.iter().enumerate() {
                    assert_eq!(
                        seen.differing_fields(&buffers),
                        Vec::<&str>::new(),
                        "task {task} episode {episode} step {step}: the policy's view diverged",
                    );
                    assert!(seen.task_ids.iter().all(|&id| id == task as i32));
                    assert!(seen.time.iter().all(|&time| time == step as i32));
                    replay.step(actions, &mut buffers.view_mut());
                }
                // the step past the last action the policy saw ends the episode
                assert!(buffers.terminated.iter().all(|&done| done));
                assert!(buffers.time.iter().all(|&time| time == LENGTH as i32));
            }
        }
    }

    /// The viewer never runs a multitask batch: it starts on the first task,
    /// unless the caller already put the env in enjoy mode for another.
    #[test]
    fn a_multitask_env_plays_one_task() {
        let specs = mixed_specs();

        let (app, _) = recording_app(MultitaskWrapper::new(&specs, 8).unwrap(), 8, 0);
        assert_eq!(app.task, Some(0));
        assert_eq!(app.task_names, ["scouts", "fr", "snake"]);
        assert_eq!(app.env.enjoy_task(), Some(0));
        assert_eq!(app.buffers.num_agents(), 3, "one scouts env");

        let mut env = MultitaskWrapper::new(&specs, 8).unwrap();
        env.set_enjoy_mode(Some(2));
        let (app, _) = recording_app(env, 8, 0);
        assert_eq!(app.task, Some(2));
        assert_eq!(app.buffers.num_agents(), 4, "one snake env");
    }

    #[test]
    fn a_single_task_env_has_no_task_to_pick() {
        let env = FindReturn::new(&FindReturnConfig::default(), 8);
        let app = RenderApp::new(Box::new(env), 8, 0, Box::new(RandomPolicy::new()));
        assert_eq!(app.task, None);
        assert!(app.task_names.is_empty());
    }

    /// Switching tasks resizes everything sized from the env, so the next
    /// steps run on the new task's agents and the policy is told about them.
    #[test]
    fn selecting_a_task_rebuilds_for_its_agents() {
        const LENGTH: usize = 4;
        let specs = mixed_specs();
        let (mut app, transcript) =
            recording_app(MultitaskWrapper::new(&specs, LENGTH).unwrap(), LENGTH, 0);

        app.focused_agent = 2;
        app.step_env(None);

        for (task, num_agents) in [(2, 4), (1, 3)] {
            app.select_task(task);
            assert_eq!(app.task, Some(task));
            assert_eq!(app.env.enjoy_task(), Some(task));
            assert_eq!(app.step_count, 0);
            assert_eq!(app.focused_agent, 0);
            assert_eq!(app.buffers.num_agents(), num_agents);
            assert_eq!(app.actions.len(), num_agents);
            assert_eq!(app.returns.len(), num_agents);
            assert_eq!(transcript.lock().unwrap().resets.last(), Some(&num_agents));

            // a whole episode and the reset after it run on the new shape
            for _ in 0..=LENGTH {
                app.step_env(None);
            }
            app.env.render_state_into(&mut app.render_state);
            assert_eq!(app.render_state.agent_positions.len(), num_agents);
        }

        let resets = transcript.lock().unwrap().resets.len();
        app.select_task(1);
        assert_eq!(
            transcript.lock().unwrap().resets.len(),
            resets,
            "reselecting the current task is a no-op"
        );
    }

    /// The overlay reads the same buffers the env writes: every agent must
    /// see the tile it stands on, and nobody can see past the map edge or
    /// through more tiles than the window holds.
    #[test]
    fn line_of_sight_matches_the_encoded_observations() {
        for mut env in [
            Box::new(FindReturn::new(&FindReturnConfig::default(), 512)) as Box<dyn Environment>,
            Box::new(Scouts::new(&ScoutsConfig::default(), 512)) as Box<dyn Environment>,
        ] {
            let settings = env.get_render_settings();
            let mask = settings
                .obs_vocab
                .get(symbols::TILE_MASK)
                .expect("these envs mask their observations");

            let mut buffers = TimeStepBuffers::new(env.as_ref());
            env.reset(0, &mut buffers.view_mut());

            let mut render_state = GridRenderState::default();
            env.render_state_into(&mut render_state);

            let seen = visible_tiles(
                &settings,
                &render_state.agent_positions,
                buffers.obs.slice(s![.., .., .., 0]),
                Some(mask),
            );

            let per_agent_window = settings.view_width * settings.fov_height();
            let seen_count = seen.iter().filter(|&&seen| seen).count();
            assert!(seen_count > 0);
            assert!(seen_count <= env.num_agents() * per_agent_window);

            for position in &render_state.agent_positions {
                assert!(seen[position.idx()], "agent's own tile is always seen");
            }
        }
    }

    #[test]
    fn scouts_render_settings_and_fov_are_correct() {
        let env = Box::new(Scouts::new(
            &ScoutsConfig {
                width: 21,
                height: 21,
                view_width: 11,
                view_height: 11,
                ..Default::default()
            },
            512,
        ));
        let settings = env.get_render_settings();
        assert_eq!(settings.tile_width, 21);
        assert_eq!(settings.tile_height, 21);
        assert_eq!(settings.view_width, 11);
        assert_eq!(settings.view_height, 13);
        assert_eq!(settings.ui_height, 2);
        assert_eq!(settings.fov_height(), 11);

        let mut render_state = GridRenderState::default();
        env.render_state_into(&mut render_state);
        assert_eq!(render_state.tilemap.dim(), (21, 21));
        for pos in &render_state.agent_positions {
            assert!(pos.x >= 0 && (pos.x as usize) < 21);
            assert!(pos.y >= 0 && (pos.y as usize) < 21);
        }
    }

    #[test]
    fn find_return_render_settings_and_fov_are_correct() {
        let env = Box::new(FindReturn::new(
            &FindReturnConfig {
                width: 21,
                height: 21,
                view_width: 11,
                view_height: 11,
                ..Default::default()
            },
            512,
        ));
        let settings = env.get_render_settings();
        assert_eq!(settings.tile_width, 21);
        assert_eq!(settings.tile_height, 21);
        assert_eq!(settings.view_width, 11);
        assert_eq!(settings.view_height, 13);
        assert_eq!(settings.ui_height, 2);
        assert_eq!(settings.fov_height(), 11);

        let mut render_state = GridRenderState::default();
        env.render_state_into(&mut render_state);
        assert_eq!(render_state.tilemap.dim(), (21, 21));
        for pos in &render_state.agent_positions {
            assert!(pos.x >= 0 && (pos.x as usize) < 21);
            assert!(pos.y >= 0 && (pos.y as usize) < 21);
        }
    }
}
