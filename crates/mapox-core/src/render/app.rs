//! The interactive render app: eframe owns the loop, a [`Policy`] supplies
//! actions once per env step, and in step-on-input pacing the keyboard
//! overrides the focused agent. Training keeps the inverse control model and
//! never touches this.

use crate::{
    env::Environment,
    policy::Policy,
    render::{
        env::{GridRenderSettings, GridRenderState, visible_tiles},
        grid::{GridLayout, draw_tile_grid},
        keys::{self, Command, Input},
        resolve_art,
        tileset::Tileset,
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

/// Fog over the tiles no agent can currently see: strong enough to read as
/// "not observed", faint enough that the map underneath stays legible. A
/// ~43% grey, premultiplied by hand because the unpremultiplied constructor
/// is not const.
const UNSEEN_TILE_OVERLAY: Color32 = Color32::from_rgba_premultiplied(41, 41, 41, 110);

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
    /// Rewritten by every env reset and step; the policy reads it through
    /// [`RenderApp::compute_actions`], and the hint bar reads reward and
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
    /// Sheet coordinates indexed by obs vocab id.
    art: Vec<(u32, u32)>,

    /// Per-agent reward summed since the last reset, so the hint bar can show
    /// where the focused agent's episode stands and not just the last step.
    returns: Vec<f32>,

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
            settings,
            render_state: GridRenderState::default(),
            art,
            returns: vec![0.0; num_agents],
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
        self.returns.fill(0.0);
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

    fn hint_text(&self) -> String {
        let pacing = match self.pacing {
            PacingMode::FreeRun => format!("free-run {}fps", self.target_fps),
            PacingMode::StepOnInput if self.clock_runs() => "step-on-input (auto)".to_owned(),
            PacingMode::StepOnInput => "step-on-input".to_owned(),
        };
        format!(
            "t={}   agent {}/{}   {}   {}",
            self.step_count,
            self.focused_agent,
            self.env.num_agents(),
            pacing,
            self.focused_agent_text(),
        )
    }

    fn focused_agent_text(&self) -> String {
        let agent = self.focused_agent;
        if self.step_count == 0 || agent >= self.buffers.num_agents() {
            return "action -   r -".to_owned();
        }

        let action = self
            .env
            .action_vocab()
            .symbols()
            .get(self.buffers.last_action[agent] as usize)
            .copied()
            .unwrap_or("?");

        format!(
            "action {}   r {:+.2}   return {:+.2}",
            action, self.buffers.reward[agent], self.returns[agent],
        )
    }

    /// The entire map; click an agent to focus it.
    fn birds_eye_ui(&mut self, ui: &mut egui::Ui) {
        let layout = GridLayout::fit(
            ui.max_rect(),
            self.settings.tile_width,
            self.settings.tile_height,
            ui.ctx().pixels_per_point(),
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
        let tileset = self.tileset.as_ref().expect("uploaded at the top of ui()");
        let tilemap = &self.render_state.tilemap;
        draw_tile_grid(&painter, tileset, &self.art, &layout, |x, y| {
            tilemap[[x, y]]
        });

        // grey out the tiles no agent's observation sees through, leaving
        // the union of everyone's line of sight at full brightness; the obs
        // vocab having no mask tile means nothing is ever hidden
        let mask = self.settings.obs_vocab.get(symbols::TILE_MASK);
        let seen = visible_tiles(
            &self.settings,
            &self.render_state.agent_positions,
            self.buffers.obs.slice(s![.., .., .., 0]),
            mask,
        );
        for x in 0..self.settings.tile_width {
            for y in 0..self.settings.tile_height {
                if !seen[[x, y]] {
                    painter.rect_filled(
                        layout.cell_rect(x as f32, y as f32, 1.0, 1.0),
                        0.0,
                        UNSEEN_TILE_OVERLAY,
                    );
                }
            }
        }

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
        let layout = GridLayout::fit(
            ui.max_rect(),
            self.settings.view_width,
            self.settings.view_height,
            ui.ctx().pixels_per_point(),
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
            // the keyboard steps the env exactly when the clock does not:
            // free-run leaves every agent to the policy, and so does
            // step-on-input while the focused agent has no choice to make
            Some(Input::Action(action)) if !self.clock_runs() => self.step_env(Some(action)),
            _ => {}
        }

        self.run_clock(ui);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::find_return::{FindReturn, FindReturnConfig};
    use crate::envs::scouts::{Scouts, ScoutsConfig};
    use crate::render::env::visible_tiles;

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
                ui_height: 2,
                ..Default::default()
            },
            512,
        ));
        let settings = env.get_render_settings();
        assert_eq!(settings.tile_width, 21);
        assert_eq!(settings.tile_height, 21);
        assert_eq!(settings.view_width, 11);
        assert_eq!(settings.view_height, 11);
        assert_eq!(settings.ui_height, 2);
        assert_eq!(settings.fov_height(), 9);

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
