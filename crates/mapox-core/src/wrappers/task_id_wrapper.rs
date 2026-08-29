use crate::env::Environment;
use crate::render::env::GridRenderSettings;
use crate::render::env::GridRenderState;
use crate::spec::ActionSpec;
use crate::timestep::TimeStepMut;
use crate::vocab::VocabId;
use crate::vocab::Vocabulary;

pub struct TaskIdWrapper {
    task_id: i32,
    env: Box<dyn Environment>,
}

impl TaskIdWrapper {
    pub fn new(env: Box<dyn Environment>, task_id: i32) -> Self {
        Self { task_id, env }
    }
}

impl Environment for TaskIdWrapper {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.env.reset(seed, timestep);
        timestep.task_ids.fill(self.task_id);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.env.step(actions, timestep);
        timestep.task_ids.fill(self.task_id);
    }

    fn num_agents(&self) -> usize {
        self.env.num_agents()
    }

    fn action_spec(&self) -> ActionSpec {
        self.env.action_spec()
    }

    fn observation_spec(&self) -> crate::spec::ObservationSpec {
        self.env.observation_spec()
    }

    fn obs_vocab(&self) -> &Vocabulary {
        self.env.obs_vocab()
    }

    fn action_vocab(&self) -> &Vocabulary {
        self.env.action_vocab()
    }

    fn get_render_settings(&self) -> GridRenderSettings {
        self.env.get_render_settings()
    }

    fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
        self.env.render_state_into(grid_render_state)
    }
}
