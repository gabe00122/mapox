use crate::env::Environment;
use crate::render::env::GridRenderSettings;
use crate::render::env::GridRenderState;
use crate::spec::ActionSpec;
use crate::timestep::TimeStepMut;
use crate::vocab::VocabId;
use crate::vocab::Vocabulary;

pub struct TaskIdWrapper {
    task_id: i32,
    inner: Box<dyn Environment>,
}

impl TaskIdWrapper {
    pub fn new(inner: Box<dyn Environment>, task_id: i32) -> Self {
        Self { task_id, inner }
    }
}

impl Environment for TaskIdWrapper {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.inner.reset(seed, timestep);
        timestep.task_ids.fill(self.task_id);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        self.inner.step(actions, timestep);
        timestep.task_ids.fill(self.task_id);
    }

    fn num_agents(&self) -> usize {
        self.inner.num_agents()
    }

    fn action_spec(&self) -> ActionSpec {
        self.inner.action_spec()
    }

    fn observation_spec(&self) -> crate::spec::ObservationSpec {
        self.inner.observation_spec()
    }

    fn obs_vocab(&self) -> &Vocabulary {
        self.inner.obs_vocab()
    }

    fn action_vocab(&self) -> &Vocabulary {
        self.inner.action_vocab()
    }

    fn get_render_settings(&self) -> GridRenderSettings {
        self.inner.get_render_settings()
    }

    fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
        self.inner.render_state_into(grid_render_state)
    }

    fn num_tasks(&self) -> usize {
        self.inner.num_tasks()
    }
}
