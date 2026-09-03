use ndarray::Array2;

use crate::{
    env::Environment,
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
};

pub struct VocabWrapper {
    action_vocab: Vocabulary,
    observation_vocab: Vocabulary,
    global_to_local_action: Vec<VocabId>,
    local_to_global_action: Vec<VocabId>,
    local_to_global_obs: Vec<VocabId>,
    inner: Box<dyn Environment>,
    temp_actions: Vec<VocabId>,
    temp_action_mask: Array2<bool>,
}

impl VocabWrapper {
    pub fn new(
        inner: Box<dyn Environment>,
        action_vocab: &Vocabulary,
        observation_vocab: &Vocabulary,
    ) -> Self {
        let num_global_actions = action_vocab.len();
        let num_agents = inner.num_agents();

        Self {
            global_to_local_action: action_vocab.lut_to(inner.action_vocab(), 0),
            local_to_global_action: inner.action_vocab().lut_to(action_vocab, 0),
            local_to_global_obs: inner.obs_vocab().lut_to(observation_vocab, 0),
            action_vocab: action_vocab.clone(),
            observation_vocab: observation_vocab.clone(),
            inner,
            temp_actions: vec![0; num_agents],
            temp_action_mask: Array2::from_elem((num_agents, num_global_actions), false),
        }
    }

    fn encode_timestep(&mut self, timestep: &mut TimeStepMut) {
        self.temp_action_mask.assign(&timestep.action_mask);
        timestep.action_mask.fill(false);

        for agent_id in 0..self.num_agents() {
            for local_action in 0..self.inner.action_vocab().len() {
                let mask = self.temp_action_mask[[agent_id, local_action]];
                let global_action = self.local_to_global_action[local_action];
                timestep.action_mask[[agent_id, global_action as usize]] = mask;
            }
        }

        timestep
            .last_action
            .map_inplace(|action| *action = self.local_to_global_action[*action as usize]);

        timestep
            .obs
            .map_inplace(|obs| *obs = self.local_to_global_obs[*obs as usize]);
    }
}

impl Environment for VocabWrapper {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.inner.reset(seed, timestep);
        self.encode_timestep(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        for (&local, target) in actions.iter().zip(self.temp_actions.iter_mut()) {
            *target = self.global_to_local_action[local as usize];
        }

        self.inner.step(&mut self.temp_actions, timestep);
        self.encode_timestep(timestep);
    }

    fn action_spec(&self) -> ActionSpec {
        ActionSpec {
            num_actions: self.action_vocab.len(),
        }
    }

    fn observation_spec(&self) -> ObservationSpec {
        ObservationSpec {
            num_types: self.observation_vocab.len(),
            ..self.inner.observation_spec()
        }
    }

    fn action_vocab(&self) -> &Vocabulary {
        &self.action_vocab
    }

    fn obs_vocab(&self) -> &Vocabulary {
        &self.observation_vocab
    }

    fn num_agents(&self) -> usize {
        self.inner.num_agents()
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
