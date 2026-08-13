use ndarray::{
    Array1, Array2, Array4, ArrayView1, ArrayView2, ArrayView4, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMut4,
};

use crate::env::Environment;
use crate::vocab::VocabId;

pub const OBS_CHANNELS: usize = 1;

// python bool's are u8

#[derive(Debug)]
pub struct TimeStepMut<'a> {
    pub obs: ArrayViewMut4<'a, VocabId>, // (num_agents, view_w, view_h, 4)
    pub time: ArrayViewMut1<'a, i32>,    // (num_agents,)
    pub terminated: ArrayViewMut1<'a, bool>, // (num_agents,)
    pub last_action: ArrayViewMut1<'a, VocabId>, // (num_agents,)
    pub reward: ArrayViewMut1<'a, f32>,  // (num_agents,)
    pub action_mask: ArrayViewMut2<'a, bool>, // (num_agents, n_actions)
    pub task_ids: ArrayViewMut1<'a, i32>, // (num_agents,)
}

#[derive(Debug)]
pub struct TimeStepRef<'a> {
    pub obs: ArrayView4<'a, VocabId>, // (num_agents, view_w, view_h, 4)
    pub time: ArrayView1<'a, i32>,    // (num_agents,)
    pub terminated: ArrayView1<'a, bool>, // (num_agents,)
    pub last_action: ArrayView1<'a, VocabId>, // (num_agents,)
    pub reward: ArrayView1<'a, f32>,  // (num_agents,)
    pub action_mask: ArrayView2<'a, bool>, // (num_agents, n_actions)
    pub task_ids: ArrayView1<'a, i32>, // (num_agents,)
}

impl TimeStepMut<'_> {
    pub fn view(&self) -> TimeStepRef<'_> {
        TimeStepRef {
            obs: self.obs.view(),
            time: self.time.view(),
            terminated: self.terminated.view(),
            last_action: self.last_action.view(),
            reward: self.reward.view(),
            action_mask: self.action_mask.view(),
            task_ids: self.task_ids.view(),
        }
    }
}
#[derive(Debug)]
pub struct TimeStepBuffers {
    pub obs: Array4<VocabId>,
    pub time: Array1<i32>,
    pub terminated: Array1<bool>,
    pub last_action: Array1<VocabId>,
    pub reward: Array1<f32>,
    pub action_mask: Array2<bool>,
    pub task_ids: Array1<i32>,
}

impl TimeStepBuffers {
    pub fn new(env: &dyn Environment) -> Self {
        let obs_spec = env.observation_spec();
        Self::with_shape(
            env.num_agents(),
            obs_spec.width as usize,
            obs_spec.height as usize,
            env.action_spec().num_actions,
        )
    }

    /// Sized by hand, for callers with no env to ask (tests, mostly).
    pub fn with_shape(
        num_agents: usize,
        view_width: usize,
        view_height: usize,
        num_actions: usize,
    ) -> Self {
        Self {
            obs: Array4::zeros((num_agents, view_width, view_height, OBS_CHANNELS)),
            time: Array1::zeros(num_agents),
            terminated: Array1::default(num_agents),
            last_action: Array1::zeros(num_agents),
            reward: Array1::zeros(num_agents),
            action_mask: Array2::default((num_agents, num_actions)),
            task_ids: Array1::zeros(num_agents),
        }
    }

    pub fn num_agents(&self) -> usize {
        self.time.len()
    }

    pub fn view_mut(&mut self) -> TimeStepMut<'_> {
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

    pub fn view(&self) -> TimeStepRef<'_> {
        TimeStepRef {
            obs: self.obs.view(),
            time: self.time.view(),
            terminated: self.terminated.view(),
            last_action: self.last_action.view(),
            reward: self.reward.view(),
            action_mask: self.action_mask.view(),
            task_ids: self.task_ids.view(),
        }
    }
}
