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
    pub obs: ArrayViewMut4<'a, VocabId>, // (num_agents, view_w, view_h, OBS_CHANNELS)
    pub time: ArrayViewMut1<'a, i32>,    // (num_agents,)
    pub terminated: ArrayViewMut1<'a, bool>, // (num_agents,)
    pub last_action: ArrayViewMut1<'a, VocabId>, // (num_agents,)
    pub reward: ArrayViewMut1<'a, f32>,  // (num_agents,)
    pub action_mask: ArrayViewMut2<'a, bool>, // (num_agents, n_actions)
    pub task_ids: ArrayViewMut1<'a, i32>, // (num_agents,)
}

#[derive(Debug)]
pub struct TimeStepRef<'a> {
    pub obs: ArrayView4<'a, VocabId>, // (num_agents, view_w, view_h, OBS_CHANNELS)
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

    pub fn partition_mut(&mut self, lens: &[usize]) -> Vec<TimeStepMut<'_>> {
        let (w, h, c) = (self.obs.dim().1, self.obs.dim().2, self.obs.dim().3);
        let num_actions = self.action_mask.dim().1;

        unsafe {
            let obs_ptr = self.obs.as_mut_ptr();
            let time_ptr = self.time.as_mut_ptr();
            let terminated_ptr = self.terminated.as_mut_ptr();
            let last_action_ptr = self.last_action.as_mut_ptr();
            let reward_ptr = self.reward.as_mut_ptr();
            let action_mask_ptr = self.action_mask.as_mut_ptr();
            let task_ids_ptr = self.task_ids.as_mut_ptr();

            let mut out = Vec::with_capacity(lens.len());
            let mut offset = 0usize;

            for &len in lens {
                out.push(TimeStepMut {
                    obs: ArrayViewMut4::from_shape_ptr(
                        (len, w, h, c),
                        obs_ptr.add(offset * w * h * c),
                    ),
                    time: ArrayViewMut1::from_shape_ptr(len, time_ptr.add(offset)),
                    terminated: ArrayViewMut1::from_shape_ptr(len, terminated_ptr.add(offset)),
                    last_action: ArrayViewMut1::from_shape_ptr(len, last_action_ptr.add(offset)),
                    reward: ArrayViewMut1::from_shape_ptr(len, reward_ptr.add(offset)),
                    action_mask: ArrayViewMut2::from_shape_ptr(
                        (len, num_actions),
                        action_mask_ptr.add(offset),
                    ),
                    task_ids: ArrayViewMut1::from_shape_ptr(len, task_ids_ptr.add(offset)),
                });
                offset += len;
            }

            out
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
