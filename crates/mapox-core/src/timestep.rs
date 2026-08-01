use ndarray::{ArrayViewMut1, ArrayViewMut2, ArrayViewMut4};

/// Channels per observation cell: tile, direction, team, health.
pub const OBS_CHANNELS: usize = 4;

// python bool's are u8

#[derive(Debug)]
pub struct TimeStepMut<'a> {
    pub obs: ArrayViewMut4<'a, i8>,   // (num_agents, view_w, view_h, 4)
    pub time: ArrayViewMut1<'a, i32>, // (num_agents,)
    pub terminated: ArrayViewMut1<'a, u8>, // (num_agents,)
    pub last_action: ArrayViewMut1<'a, i32>, // (num_agents,)
    pub reward: ArrayViewMut1<'a, f32>, // (num_agents,)
    pub action_mask: ArrayViewMut2<'a, u8>, // (num_agents, n_actions)
    pub task_ids: ArrayViewMut1<'a, i32>, // (num_agents,)
}
