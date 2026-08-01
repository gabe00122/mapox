type ObsType = i8;

// python bool's are u8

#[derive(Debug)]
pub struct TimeStepMut<'a> {
    pub obs: &'a mut [ObsType],     // (num_agents, view_w, view_h, 4)
    pub time: &'a mut [i32],        // (num_agents,)
    pub terminated: &'a mut [u8],   // (num_agents,)
    pub last_action: &'a mut [i32], // (num_agents,)
    pub reward: &'a mut [f32],      // (num_agents,)
    pub action_mask: &'a mut [u8],  // (num_agents, n_actions)
    pub task_ids: &'a mut [i32],    // (num_agents,)
}
