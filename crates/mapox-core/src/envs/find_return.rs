use rand::{Rng, rngs::SmallRng};

pub struct FindReturnConfig {
    pub num_agents: usize,
    pub num_flags: usize,

    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,

    pub mapgen_threshold: f64,
    pub digging_timeout: i32,
    pub treasure_reward: f64,
}

pub struct FindReturnAgent {
    pub x: i32,
    pub y: i32,
    pub found_reward: bool,
}

pub struct FindReturnState {
    pub agents: Vec<FindReturnAgent>,
    pub time: i32,
    pub map: Vec<u8>,
}

pub struct FindReturn {
    pub config: FindReturnConfig,
    pub state: Option<FindReturnState>,
}

impl FindReturn {
    fn new() {}
}

// impl Env for FindReturnConfig {
//     type EnvState = FindReturnState;

//     fn create(self, rngs: &mut SmallRng) -> FindReturnState {
//         let mut state = FindReturnState {
//             agents: Vec::with_capacity(self.num_agents),
//             time: 0,
//             map: Vec::with_capacity((self.width * self.height) as usize),
//         };

//         self.reset(&mut state, rngs);
//         state
//     }

//     fn reset(self, state: &mut Self::EnvState, _rngs: &mut SmallRng) {}
// }
