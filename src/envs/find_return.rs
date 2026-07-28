use crate::env::Env;
use rand::{Rng, rngs::SmallRng};

struct FindReturnConfig {
    num_agents: usize,
    num_flags: usize,

    width: i32,
    height: i32,
    view_width: i32,
    view_height: i32,

    mapgen_threshold: f64,
    digging_timeout: i32,
    treasure_reward: f64,
}

struct FindReturnAgent {
    x: i32,
    y: i32,
    found_reward: bool,
}

struct FindReturnState {
    agents: Vec<FindReturnAgent>,
    time: i32,
    map: Vec<u8>,
}

impl Env for FindReturnConfig {
    type EnvState = FindReturnState;

    fn create(self, rngs: &mut SmallRng) -> FindReturnState {
        let mut state = FindReturnState {
            agents: Vec::with_capacity(self.num_agents),
            time: 0,
            map: Vec::with_capacity((self.width * self.height) as usize),
        };

        self.reset(&mut state, rngs);
        state
    }

    fn reset(self, state: &mut Self::EnvState, rngs: &mut SmallRng) {}
}
