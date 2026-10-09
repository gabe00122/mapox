use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SurvivalConfig {
    pub num_agents: usize,

    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,
    /// How far an agent sees around itself in the dark. Past it, only lit
    /// ground shows.
    pub night_vision_radius: i32,
}

impl Default for SurvivalConfig {
    fn default() -> Self {
        Self {
            num_agents: 8,
            width: 40,
            height: 40,
            view_width: 15,
            view_height: 15,
            night_vision_radius: 2,
        }
    }
}
