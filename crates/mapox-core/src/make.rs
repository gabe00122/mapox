use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::{
        find_return::{FindReturn, FindReturnConfig},
        scouts::{Scouts, ScoutsConfig},
    },
    wrappers::vector::VectorWrapper,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "env_type", rename_all = "snake_case")]
pub enum EnvConfig {
    RustFindReturn(FindReturnConfig),
    RustScouts(ScoutsConfig),
}

pub fn make(config: &EnvConfig, length: usize) -> Box<dyn Environment> {
    match config {
        EnvConfig::RustFindReturn(config) => Box::new(FindReturn::new(config, length)),
        EnvConfig::RustScouts(config) => Box::new(Scouts::new(config, length)),
    }
}

pub fn make_vec(config: &EnvConfig, length: usize, num_envs: usize) -> Box<dyn Environment> {
    if num_envs == 1 {
        return make(config, length);
    }

    let envs: Vec<Box<dyn Environment>> = (0..num_envs).map(|_| make(config, length)).collect();
    Box::new(VectorWrapper::new(envs))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The python side builds these configs with pydantic and hands them over
    /// as JSON, so the tag and the field names are a contract between the two.
    /// `RustScoutsConfig.model_dump_json()`, verbatim:
    #[test]
    fn a_scouts_config_dump_parses() {
        let json = r#"{"env_type":"rust_scouts","num_scouts":4,"num_harvesters":4,
            "num_treasures":12,"width":40,"height":40,"view_width":11,"view_height":13,
            "ui_height":2,"mapgen_threshold":0.3,"water_threshold":-0.45,
            "harvesters_move_every":6,"scout_reward":1.0,"harvester_reward":1.0}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(parsed, EnvConfig::RustScouts(ScoutsConfig::default()));
    }

    /// And the same for the env that was here first.
    #[test]
    fn a_find_return_config_dump_parses() {
        let json = r#"{"env_type":"rust_find_return","num_agents":8,"num_flags":1,
            "width":40,"height":40,"view_width":11,"view_height":11,"mapgen_threshold":0.3,
            "water_threshold":-0.45,"digging_timeout":5,"preparation_steps":256,
            "treasure_reward":1.0}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed,
            EnvConfig::RustFindReturn(FindReturnConfig::default())
        );
    }
}
