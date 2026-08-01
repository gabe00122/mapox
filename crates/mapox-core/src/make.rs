use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::find_return::{FindReturn, FindReturnConfig},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "env_type", rename_all = "snake_case")]
pub enum EnvConfig {
    FindReturn(FindReturnConfig),
}

pub fn make(config: &EnvConfig) -> Box<dyn Environment> {
    match config {
        EnvConfig::FindReturn(config) => Box::new(FindReturn::new(config)),
    }
}
