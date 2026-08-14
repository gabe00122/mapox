use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::find_return::{FindReturn, FindReturnConfig},
    wrappers::vector::VectorWrapper,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "env_type", rename_all = "snake_case")]
pub enum EnvConfig {
    RustFindReturn(FindReturnConfig),
}

pub fn make(config: &EnvConfig, length: usize) -> Box<dyn Environment> {
    match config {
        EnvConfig::RustFindReturn(config) => Box::new(FindReturn::new(config, length)),
    }
}

pub fn make_vec(config: &EnvConfig, length: usize, num_envs: usize) -> Box<dyn Environment> {
    if num_envs == 1 {
        return make(config, length);
    }

    let envs: Vec<Box<dyn Environment>> = (0..num_envs).map(|_| make(config, length)).collect();
    Box::new(VectorWrapper::new(envs))
}
