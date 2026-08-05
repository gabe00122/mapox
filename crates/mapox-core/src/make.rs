use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::find_return::{FindReturn, FindReturnConfig},
    wrappers::vector::VectorWrapper,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "env_type", rename_all = "snake_case")]
pub enum EnvConfig {
    FindReturn(FindReturnConfig),
}

pub fn make(config: &EnvConfig) -> Box<dyn Environment + Send + Sync> {
    match config {
        EnvConfig::FindReturn(config) => {
            let fr = FindReturn::new(config);
            let vec = VectorWrapper::new(fr, 128);
            Box::new(vec)
        }
    }
}
