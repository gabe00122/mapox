use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::{
        find_return::{FindReturn, FindReturnConfig},
        scouts::{Scouts, ScoutsConfig},
        snake::{Snake, SnakeConfig},
    },
    wrappers::{multitask::MultitaskWrapper, vector::VectorWrapper},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "env_type", rename_all = "snake_case")]
pub enum EnvConfig {
    RustFindReturn(FindReturnConfig),
    RustScouts(ScoutsConfig),
    RustSnake(SnakeConfig),
    #[serde(rename = "vec")]
    RustVec {
        num: usize,
        env: Box<EnvConfig>,
    },
    #[serde(rename = "multi")]
    RustMulti {
        envs: Vec<MultiEnvSpec>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MultiEnvSpec {
    pub name: String,
    pub num: usize,
    pub env: Box<EnvConfig>,
}

pub fn make(config: &EnvConfig, length: usize) -> Result<Box<dyn Environment>, String> {
    match config {
        EnvConfig::RustFindReturn(config) => Ok(Box::new(FindReturn::new(config, length))),
        EnvConfig::RustScouts(config) => Ok(Box::new(Scouts::new(config, length))),
        EnvConfig::RustSnake(config) => Ok(Box::new(Snake::new(config, length))),
        EnvConfig::RustVec { num, env } => Ok(Box::new(VectorWrapper::new(*num, env, length)?)),
        EnvConfig::RustMulti { envs } => Ok(Box::new(MultitaskWrapper::new(envs, length)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timestep::TimeStepBuffers;

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

    /// Same again for the snake, whose config carries no mapgen keys.
    #[test]
    fn a_snake_config_dump_parses() {
        let json = r#"{"env_type":"rust_snake","num_agents":4,"width":24,"height":24,
            "view_width":11,"view_height":11,"initial_length":3,"initial_food":8,
            "food_spawn_prob":0.15,"food_reward":1.0,"death_reward":-1.0}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(parsed, EnvConfig::RustSnake(SnakeConfig::default()));
    }

    /// The vectorized shape: `num` copies of a plain env config.
    #[test]
    fn a_vec_config_parses() {
        let json = r#"{"env_type":"vec","num":32,"env":{"env_type":"rust_scouts",
            "num_scouts":4,"num_harvesters":4,"num_treasures":12,"width":40,"height":40,
            "view_width":11,"view_height":13,"ui_height":2,"mapgen_threshold":0.3,
            "water_threshold":-0.45,"harvesters_move_every":6,
            "scout_reward":1.0,"harvester_reward":1.0}}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed,
            EnvConfig::RustVec {
                num: 32,
                env: Box::new(EnvConfig::RustScouts(ScoutsConfig::default())),
            }
        );
    }

    /// The multitask shape: named entries, each carrying its instance count
    /// and a plain env config.
    #[test]
    fn a_multi_config_parses() {
        let json = r#"{"env_type":"multi","envs":[
            {"name":"scouts","num":2,"env":{"env_type":"rust_scouts","num_scouts":4,
                "num_harvesters":4,"num_treasures":12,"width":40,"height":40,
                "view_width":11,"view_height":13,"ui_height":2,"mapgen_threshold":0.3,
                "water_threshold":-0.45,"harvesters_move_every":6,
                "scout_reward":1.0,"harvester_reward":1.0}},
            {"name":"fr","num":3,"env":{"env_type":"rust_find_return","num_agents":8,
                "num_flags":1,"width":40,"height":40,"view_width":11,"view_height":11,
                "mapgen_threshold":0.3,"water_threshold":-0.45,"digging_timeout":5,
                "preparation_steps":256,"treasure_reward":1.0}}
        ]}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed,
            EnvConfig::RustMulti {
                envs: vec![
                    MultiEnvSpec {
                        name: "scouts".into(),
                        num: 2,
                        env: Box::new(EnvConfig::RustScouts(ScoutsConfig::default())),
                    },
                    MultiEnvSpec {
                        name: "fr".into(),
                        num: 3,
                        env: Box::new(EnvConfig::RustFindReturn(FindReturnConfig::default())),
                    },
                ],
            }
        );
    }

    #[test]
    fn make_multi_groups_each_entry_and_assigns_task_ids_per_entry() {
        let config = EnvConfig::RustMulti {
            envs: vec![
                MultiEnvSpec {
                    name: "scouts".into(),
                    num: 2,
                    env: Box::new(EnvConfig::RustScouts(ScoutsConfig {
                        num_scouts: 2,
                        num_harvesters: 2,
                        width: 12,
                        height: 12,
                        view_width: 11,
                        view_height: 13,
                        ..ScoutsConfig::default()
                    })),
                },
                MultiEnvSpec {
                    name: "fr".into(),
                    num: 1,
                    env: Box::new(EnvConfig::RustFindReturn(FindReturnConfig {
                        num_agents: 4,
                        width: 12,
                        height: 12,
                        view_width: 11,
                        // FindReturn appends a 2-row UI strip, so 11 + 2 = 13
                        // matches the scouts obs view above.
                        view_height: 11,
                        ..FindReturnConfig::default()
                    })),
                },
            ],
        };

        let mut env = make(&config, 32).unwrap();
        assert_eq!(env.num_agents(), 12); // 2 * (2 + 2) + 4
        assert_eq!(env.num_tasks(), 2); // copies of an entry are one task

        let mut buffers = TimeStepBuffers::new(&*env);
        env.reset(1, &mut buffers.view_mut());
        assert_eq!(
            buffers.task_ids.to_vec(),
            (0..8)
                .map(|_| 0)
                .chain((0..4).map(|_| 1))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn make_vec_builds_num_copies() {
        let scouts = Box::new(EnvConfig::RustScouts(ScoutsConfig {
            num_scouts: 1,
            num_harvesters: 1,
            width: 12,
            height: 12,
            ..ScoutsConfig::default()
        }));

        let env = make(
            &EnvConfig::RustVec {
                num: 3,
                env: scouts.clone(),
            },
            32,
        )
        .unwrap();
        assert_eq!(env.num_agents(), 6);

        assert_eq!(env.num_tasks(), 1);

        let env = make(
            &EnvConfig::RustVec {
                num: 1,
                env: scouts,
            },
            32,
        )
        .unwrap();
        assert_eq!(env.num_agents(), 2);
    }

    #[test]
    fn make_rejects_degenerate_configs() {
        assert!(make(&EnvConfig::RustMulti { envs: vec![] }, 32).is_err());
        assert!(
            make(
                &EnvConfig::RustMulti {
                    envs: vec![MultiEnvSpec {
                        name: "scouts".into(),
                        num: 0,
                        env: Box::new(EnvConfig::RustScouts(ScoutsConfig::default())),
                    }],
                },
                32
            )
            .is_err()
        );
        assert!(
            make(
                &EnvConfig::RustVec {
                    num: 0,
                    env: Box::new(EnvConfig::RustScouts(ScoutsConfig::default())),
                },
                32
            )
            .is_err()
        );
    }

    #[test]
    fn make_rejects_nested_multitask() {
        let config = EnvConfig::RustMulti {
            envs: vec![MultiEnvSpec {
                name: "inner".into(),
                num: 1,
                env: Box::new(EnvConfig::RustMulti {
                    envs: vec![MultiEnvSpec {
                        name: "fr".into(),
                        num: 1,
                        env: Box::new(EnvConfig::RustFindReturn(FindReturnConfig::default())),
                    }],
                }),
            }],
        };

        assert!(make(&config, 32).is_err());
    }

    /// The obs view is what `MultitaskWrapper` serves from one shared shape,
    /// not the config field: FindReturn's actual view is `view_height + 2`
    /// (UI strip), so its default 11 actually matches scouts' default 13.
    /// A config view of 13 (→ obs 15) does not.
    #[test]
    fn make_rejects_mismatched_view_sizes() {
        let config = EnvConfig::RustMulti {
            envs: vec![
                MultiEnvSpec {
                    name: "scouts".into(),
                    num: 1,
                    env: Box::new(EnvConfig::RustScouts(ScoutsConfig::default())),
                },
                MultiEnvSpec {
                    name: "fr".into(),
                    num: 1,
                    env: Box::new(EnvConfig::RustFindReturn(FindReturnConfig {
                        view_height: 13,
                        ..FindReturnConfig::default()
                    })),
                },
            ],
        };

        assert!(matches!(
            make(&config, 32),
            Err(err)
                if err.contains("multitask env 1 'fr' has view 11x15, but env 0 has view 11x13")
        ));
    }
}
