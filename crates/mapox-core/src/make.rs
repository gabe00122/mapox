use serde::{Deserialize, Serialize};

#[cfg(not(target_arch = "wasm32"))]
use crate::wrappers::video::{VideoConfig, VideoWrapper};
use crate::{
    env::Environment,
    envs::{
        find_return::{FindReturn, FindReturnConfig},
        scouts::{Scouts, ScoutsConfig},
        snake::{Snake, SnakeConfig},
    },
    error::MapoxResult,
    wrappers::{multitask::MultitaskWrapper, vector::VectorWrapper},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "env_type", rename_all = "snake_case")]
pub enum EnvConfig {
    RustFindReturn(Box<FindReturnConfig>),
    RustScouts(Box<ScoutsConfig>),
    RustSnake(Box<SnakeConfig>),
    #[cfg(not(target_arch = "wasm32"))]
    RustVideo(Box<VideoConfig>),
    RustVec {
        num: usize,
        env: Box<EnvConfig>,
    },
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

pub fn make(config: &EnvConfig, length: usize) -> MapoxResult<Box<dyn Environment>> {
    match config {
        EnvConfig::RustFindReturn(config) => Ok(Box::new(FindReturn::new(config, length))),
        EnvConfig::RustScouts(config) => Ok(Box::new(Scouts::new(config, length))),
        EnvConfig::RustSnake(config) => Ok(Box::new(Snake::new(config, length))),
        #[cfg(not(target_arch = "wasm32"))]
        EnvConfig::RustVideo(config) => Ok(Box::new(VideoWrapper::new(config, length)?)),
        EnvConfig::RustVec { num, env } => Ok(Box::new(VectorWrapper::new(*num, env, length)?)),
        EnvConfig::RustMulti { envs } => Ok(Box::new(MultitaskWrapper::new(envs, length)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MapoxError;
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
        assert_eq!(
            parsed,
            EnvConfig::RustScouts(Box::new(ScoutsConfig::default()))
        );
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
            EnvConfig::RustFindReturn(Box::new(FindReturnConfig::default()))
        );
    }

    /// Same again for the snake, whose config carries no mapgen keys.
    #[test]
    fn a_snake_config_dump_parses() {
        let json = r#"{"env_type":"rust_snake","num_agents":4,"width":24,"height":24,
            "view_width":11,"view_height":11,"food_spawn_prob":0.002,
            "food_reward":1.0,"death_reward":-1.0}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed,
            EnvConfig::RustSnake(Box::new(SnakeConfig::default()))
        );
    }

    /// The vectorized shape: `num` copies of a plain env config.
    #[test]
    fn a_vec_config_parses() {
        let json = r#"{"env_type":"rust_vec","num":32,"env":{"env_type":"rust_scouts",
            "num_scouts":4,"num_harvesters":4,"num_treasures":12,"width":40,"height":40,
            "view_width":11,"view_height":13,"ui_height":2,"mapgen_threshold":0.3,
            "water_threshold":-0.45,"harvesters_move_every":6,
            "scout_reward":1.0,"harvester_reward":1.0}}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed,
            EnvConfig::RustVec {
                num: 32,
                env: Box::new(EnvConfig::RustScouts(Box::new(ScoutsConfig::default()))),
            }
        );
    }

    /// The multitask shape: named entries, each carrying its instance count
    /// and a plain env config.
    #[test]
    fn a_multi_config_parses() {
        let json = r#"{"env_type":"rust_multi","envs":[
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
                        env: Box::new(EnvConfig::RustScouts(Box::new(ScoutsConfig::default()))),
                    },
                    MultiEnvSpec {
                        name: "fr".into(),
                        num: 3,
                        env: Box::new(EnvConfig::RustFindReturn(Box::new(
                            FindReturnConfig::default(),
                        ))),
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
                    env: Box::new(EnvConfig::RustScouts(Box::new(ScoutsConfig {
                        num_scouts: 2,
                        num_harvesters: 2,
                        width: 12,
                        height: 12,
                        view_width: 11,
                        view_height: 13,
                        ..ScoutsConfig::default()
                    }))),
                },
                MultiEnvSpec {
                    name: "fr".into(),
                    num: 1,
                    env: Box::new(EnvConfig::RustFindReturn(Box::new(FindReturnConfig {
                        num_agents: 4,
                        width: 12,
                        height: 12,
                        view_width: 11,
                        // FindReturn appends a 2-row UI strip, so 11 + 2 = 13
                        // matches the scouts obs view above.
                        view_height: 11,
                        ..FindReturnConfig::default()
                    }))),
                },
            ],
        };

        let mut env = make(&config, 32).unwrap();
        assert_eq!(env.num_agents(), 12); // 2 * (2 + 2) + 4
        assert_eq!(env.num_tasks(), 2); // copies of an entry are one task
        assert_eq!(env.task_names(), vec!["scouts", "fr"]);

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
    fn make_multi_rejects_mismatched_observation_shapes() {
        let config = EnvConfig::RustMulti {
            envs: vec![
                MultiEnvSpec {
                    name: "scouts".into(),
                    num: 1,
                    env: Box::new(EnvConfig::RustScouts(Box::new(ScoutsConfig {
                        num_scouts: 1,
                        num_harvesters: 1,
                        view_width: 11,
                        view_height: 11,
                        ..ScoutsConfig::default()
                    }))),
                },
                MultiEnvSpec {
                    name: "fr".into(),
                    num: 1,
                    env: Box::new(EnvConfig::RustFindReturn(Box::new(FindReturnConfig {
                        num_agents: 2,
                        view_width: 11,
                        view_height: 11, // + the 2-row UI strip => 13
                        ..FindReturnConfig::default()
                    }))),
                },
            ],
        };

        assert!(matches!(
            make(&config, 32),
            Err(MapoxError::ObservationShapeMismatch {
                task,
                expected_task,
                width: 11,
                height: 13,
                expected_width: 11,
                expected_height: 11,
            }) if task == "fr" && expected_task == "scouts"
        ));
    }

    #[test]
    fn metrics_average_nested_copies_and_drain_every_task() {
        // A one-cell board makes every step a death, independently of the seed
        // or chosen action. Unequal copy counts expose incorrect group bounds.
        let task = |name: &str, num, death_reward| MultiEnvSpec {
            name: name.into(),
            num,
            env: Box::new(EnvConfig::RustSnake(Box::new(SnakeConfig {
                num_agents: 1,
                width: 1,
                height: 1,
                food_spawn_prob: 0.0,
                death_reward,
                ..SnakeConfig::default()
            }))),
        };
        let config = EnvConfig::RustVec {
            num: 2,
            env: Box::new(EnvConfig::RustMulti {
                envs: vec![task("snake", 2, -4.0), task("other", 3, -6.0)],
            }),
        };
        let mut env = make(&config, 32).unwrap();
        let zeros = serde_json::json!({
            "snake": {"reward": 0.0, "food_eaten": 0.0, "deaths": 0.0},
            "other": {"reward": 0.0, "food_eaten": 0.0, "deaths": 0.0},
        });
        assert_eq!(env.consume_metrics(), zeros);

        // Enjoy mode steps one copy only, but consume must drain and average
        // every configured copy, retaining inactive task keys as well.
        for (task, steps) in [(0, 1), (1, 3)] {
            env.set_enjoy_mode(Some(task));
            let mut buffers = TimeStepBuffers::new(&*env);
            env.reset(42, &mut buffers.view_mut());
            for _ in 0..steps {
                env.step(&[0], &mut buffers.view_mut());
            }
            env.reset(43, &mut buffers.view_mut());
        }
        assert_eq!(
            env.consume_metrics(),
            serde_json::json!({
                "snake": {"reward": -1.0, "food_eaten": 0.0, "deaths": 0.25},
                "other": {"reward": -3.0, "food_eaten": 0.0, "deaths": 0.5},
            }),
        );
        assert_eq!(env.consume_metrics(), zeros);

        env.set_enjoy_mode(None);
        let mut buffers = TimeStepBuffers::new(&*env);
        env.reset(44, &mut buffers.view_mut());
        env.step(&vec![0; env.num_agents()], &mut buffers.view_mut());
        assert_eq!(
            env.consume_metrics(),
            serde_json::json!({
                "snake": {"reward": -4.0, "food_eaten": 0.0, "deaths": 1.0},
                "other": {"reward": -6.0, "food_eaten": 0.0, "deaths": 1.0},
            }),
        );
        assert_eq!(env.consume_metrics(), zeros);
    }

    #[test]
    fn make_vec_builds_num_copies() {
        let scouts = Box::new(EnvConfig::RustScouts(Box::new(ScoutsConfig {
            num_scouts: 1,
            num_harvesters: 1,
            width: 12,
            height: 12,
            ..ScoutsConfig::default()
        })));

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
        assert!(env.task_names().is_empty());

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
    fn make_rejects_zero_vec_num() {
        assert!(
            make(
                &EnvConfig::RustVec {
                    num: 0,
                    env: Box::new(EnvConfig::RustScouts(Box::new(ScoutsConfig::default()))),
                },
                32
            )
            .is_err()
        );
    }
}
