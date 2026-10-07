use serde::{Deserialize, Serialize};

#[cfg(not(target_arch = "wasm32"))]
use crate::wrappers::video::{VideoConfig, VideoWrapper};
use crate::{
    env::Environment,
    envs::{
        find_return::{FindReturn, FindReturnConfig},
        pacman::{Pacman, PacmanConfig},
        scouts::{Scouts, ScoutsConfig},
        snake::{Snake, SnakeConfig},
        survival::{Survival, SurvivalConfig},
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
    RustPacman(Box<PacmanConfig>),
    RustSurvival(Box<SurvivalConfig>),
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
        EnvConfig::RustPacman(config) => Ok(Box::new(Pacman::new(config, length))),
        EnvConfig::RustSurvival(config) => Ok(Box::new(Survival::new(config, length))),
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
        let json = r#"{"env_type":"rust_scouts","num_scouts":1,"num_harvesters":1,
            "num_treasures":12,"width":80,"height":70,"view_width":15,"view_height":15,
            "mapgen_threshold":0.07,"water_threshold":-0.45,
            "harvesters_move_every":6,"scout_reward":1.0,"harvester_reward":1.0}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(parsed, EnvConfig::RustScouts(Box::default()));
    }

    /// And the same for the env that was here first.
    #[test]
    fn a_find_return_config_dump_parses() {
        let json = r#"{"env_type":"rust_find_return","num_agents":8,"num_flags":1,
            "width":80,"height":70,"view_width":15,"view_height":15,"mapgen_threshold":0.07,
            "water_threshold":-0.45,"digging_timeout":5,"preparation_steps":256,
            "treasure_reward":1.0,"pipes_enabled":true}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(parsed, EnvConfig::RustFindReturn(Box::default()));
    }

    /// Same again for the snake, whose config carries no mapgen keys.
    #[test]
    fn a_snake_config_dump_parses() {
        let json = r#"{"env_type":"rust_snake","num_agents":16,"width":80,"height":70,
            "view_width":15,"view_height":15,"food_spawn_prob":0.0001,
            "food_reward":0.01,"death_reward":0.0}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(parsed, EnvConfig::RustSnake(Box::default()));
    }

    /// And for pac-man, which plays on a fixed maze with no size keys.
    #[test]
    fn a_pacman_config_dump_parses() {
        let json = r#"{"env_type":"rust_pacman","view_width":15,"view_height":15,
            "randomize_starting_position":false,"min_start_timeout":0,
            "max_start_timeout":49,"frightened_time":35,"max_mode_changes":6,
            "scatter_mode_length":70,"chase_mode_length":140,"pellet_reward":1.0,
            "ghost_reward":1.0,"death_reward":0.0,"clear_reward":0.0}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(parsed, EnvConfig::RustPacman(Box::default()));
    }

    /// And for survival, `RustSurvivalConfig().model_dump_json()` verbatim.
    #[test]
    fn a_survival_config_dump_parses() {
        let json = r#"{"env_type":"rust_survival","num_agents":8,"width":80,"height":70,
            "view_width":15,"view_height":15,"water_fraction":0.1,"rock_fraction":0.15,
            "forest_fraction":0.35,"scrub_fraction":0.25,"start_hunger":100,
            "start_health":150,"hunger_interval":2,"starve_damage":1,"regen_threshold":100,
            "regen_interval":4,"berry_food":20,"cooked_berry_food":50,
            "carrot_food":35,"bush_regrow_steps":200,"chop_steps":6,"harvest_steps":3,"dig_carrot_steps":1,
            "fire_burn_steps":200,"fire_low_steps":50,
            "fire_light_radius":4,"day_length":200,"night_length":100,
            "night_vision_radius":2,"dusk_length":50,"num_spider_eggs":4,"spider_damage":15,
            "spider_hunt_radius":12}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(parsed, EnvConfig::RustSurvival(Box::default()));
    }

    /// The vectorized shape: `num` copies of a plain env config.
    #[test]
    fn a_vec_config_parses() {
        let json = r#"{"env_type":"rust_vec","num":32,"env":{"env_type":"rust_scouts",
            "num_scouts":1,"num_harvesters":1,"num_treasures":12,"width":80,"height":70,
            "view_width":15,"view_height":15,"mapgen_threshold":0.07,
            "water_threshold":-0.45,"harvesters_move_every":6,
            "scout_reward":1.0,"harvester_reward":1.0}}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed,
            EnvConfig::RustVec {
                num: 32,
                env: Box::new(EnvConfig::RustScouts(Box::default())),
            }
        );
    }

    /// The multitask shape: named entries, each carrying its instance count
    /// and a plain env config.
    #[test]
    fn a_multi_config_parses() {
        let json = r#"{"env_type":"rust_multi","envs":[
            {"name":"scouts","num":2,"env":{"env_type":"rust_scouts","num_scouts":1,
                "num_harvesters":1,"num_treasures":12,"width":80,"height":70,
                "view_width":15,"view_height":15,"mapgen_threshold":0.07,
                "water_threshold":-0.45,"harvesters_move_every":6,
                "scout_reward":1.0,"harvester_reward":1.0}},
            {"name":"fr","num":3,"env":{"env_type":"rust_find_return","num_agents":8,
                "num_flags":1,"width":80,"height":70,"view_width":15,"view_height":15,
                "mapgen_threshold":0.07,"water_threshold":-0.45,"digging_timeout":5,
                "preparation_steps":256,"treasure_reward":1.0,"pipes_enabled":true}}
        ]}"#;

        let parsed: EnvConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed,
            EnvConfig::RustMulti {
                envs: vec![
                    MultiEnvSpec {
                        name: "scouts".into(),
                        num: 2,
                        env: Box::new(EnvConfig::RustScouts(Box::default())),
                    },
                    MultiEnvSpec {
                        name: "fr".into(),
                        num: 3,
                        env: Box::new(EnvConfig::RustFindReturn(Box::default())),
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
                        view_height: 11,
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
                        view_height: 13,
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
                // observation heights: view_height + the UI band
                height: 15,
                expected_width: 11,
                expected_height: 13,
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
            assert_eq!(env.enjoy_task(), Some(task));
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
        assert_eq!(env.enjoy_task(), None);
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
                    env: Box::new(EnvConfig::RustScouts(Box::default())),
                },
                32
            )
            .is_err()
        );
    }
}
