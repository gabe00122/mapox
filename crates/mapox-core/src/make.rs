use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::{
        find_return::{FindReturn, FindReturnConfig},
        scouts::{Scouts, ScoutsConfig},
        snake::{Snake, SnakeConfig},
    },
    wrappers::{
        multitask::MultitaskWrapper, task_id_wrapper::TaskIdWrapper, vector::VectorWrapper,
    },
};

/// Config for anything the rust side can build: the concrete gridworlds,
/// vectorized copies of one env (`Vec`), or a multitask batch of them
/// (`Multi`). `Vec` and `Multi` embed another `EnvConfig`; the `Box` keeps
/// the enum's size finite and is the python side's recursion boundary.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "env_type", rename_all = "snake_case")]
pub enum EnvConfig {
    RustFindReturn(FindReturnConfig),
    RustScouts(ScoutsConfig),
    RustSnake(SnakeConfig),
    Vec {
        num: usize,
        env: Box<EnvConfig>,
    },
    Multi {
        envs: Vec<MultiEnvSpec>,
    },
}

/// One entry of a `Multi` config: `num` copies of `env`. `name` is what the
/// python side uses to address a single task out of the batch.
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
        EnvConfig::Vec { num, env } => {
            if *num == 0 {
                return Err("vec env num must be at least 1".into());
            }
            if *num == 1 {
                return make(env, length);
            }
            let envs = (0..*num)
                .map(|_| make(env, length))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Box::new(VectorWrapper::new(envs)))
        }
        EnvConfig::Multi { envs } => make_multi(envs, length),
    }
}

/// Build a `MultitaskWrapper` from the entries of a `Multi` config.
/// Every env instance becomes a direct (flat) child of the wrapper — the
/// wrapper's own parallel reset/step is the vectorizer — each wrapped in
/// its entry's TaskIdWrapper.
fn make_multi(specs: &[MultiEnvSpec], length: usize) -> Result<Box<dyn Environment>, String> {
    if specs.is_empty() {
        return Err("multitask config must list at least one env".into());
    }
    for (i, spec) in specs.iter().enumerate() {
        if spec.num == 0 {
            return Err(format!(
                "multitask env {i} '{}' has num 0; every env needs at least one instance",
                spec.name
            ));
        }
        // A nested `Multi` would sit inside this entry's TaskIdWrapper,
        // which clobbers the task ids the inner wrapper wrote.
        if contains_multi(&spec.env) {
            return Err(format!(
                "multitask env {i} '{}' nests another multitask config; its task ids would be clobbered",
                spec.name
            ));
        }
    }

    // One flat child per env instance: each instance is wrapped in the
    // entry's TaskIdWrapper before being handed to the MultitaskWrapper
    // (which adds a VocabWrapper on top). A VectorWrapper child would run
    // a second par_iter_mut inside the MultitaskWrapper's own loop.
    let mut envs: Vec<Box<dyn Environment>> = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        for _ in 0..spec.num {
            envs.push(Box::new(TaskIdWrapper::new(
                make(&spec.env, length)?,
                i as i32,
            )));
        }
    }

    // `MultitaskWrapper` serves every agent's obs from one shared
    // (width, height) shape taken from the first env (see the TODO on its
    // `observation_spec`), so mismatched view sizes must be rejected here
    // instead of panicking deep in buffer indexing.
    let first = envs[0].observation_spec();
    for (i, (env, entry)) in envs.iter().zip(specs.iter()).enumerate().skip(1) {
        let spec = env.observation_spec();
        if (spec.width, spec.height) != (first.width, first.height) {
            return Err(format!(
                "multitask env {i} '{}' has view {}x{}, but env 0 has view {}x{}; all envs must share one view size",
                entry.name, spec.width, spec.height, first.width, first.height
            ));
        }
    }

    Ok(Box::new(MultitaskWrapper::new(envs)))
}

/// Whether `config` builds a `MultitaskWrapper` anywhere in its subtree.
fn contains_multi(config: &EnvConfig) -> bool {
    match config {
        EnvConfig::Multi { .. } => true,
        EnvConfig::Vec { env, .. } => contains_multi(env),
        _ => false,
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
            EnvConfig::Vec {
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
            EnvConfig::Multi {
                envs: vec![
                    MultiEnvSpec {
                        name: "scouts".into(),
                        num: 2,
                        env: Box::new(EnvConfig::RustScouts(ScoutsConfig::default())),
                    },
                    MultiEnvSpec {
                        name: "fr".into(),
                        num: 3,
                        env: Box::new(EnvConfig::RustFindReturn(
                            FindReturnConfig::default()
                        )),
                    },
                ],
            }
        );
    }

    #[test]
    fn make_multi_groups_each_entry_and_assigns_task_ids_per_entry() {
        let config = EnvConfig::Multi {
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
    fn make_vec_builds_num_copies_and_num_one_is_plain() {
        let scouts = Box::new(EnvConfig::RustScouts(ScoutsConfig {
            num_scouts: 1,
            num_harvesters: 1,
            width: 12,
            height: 12,
            ..ScoutsConfig::default()
        }));

        let env = make(&EnvConfig::Vec { num: 3, env: scouts.clone() }, 32).unwrap();
        assert_eq!(env.num_agents(), 6);

        let env = make(&EnvConfig::Vec { num: 1, env: scouts }, 32).unwrap();
        assert_eq!(env.num_agents(), 2);
    }

    #[test]
    fn make_rejects_degenerate_configs() {
        assert!(make(&EnvConfig::Multi { envs: vec![] }, 32).is_err());
        assert!(make(
            &EnvConfig::Multi {
                envs: vec![MultiEnvSpec {
                    name: "scouts".into(),
                    num: 0,
                    env: Box::new(EnvConfig::RustScouts(ScoutsConfig::default())),
                }],
            },
            32
        )
        .is_err());
        assert!(make(
            &EnvConfig::Vec {
                num: 0,
                env: Box::new(EnvConfig::RustScouts(ScoutsConfig::default())),
            },
            32
        )
        .is_err());
    }

    #[test]
    fn make_rejects_nested_multitask() {
        let config = EnvConfig::Multi {
            envs: vec![MultiEnvSpec {
                name: "inner".into(),
                num: 1,
                env: Box::new(EnvConfig::Multi {
                    envs: vec![MultiEnvSpec {
                        name: "fr".into(),
                        num: 1,
                        env: Box::new(EnvConfig::RustFindReturn(
                            FindReturnConfig::default()
                        )),
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
        let config = EnvConfig::Multi {
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
