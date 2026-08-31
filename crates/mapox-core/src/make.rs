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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "env_type", rename_all = "snake_case")]
pub enum EnvConfig {
    RustFindReturn(FindReturnConfig),
    RustScouts(ScoutsConfig),
    RustSnake(SnakeConfig),
}

pub fn make(config: &EnvConfig, length: usize) -> Box<dyn Environment> {
    match config {
        EnvConfig::RustFindReturn(config) => Box::new(FindReturn::new(config, length)),
        EnvConfig::RustScouts(config) => Box::new(Scouts::new(config, length)),
        EnvConfig::RustSnake(config) => Box::new(Snake::new(config, length)),
    }
}

pub fn make_vec(config: &EnvConfig, length: usize, num_envs: usize) -> Box<dyn Environment> {
    if num_envs == 1 {
        return make(config, length);
    }

    let envs: Vec<Box<dyn Environment>> = (0..num_envs).map(|_| make(config, length)).collect();
    Box::new(VectorWrapper::new(envs))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MultitaskEnvSpec {
    pub config: EnvConfig,
    pub count: usize,
}

/// Build a `MultitaskWrapper` from a list of {config, count} entries.
/// Every env instance becomes a direct (flat) child of the wrapper — the
/// wrapper's own parallel reset/step is the vectorizer — each wrapped in
/// its entry's TaskIdWrapper.
pub fn make_multitask(
    specs: &[MultitaskEnvSpec],
    length: usize,
) -> Result<Box<dyn Environment>, String> {
    if specs.is_empty() {
        return Err("multitask config must list at least one env".into());
    }
    for (i, spec) in specs.iter().enumerate() {
        if spec.count == 0 {
            return Err(format!(
                "multitask env {i} has count 0; every env needs at least one instance"
            ));
        }
    }

    // One flat child per env instance: each instance is wrapped in the
    // entry's TaskIdWrapper before being handed to the MultitaskWrapper
    // (which adds a VocabWrapper on top). A VectorWrapper child would run
    // a second par_iter_mut inside the MultitaskWrapper's own loop.
    let mut envs: Vec<Box<dyn Environment>> = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        for _ in 0..spec.count {
            envs.push(Box::new(TaskIdWrapper::new(
                make(&spec.config, length),
                i as i32,
            )));
        }
    }

    // `MultitaskWrapper` serves every agent's obs from one shared
    // (width, height) shape taken from the first env (see the TODO on its
    // `observation_spec`), so mismatched view sizes must be rejected here
    // instead of panicking deep in buffer indexing.
    let first = envs[0].observation_spec();
    for (i, env) in envs.iter().enumerate().skip(1) {
        let spec = env.observation_spec();
        if (spec.width, spec.height) != (first.width, first.height) {
            return Err(format!(
                "multitask env {i} has view {}x{}, but env 0 has view {}x{}; all envs must share one view size",
                spec.width, spec.height, first.width, first.height
            ));
        }
    }

    Ok(Box::new(MultitaskWrapper::new(envs)))
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

    /// The multitask shape: the same config dumps as the single-env tests,
    /// each entry carrying its instance count.
    #[test]
    fn a_multitask_spec_list_parses() {
        let json = r#"[
            {"config": {"env_type":"rust_scouts","num_scouts":4,"num_harvesters":4,
                "num_treasures":12,"width":40,"height":40,"view_width":11,"view_height":13,
                "ui_height":2,"mapgen_threshold":0.3,"water_threshold":-0.45,
                "harvesters_move_every":6,"scout_reward":1.0,"harvester_reward":1.0}, "count": 2},
            {"config": {"env_type":"rust_find_return","num_agents":8,"num_flags":1,
                "width":40,"height":40,"view_width":11,"view_height":11,"mapgen_threshold":0.3,
                "water_threshold":-0.45,"digging_timeout":5,"preparation_steps":256,
                "treasure_reward":1.0}, "count": 3}
        ]"#;

        let parsed: Vec<MultitaskEnvSpec> = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed,
            vec![
                MultitaskEnvSpec {
                    config: EnvConfig::RustScouts(ScoutsConfig::default()),
                    count: 2,
                },
                MultitaskEnvSpec {
                    config: EnvConfig::RustFindReturn(FindReturnConfig::default()),
                    count: 3,
                },
            ]
        );
    }

    #[test]
    fn make_multitask_groups_each_entry_and_assigns_task_ids_per_entry() {
        let specs = vec![
            MultitaskEnvSpec {
                config: EnvConfig::RustScouts(ScoutsConfig {
                    num_scouts: 2,
                    num_harvesters: 2,
                    width: 12,
                    height: 12,
                    view_width: 11,
                    view_height: 13,
                    ..ScoutsConfig::default()
                }),
                count: 2,
            },
            MultitaskEnvSpec {
                config: EnvConfig::RustFindReturn(FindReturnConfig {
                    num_agents: 4,
                    width: 12,
                    height: 12,
                    view_width: 11,
                    // FindReturn appends a 2-row UI strip, so 11 + 2 = 13
                    // matches the scouts obs view above.
                    view_height: 11,
                    ..FindReturnConfig::default()
                }),
                count: 1,
            },
        ];

        let mut env = make_multitask(&specs, 32).unwrap();
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
    fn make_multitask_rejects_degenerate_specs() {
        assert!(make_multitask(&[], 32).is_err());
        assert!(
            make_multitask(
                &[MultitaskEnvSpec {
                    config: EnvConfig::RustScouts(ScoutsConfig::default()),
                    count: 0,
                }],
                32
            )
            .is_err()
        );
    }

    /// The obs view is what `MultitaskWrapper` serves from one shared shape,
    /// not the config field: FindReturn's actual view is `view_height + 2`
    /// (UI strip), so its default 11 actually matches scouts' default 13.
    /// A config view of 13 (→ obs 15) does not.
    #[test]
    fn make_multitask_rejects_mismatched_view_sizes() {
        let specs = vec![
            MultitaskEnvSpec {
                config: EnvConfig::RustScouts(ScoutsConfig::default()),
                count: 1,
            },
            MultitaskEnvSpec {
                config: EnvConfig::RustFindReturn(FindReturnConfig {
                    view_height: 13,
                    ..FindReturnConfig::default()
                }),
                count: 1,
            },
        ];

        assert!(matches!(
            make_multitask(&specs, 32),
            Err(err)
                if err.contains("multitask env 1 has view 11x15, but env 0 has view 11x13")
        ));
    }
}
