pub mod common;
pub mod find_return;
pub mod hide_seek;
pub mod maze;
pub mod pacman;
pub mod scouts;
pub mod survival;

#[cfg(test)]
mod tests {
    use crate::{
        env::Environment,
        envs::{
            find_return::{FindReturn, FindReturnConfig},
            hide_seek::{HideSeek, HideSeekConfig},
            maze::{Maze, MazeConfig},
            pacman::{Pacman, PacmanConfig},
            scouts::{Scouts, ScoutsConfig},
            survival::{Survival, config::SurvivalConfig},
        },
        policy::{Policy, RandomPolicy},
        timestep::TimeStepBuffers,
    };

    const LENGTH: usize = 96;

    /// Every timestep of one masked-random episode from `reset(seed)`.
    fn episode(env: &mut dyn Environment, seed: u64) -> Vec<TimeStepBuffers> {
        let num_agents = env.num_agents();
        let mut buffers = TimeStepBuffers::new(env);
        let mut actions = vec![0; num_agents];
        let mut policy = RandomPolicy::new();
        policy.reset(num_agents, seed).unwrap();

        env.reset(seed, &mut buffers.view_mut());
        let mut trace = vec![buffers.rows(0, num_agents)];
        for _ in 0..LENGTH {
            policy.act(&buffers.view(), &mut actions).unwrap();
            env.step(&actions, &mut buffers.view_mut());
            trace.push(buffers.rows(0, num_agents));
        }
        trace
    }

    /// The seed is the whole of an episode's randomness: an env that has
    /// already played must replay `reset(seed)` exactly as a fresh one does.
    /// The maps are crowded so that the agents' turn order decides who gets
    /// a contested cell.
    #[test]
    fn a_reset_does_not_remember_the_last_episode() {
        type Make = Box<dyn Fn() -> Box<dyn Environment>>;
        let envs: Vec<(&str, Make)> = vec![
            (
                "find_return",
                Box::new(|| {
                    Box::new(FindReturn::new(
                        &FindReturnConfig {
                            num_agents: 30,
                            width: 12,
                            height: 12,
                            preparation_steps: 8,
                            ..Default::default()
                        },
                        LENGTH,
                    ))
                }),
            ),
            (
                "scouts",
                Box::new(|| {
                    Box::new(Scouts::new(
                        &ScoutsConfig {
                            num_scouts: 6,
                            num_harvesters: 6,
                            num_treasures: 30,
                            width: 14,
                            height: 14,
                            harvesters_move_timeout: 1,
                            ..Default::default()
                        },
                        LENGTH,
                    ))
                }),
            ),
            (
                "hide_seek",
                Box::new(|| {
                    Box::new(HideSeek::new(
                        &HideSeekConfig {
                            num_hiders: 6,
                            num_seekers: 6,
                            width: 14,
                            height: 14,
                            preparation_steps: 8,
                            ..Default::default()
                        },
                        LENGTH,
                    ))
                }),
            ),
            (
                "maze",
                Box::new(|| Box::new(Maze::new(&MazeConfig::default(), LENGTH))),
            ),
            (
                "pacman",
                Box::new(|| Box::new(Pacman::new(&PacmanConfig::default(), LENGTH))),
            ),
            (
                "survival",
                Box::new(|| Box::new(Survival::new(&SurvivalConfig::default(), LENGTH))),
            ),
        ];

        for (name, make) in envs {
            for seed in 0..4 {
                let fresh = episode(make().as_mut(), seed);

                let mut used = make();
                episode(used.as_mut(), seed + 100);
                let replayed = episode(used.as_mut(), seed);

                for (step, (fresh, replayed)) in fresh.iter().zip(&replayed).enumerate() {
                    assert_eq!(
                        fresh.differing_fields(replayed),
                        Vec::<&str>::new(),
                        "{name} seed {seed} diverged at step {step}",
                    );
                }
            }
        }
    }
}
