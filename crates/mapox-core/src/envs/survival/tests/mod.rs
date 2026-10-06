//! Tests by system, mirroring the modules they cover; the helpers here set up
//! worlds by hand.

mod actions;
mod clock;
mod fire;
mod obs;
mod plants;
mod spiders;
mod survivor;
mod terrain;

use ndarray::s;

use super::{
    MAX_STAT, Survival, SurvivalConfig,
    actions::SurvivalAction::{self, *},
    fire::Fire,
    items::Item,
    metrics::Achievement,
    obs::CLOCK_COL,
    survivor::Survivor,
    tiles::{
        AROUND, DIRECTIONS,
        SurvivalObs::{self, *},
    },
};
use crate::{
    env::Environment,
    envs::common::{Position, vocab_enum::VocabEnum},
    timestep::TimeStepBuffers,
    vocab::VocabId,
};

/// A bare walled floor with nothing scattered on it and no agents, for
/// placing things by hand.
fn empty_env_with(config: SurvivalConfig) -> Survival {
    let mut env = Survival::new(
        &SurvivalConfig {
            width: 21,
            height: 21,
            ..config
        },
        512,
    );

    env.state.base_map.fill(TileWall);
    env.state
        .base_map
        .slice_mut(s![
            env.pad_width as usize..(env.width - env.pad_width) as usize,
            env.pad_height as usize..(env.height - env.pad_height) as usize,
        ])
        .fill(TileEmpty);
    env.state.map.assign(&env.state.base_map);

    env
}

fn center(env: &Survival) -> Position {
    Position::new(env.width / 2, env.height / 2)
}

/// Puts an agent on the map facing `dir`; call in agent-id order, and as
/// many times as the config has agents.
fn spawn_facing(env: &mut Survival, position: Position, dir: u8) {
    let agent = Survivor::spawn(position, dir, &env.config);
    env.state.map[position.idx()] = agent.tile();
    env.state.agents.push(agent);
}

fn agent(env: &mut Survival, agent_id: usize) -> &mut Survivor {
    &mut env.state.agents[agent_id]
}

fn step(env: &mut Survival, buffers: &mut TimeStepBuffers, actions: &[SurvivalAction]) {
    let actions: Vec<VocabId> = actions.iter().map(|&a| a.into()).collect();
    env.step(&actions, &mut buffers.view_mut());
}

/// Encodes the current state without stepping, for checking a setup.
fn observe(env: &Survival) -> TimeStepBuffers {
    let mut buffers = TimeStepBuffers::new(env);
    env.encode_observations(&mut buffers.view_mut());
    env.encode_action_mask(&mut buffers.view_mut());
    buffers
}

fn legal(buffers: &TimeStepBuffers, agent_id: usize, action: SurvivalAction) -> bool {
    buffers.action_mask[[agent_id, action as usize]]
}

fn id(tile: SurvivalObs) -> VocabId {
    tile.into()
}

fn achieved(env: &Survival, achievement: Achievement) -> f64 {
    env.metrics.achievements[achievement as usize]
}

const UP: Position = DIRECTIONS[0];
const RIGHT: Position = DIRECTIONS[1];
const DOWN: Position = DIRECTIONS[2];
const LEFT: Position = DIRECTIONS[3];

/// Sets a fire burning on `position`, lighting the ground around it.
fn light_fire(env: &mut Survival, position: Position) {
    env.set_ground(position, TileFire);
    env.state.fires.push(Fire {
        position,
        burn_left: 100,
    });
    env.light_up();
}

/// An episode follows from its seed alone, whatever the env ran before it.
#[test]
fn reset_is_reproducible_from_the_seed() {
    let config = SurvivalConfig::default();
    let actions: Vec<SurvivalAction> =
        [MoveUp, MoveLeft, Grab, MoveDown, Use, Noop, Eat, MoveRight]
            .into_iter()
            .cycle()
            .take(config.num_agents)
            .collect();
    let run = |env: &mut Survival| {
        let mut buffers = TimeStepBuffers::new(env);
        env.reset(7, &mut buffers.view_mut());
        for _ in 0..20 {
            step(env, &mut buffers, &actions);
        }
        (env.state.agent_order.clone(), buffers.obs.clone())
    };

    let mut fresh = Survival::new(&config, 512);
    let mut used = Survival::new(&config, 512);
    let mut buffers = TimeStepBuffers::new(&used);
    used.reset(1, &mut buffers.view_mut());
    for _ in 0..5 {
        step(&mut used, &mut buffers, &actions);
    }

    assert_eq!(run(&mut fresh), run(&mut used));
}
