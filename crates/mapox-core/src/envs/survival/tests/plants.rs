use super::*;

/// Grabbing at a ripe bush picks a berry and leaves it bare, and a bare
/// bush can't be picked again until it regrows.
#[test]
fn bushes_give_berries_and_regrow() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        bush_regrow_steps: 3,
        ..Default::default()
    });
    let start = center(&env);
    let bush = start + UP;
    env.set_ground(bush, TileBerryBush);
    spawn_facing(&mut env, start, 0);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Grab]);
    assert_eq!(env.state.agents[0].hands, Some(Item::Berry));
    assert_eq!(env.state.map[bush.idx()], TileBush);
    assert_eq!(achieved(&env, Achievement::CollectBerry), 1.0);

    step(&mut env, &mut buffers, &[Swap]);
    assert!(!legal(&buffers, 0, Grab), "the bush is bare");
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.map[bush.idx()], TileBush);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.map[bush.idx()], TileBerryBush);
    assert!(legal(&buffers, 0, Grab));
}
