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

/// Bushes don't block, ripe or bare: an agent walks onto one and hides it,
/// and it shows again once the agent steps off.
#[test]
fn agents_walk_through_bushes() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        ..Default::default()
    });
    let start = center(&env);
    env.set_ground(start + UP, TileBerryBush);
    env.set_ground(start + UP * 2, TileBush);
    spawn_facing(&mut env, start, 0);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[MoveUp]);
    assert_eq!(env.state.agents[0].position, start + UP);
    step(&mut env, &mut buffers, &[MoveUp]);
    assert_eq!(env.state.agents[0].position, start + UP * 2);
    assert_eq!(env.state.map[(start + UP).idx()], TileBerryBush);
    step(&mut env, &mut buffers, &[MoveUp]);
    assert_eq!(env.state.map[(start + UP * 2).idx()], TileBush);
}

/// Using empty hands on tall grass harvests it: like chopping, the agent can
/// only wait until `harvest_steps` are done, and then a bundle of grass lies
/// where it grew. Grab doesn't harvest, and holding something rules it out.
#[test]
fn tall_grass_is_harvested_slowly_by_hand() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        harvest_steps: 3,
        ..Default::default()
    });
    let start = center(&env);
    let grass = start + UP;
    env.set_ground(grass, TileTallGrass);
    spawn_facing(&mut env, start, 0);
    agent(&mut env, 0).backpack = Some(Item::Stone);
    assert!(!legal(&observe(&env), 0, Grab), "grab doesn't harvest");
    agent(&mut env, 0).hands = Some(Item::Stick);
    assert!(
        !legal(&observe(&env), 0, Use),
        "harvesting needs empty hands"
    );
    agent(&mut env, 0).hands = None;

    let mut buffers = TimeStepBuffers::new(&env);
    assert!(legal(&observe(&env), 0, Use));
    step(&mut env, &mut buffers, &[Use]);
    step(&mut env, &mut buffers, &[MoveDown]);
    assert_eq!(env.state.agents[0].position, start, "busy harvesting");
    assert_eq!(env.state.map[grass.idx()], TileTallGrass);

    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.map[grass.idx()], ItemGrass);
    assert_eq!(achieved(&env, Achievement::HarvestGrass), 1.0);
    assert_eq!(env.state.agents[0].hands, None);

    step(&mut env, &mut buffers, &[Grab]);
    assert_eq!(env.state.agents[0].hands, Some(Item::Grass));
    assert_eq!(achieved(&env, Achievement::CollectGrass), 1.0);
}

/// Tall grass doesn't block: an agent walks through it, hiding it, and
/// can't harvest the grass it stands in.
#[test]
fn agents_walk_through_tall_grass() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        ..Default::default()
    });
    let start = center(&env);
    env.set_ground(start + UP, TileTallGrass);
    spawn_facing(&mut env, start, 0);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[MoveUp]);
    assert_eq!(env.state.agents[0].position, start + UP);
    assert!(!legal(&buffers, 0, Use), "the grass is underfoot");
    step(&mut env, &mut buffers, &[MoveUp]);
    assert_eq!(env.state.map[(start + UP).idx()], TileTallGrass);
}

/// Carrots start buried: using empty hands on one digs it up, leaving the
/// carrot lying there to grab. Grab can't reach a buried carrot, and holding
/// something rules digging out. A carrot taken is gone for good: nothing
/// grows back where it lay.
#[test]
fn buried_carrots_are_dug_up_and_never_regrow() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        bush_regrow_steps: 1,
        ..Default::default()
    });
    let start = center(&env);
    let carrot = start + UP;
    env.set_ground(carrot, TileBuriedCarrot);
    spawn_facing(&mut env, start, 0);
    assert!(!legal(&observe(&env), 0, Grab), "the carrot is buried");
    agent(&mut env, 0).hands = Some(Item::Stick);
    assert!(!legal(&observe(&env), 0, Use), "digging needs empty hands");
    agent(&mut env, 0).hands = None;

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Use]);
    assert_eq!(env.state.map[carrot.idx()], ItemCarrot);
    assert_eq!(achieved(&env, Achievement::DigCarrot), 1.0);
    assert!(env.state.agents[0].work.is_none(), "one use digs it up");

    step(&mut env, &mut buffers, &[Grab]);
    assert_eq!(env.state.agents[0].hands, Some(Item::Carrot));
    assert_eq!(achieved(&env, Achievement::CollectCarrot), 1.0);
    for _ in 0..10 {
        step(&mut env, &mut buffers, &[Noop]);
    }
    assert_eq!(env.state.map[carrot.idx()], TileEmpty);
}
