use super::*;

/// Winter sets in on its first step: every bush dies, ripe or picked, tall
/// grass dies back to bare ground, and water freezes into ice an agent can
/// walk on. Buried carrots and items lying about keep, nothing picked grows
/// back, and the band shows the snowflake from then on.
#[test]
fn winter_kills_the_plants_and_freezes_the_water() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        // winter from step 2 of the helper's 512
        winter_length: 510,
        bush_regrow_steps: 1,
        ..Default::default()
    });
    let start = center(&env);
    let tiles = [
        (start + LEFT, TileBerryBush),
        (start + LEFT * 2, TileBush),
        (start + DOWN, TileTallGrass),
        (start + RIGHT, TileWater),
        (start + UP, TileBuriedCarrot),
        (start + UP * 2, ItemBerry),
    ];
    for (position, tile) in tiles {
        env.set_ground(position, tile);
    }
    // a bush picked bare just before, that would fruit again in winter
    env.pick_bush(start + LEFT * 3);
    spawn_facing(&mut env, start, 0);
    let slots_row = env.config.view_height as usize;
    let season = move |buffers: &TimeStepBuffers| {
        SurvivalObs::from_id(buffers.obs[[0, SEASON_COL, slots_row, 0]])
    };

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(season(&buffers), UI, "still summer");
    for (position, tile) in tiles {
        assert_eq!(env.state.map[position.idx()], tile);
    }

    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(season(&buffers), UiWinter);
    for (position, tile) in [
        (start + LEFT, TileDeadBush),
        (start + LEFT * 2, TileDeadBush),
        (start + LEFT * 3, TileDeadBush),
        (start + DOWN, TileEmpty),
        (start + RIGHT, TileIce),
        (start + UP, TileBuriedCarrot),
        (start + UP * 2, ItemBerry),
    ] {
        assert_eq!(env.state.map[position.idx()], tile);
    }
    assert!(env.state.regrowing.is_empty());

    step(&mut env, &mut buffers, &[MoveRight]);
    assert_eq!(env.state.agents[0].position, start + RIGHT, "onto the ice");
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.map[(start + LEFT * 3).idx()], TileDeadBush);
}

/// An episode shorter than winter is winter from reset.
#[test]
fn an_episode_all_winter_starts_frozen() {
    let config = SurvivalConfig {
        winter_length: 1000,
        ..Default::default()
    };
    let mut env = Survival::new(&config, 512);
    let mut buffers = TimeStepBuffers::new(&env);
    env.reset(0, &mut buffers.view_mut());

    let map = &env.state.base_map;
    for tile in [TileWater, TileBerryBush, TileBush, TileTallGrass] {
        assert!(!map.iter().any(|&t| t == tile), "{tile:?} left in winter");
    }
    assert!(map.iter().any(|&t| t == TileIce));
    assert!(map.iter().any(|&t| t == TileDeadBush));
}

/// Through summer temperature holds wherever the agent goes.
#[test]
fn summer_is_warm() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        winter_length: 0,
        start_temperature: 100,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);

    let mut buffers = TimeStepBuffers::new(&env);
    for _ in 0..10 {
        step(&mut env, &mut buffers, &[Noop]);
    }
    assert_eq!(env.state.agents[0].temperature, 100);
}

/// In winter, off firelit ground, temperature drops one every
/// `chill_interval` steps; at zero, health drains by `freeze_damage` a step
/// and stops growing back. Firelit ground warms by `fire_warmth` a step, up
/// to the cap.
#[test]
fn winter_cold_chills_and_fires_warm() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        // winter throughout
        winter_length: 512,
        chill_interval: 1,
        freeze_damage: 2,
        fire_warmth: 5,
        // no hunger to muddle the sums, and regen every step if it could
        hunger_interval: 1000,
        start_hunger: MAX_STAT,
        regen_interval: 1,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    agent(&mut env, 0).temperature = 2;
    agent(&mut env, 0).health = 100;

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.agents[0].temperature, 1);
    assert_eq!(env.state.agents[0].health, 101, "still warm enough to heal");
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.agents[0].temperature, 0);
    assert_eq!(env.state.agents[0].health, 99);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.agents[0].health, 97);

    light_fire(&mut env, start + UP * 2);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.agents[0].temperature, 5);
    assert_eq!(env.state.agents[0].health, 98, "thawed, it heals again");

    agent(&mut env, 0).temperature = MAX_STAT - 1;
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.agents[0].temperature, MAX_STAT);
}
