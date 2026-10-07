use super::*;

/// A campfire put on open ground is lit there, and goes out after
/// `fire_burn_steps`.
#[test]
fn a_campfire_is_set_down_lit_and_burns_out() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        fire_burn_steps: 3,
        fire_low_steps: 0,
        ..Default::default()
    });
    let start = center(&env);
    let fire = start + UP;
    spawn_facing(&mut env, start, 0);
    agent(&mut env, 0).hands = Some(Item::Campfire);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Put]);
    assert_eq!(env.state.map[fire.idx()], TileFire);
    assert_eq!(env.state.agents[0].hands, None);
    assert_eq!(achieved(&env, Achievement::PlaceFire), 1.0);

    step(&mut env, &mut buffers, &[Noop]);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.map[fire.idx()], TileFire);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.map[fire.idx()], TileEmpty);
}

/// Putting a raw berry or carrot to the fire in front cooks it in hand at
/// once. Cooked food doesn't cook again.
#[test]
fn food_put_to_a_fire_cooks_at_once() {
    for (raw, cooked, achievement) in [
        (Item::Berry, Item::CookedBerry, Achievement::CookBerry),
        (Item::Carrot, Item::CookedCarrot, Achievement::CookCarrot),
    ] {
        let mut env = empty_env_with(SurvivalConfig {
            num_agents: 1,
            ..Default::default()
        });
        let start = center(&env);
        spawn_facing(&mut env, start, 0);
        agent(&mut env, 0).hands = Some(raw);

        light_fire(&mut env, start + UP);
        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Put]);

        assert_eq!(env.state.agents[0].hands, Some(cooked));
        assert_eq!(achieved(&env, achievement), 1.0);
        assert!(!legal(&buffers, 0, Put), "cooked food doesn't cook again");
    }
}

/// A fire burns low for its last `fire_low_steps`, and wood put on a low
/// fire stokes it back up, no further than full. Wood can't go on a fire
/// burning high, and a berry cooks on a low fire as on any other.
#[test]
fn a_fire_burns_low_and_wood_stokes_it() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        fire_burn_steps: 4,
        fire_low_steps: 2,
        ..Default::default()
    });
    let start = center(&env);
    let fire = start + UP;
    spawn_facing(&mut env, start, 0);
    agent(&mut env, 0).hands = Some(Item::Campfire);

    // shows for four steps, the last two low
    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Put]);
    agent(&mut env, 0).hands = Some(Item::Wood);
    let mut burning = vec![env.state.map[fire.idx()]];
    for _ in 0..3 {
        let low = burning.last() == Some(&TileFireLow);
        assert_eq!(
            legal(&observe(&env), 0, Put),
            low,
            "wood goes only on a low fire"
        );
        step(&mut env, &mut buffers, &[Noop]);
        burning.push(env.state.map[fire.idx()]);
    }
    assert_eq!(burning, [TileFire, TileFire, TileFireLow, TileFireLow]);
    assert!(legal(&buffers, 0, Put));

    step(&mut env, &mut buffers, &[Put]);
    assert_eq!(env.state.map[fire.idx()], TileFire);
    assert_eq!(env.state.fires[0].burn_left, 3);
    assert_eq!(env.state.agents[0].hands, None);
    assert_eq!(achieved(&env, Achievement::RefuelFire), 1.0);

    step(&mut env, &mut buffers, &[Noop]);
    step(&mut env, &mut buffers, &[Noop]);
    agent(&mut env, 0).hands = Some(Item::Berry);
    assert_eq!(env.state.map[fire.idx()], TileFireLow);
    assert!(legal(&observe(&env), 0, Put), "a low fire still cooks");
}

/// Sticks and grass stoke a low fire too, by `stick_fuel` and `grass_fuel`
/// steps, less than wood's `wood_fuel`; other items don't burn.
#[test]
fn sticks_and_grass_stoke_less_than_wood() {
    let config = SurvivalConfig {
        num_agents: 1,
        fire_burn_steps: 300,
        fire_low_steps: 50,
        ..Default::default()
    };
    let mut gains = Vec::new();
    for fuel in [Item::Wood, Item::Stick, Item::Grass] {
        let mut env = empty_env_with(config.clone());
        let start = center(&env);
        spawn_facing(&mut env, start, 0);
        light_fire(&mut env, start + UP);
        env.state.fires[0].burn_left = 10;
        env.set_ground(start + UP, TileFireLow);
        agent(&mut env, 0).hands = Some(fuel);

        let mut buffers = TimeStepBuffers::new(&env);
        step(&mut env, &mut buffers, &[Put]);
        assert_eq!(env.state.agents[0].hands, None);
        assert_eq!(achieved(&env, Achievement::RefuelFire), 1.0);
        // the fire burns a step after it is stoked
        gains.push(env.state.fires[0].burn_left + 1 - 10);
    }
    assert_eq!(
        gains,
        [config.wood_fuel, config.stick_fuel, config.grass_fuel]
    );
    assert!(gains[0] > gains[1] && gains[1] > gains[2]);

    let mut env = empty_env_with(config);
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    light_fire(&mut env, start + UP);
    env.state.fires[0].burn_left = 10;
    env.set_ground(start + UP, TileFireLow);
    agent(&mut env, 0).hands = Some(Item::Stone);
    assert!(!legal(&observe(&env), 0, Put), "a stone doesn't burn");
}

/// A torch in hand lights `torch_light_radius` around its holder, following
/// it as it moves, and is gone after `torch_burn_steps` held. Its light
/// gives no warmth.
#[test]
fn a_held_torch_lights_the_way_and_burns_out() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        torch_burn_steps: 3,
        torch_light_radius: 2,
        // all winter, so unwarmed ground chills
        winter_length: 512,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    agent(&mut env, 0).hands = Some(Item::Torch { burnt: 0 });

    let mut buffers = TimeStepBuffers::new(&env);
    let temperature = env.state.agents[0].temperature;
    step(&mut env, &mut buffers, &[MoveUp]);
    let here = start + UP;
    assert!(env.state.lit[(here + UP * 2).idx()]);
    assert!(!env.state.lit[(here + UP * 3).idx()], "out of reach");
    assert!(!env.state.lit[(start + DOWN * 2).idx()], "left behind");
    assert_eq!(env.state.agents[0].hands, Some(Item::Torch { burnt: 1 }));
    assert_eq!(
        env.state.agents[0].temperature,
        temperature - 1,
        "no warmth"
    );

    step(&mut env, &mut buffers, &[Noop]);
    step(&mut env, &mut buffers, &[Noop]);
    assert!(env.state.lit[here.idx()], "lit for its last step");
    assert_eq!(env.state.agents[0].hands, None);
    step(&mut env, &mut buffers, &[Noop]);
    assert!(!env.state.lit[here.idx()]);
}

/// A torch only lights and burns in hand: in the backpack or on the ground
/// it keeps its wear, and picked up again it burns on from where it was.
#[test]
fn a_torch_put_away_keeps_its_wear() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        torch_burn_steps: 10,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    agent(&mut env, 0).backpack = Some(Item::Torch { burnt: 4 });

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Noop]);
    assert!(!env.state.lit[start.idx()]);
    assert_eq!(env.state.agents[0].backpack, Some(Item::Torch { burnt: 4 }));

    step(&mut env, &mut buffers, &[Swap]);
    assert!(env.state.lit[start.idx()]);
    step(&mut env, &mut buffers, &[Put]);
    assert_eq!(env.state.map[(start + UP).idx()], ItemTorch);
    step(&mut env, &mut buffers, &[Noop]);
    assert!(
        !env.state.lit[(start + UP).idx()],
        "a laid torch gives no light"
    );
    step(&mut env, &mut buffers, &[Grab]);
    assert_eq!(env.state.agents[0].hands, Some(Item::Torch { burnt: 6 }));
    assert!(env.state.laid_torches.is_empty());
}
