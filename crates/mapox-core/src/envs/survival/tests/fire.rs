use super::*;

/// A campfire is set down lit in front, and goes out after
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
    step(&mut env, &mut buffers, &[Use]);
    assert_eq!(env.state.map[fire.idx()], TileFire);
    assert_eq!(env.state.agents[0].hands, None);
    assert_eq!(achieved(&env, Achievement::PlaceFire), 1.0);

    step(&mut env, &mut buffers, &[Noop]);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.map[fire.idx()], TileFire);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.map[fire.idx()], TileEmpty);
}

/// Using a raw berry on the fire in front cooks it in hand at once.
/// Away from a fire a berry has no use, and a cooked one none at all.
#[test]
fn a_berry_held_to_a_fire_cooks_at_once() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    agent(&mut env, 0).hands = Some(Item::Berry);
    assert!(!legal(&observe(&env), 0, Use), "no fire in front");

    light_fire(&mut env, start + UP);
    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Use]);

    assert_eq!(env.state.agents[0].hands, Some(Item::CookedBerry));
    assert_eq!(achieved(&env, Achievement::CookBerry), 1.0);
    assert!(!legal(&buffers, 0, Use), "a cooked berry has no use");
}

/// A fire burns low for its last `fire_low_steps`, and wood used on a low
/// fire stokes it back up to full. Wood does nothing for a fire burning
/// high, and a berry cooks on a low fire as on any other.
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
    step(&mut env, &mut buffers, &[Use]);
    agent(&mut env, 0).hands = Some(Item::Wood);
    let mut burning = vec![env.state.map[fire.idx()]];
    for _ in 0..3 {
        let low = burning.last() == Some(&TileFireLow);
        assert_eq!(
            legal(&observe(&env), 0, Use),
            low,
            "wood stokes only a low fire"
        );
        step(&mut env, &mut buffers, &[Noop]);
        burning.push(env.state.map[fire.idx()]);
    }
    assert_eq!(burning, [TileFire, TileFire, TileFireLow, TileFireLow]);
    assert!(legal(&buffers, 0, Use));

    step(&mut env, &mut buffers, &[Use]);
    assert_eq!(env.state.map[fire.idx()], TileFire);
    assert_eq!(env.state.fires[0].burn_left, 3);
    assert_eq!(env.state.agents[0].hands, None);
    assert_eq!(achieved(&env, Achievement::RefuelFire), 1.0);

    step(&mut env, &mut buffers, &[Noop]);
    step(&mut env, &mut buffers, &[Noop]);
    agent(&mut env, 0).hands = Some(Item::Berry);
    assert_eq!(env.state.map[fire.idx()], TileFireLow);
    assert!(legal(&observe(&env), 0, Use), "a low fire still cooks");
}
