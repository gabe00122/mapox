use super::*;

/// What the agent's view shows at `offset` from it.
fn seen(
    env: &Survival,
    buffers: &TimeStepBuffers,
    agent_id: usize,
    offset: Position,
) -> SurvivalObs {
    let x = env.config.view_width / 2 + offset.x;
    let y = env.config.view_height / 2 + offset.y;
    SurvivalObs::from_id(buffers.obs[[agent_id, x as usize, y as usize, 0]])
}

/// The clock in the band shows the sun through `day_length` steps and
/// the moon through `night_length`, then the sun again.
#[test]
fn night_follows_day_on_the_clock() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        day_length: 2,
        dusk_length: 0,
        night_length: 3,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    let slots_row = env.config.view_height as usize;
    let clock = move |buffers: &TimeStepBuffers| {
        SurvivalObs::from_id(buffers.obs[[0, CLOCK_COL, slots_row, 0]])
    };

    let mut buffers = observe(&env);
    let mut seen = vec![clock(&buffers)];
    for _ in 0..6 {
        step(&mut env, &mut buffers, &[Noop]);
        seen.push(clock(&buffers));
    }
    assert_eq!(
        seen,
        [UiDay, UiDay, UiNight, UiNight, UiNight, UiDay, UiDay]
    );
}

/// By night the agent sees only what is near it, and lit ground as far as
/// by day: a fire in the distance shows, with what is around it.
#[test]
fn at_night_only_the_near_and_the_lit_show() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        day_length: 1,
        dusk_length: 0,
        night_length: 10,
        night_vision_radius: 1,
        fire_light_radius: 1,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    light_fire(&mut env, start + RIGHT * 5);

    let by_day = observe(&env);
    assert_eq!(seen(&env, &by_day, 0, RIGHT * 3), TileEmpty);
    assert_eq!(seen(&env, &by_day, 0, UP * 5), TileEmpty);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(seen(&env, &buffers, 0, RIGHT), TileEmpty, "near");
    assert_eq!(seen(&env, &buffers, 0, RIGHT * 3), Mask, "dark");
    assert_eq!(seen(&env, &buffers, 0, RIGHT * 4), TileEmpty, "lit");
    assert_eq!(seen(&env, &buffers, 0, RIGHT * 5), TileFire);
    assert_eq!(
        seen(&env, &buffers, 0, RIGHT * 5 + UP * 2),
        Mask,
        "past the light"
    );
    assert_eq!(seen(&env, &buffers, 0, UP * 5), Mask);
}

/// Through dusk, sight closes in a little every step, from the whole view
/// down to the night's radius.
#[test]
fn dusk_closes_in_step_by_step() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        day_length: 10,
        dusk_length: 4,
        night_length: 5,
        night_vision_radius: 1,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    let fov = env.config.view_height as usize;
    let visible = |buffers: &TimeStepBuffers| {
        let view = buffers.obs.slice(s![0, .., ..fov, 0]);
        view.iter().filter(|&&id| id != VocabId::from(Mask)).count()
    };

    let mut buffers = observe(&env);
    let mut counts = vec![visible(&buffers)];
    for _ in 0..11 {
        step(&mut env, &mut buffers, &[Noop]);
        counts.push(visible(&buffers));
    }

    let whole = (env.config.view_width as usize) * fov;
    assert_eq!(counts[..6], [whole; 6], "full day");
    assert!(counts[5..11].is_sorted_by(|a, b| a > b), "dusk: {counts:?}");
    assert_eq!(counts[10], counts[11], "night holds steady");
    assert_eq!(
        counts[10], 9,
        "the night's radius-1 disc, rounded out to 3x3"
    );
}
