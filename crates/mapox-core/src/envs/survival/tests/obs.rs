use super::*;

/// The band's top row reads health, hunger and temperature as numbers after
/// their labels; the row under it, the hands and backpack items after theirs,
/// then the time of day.
/// The view's centre is the agent, facing the way it faces.
#[test]
fn the_ui_band_shows_stats_and_inventory() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 1);
    let survivor = agent(&mut env, 0);
    survivor.health = 150;
    survivor.hunger = 7;
    survivor.temperature = 42;
    survivor.hands = Some(Item::Axe);

    let buffers = observe(&env);
    let obs = buffers.obs.slice(s![0, .., .., 0]);
    let fov = env.config.view_height as usize;
    let width = env.config.view_width as usize;

    let blank = |n: usize| std::iter::repeat_n(id(UI), n);
    let stats: Vec<VocabId> = [
        UiHealth,
        Digit1,
        Digit5,
        Digit0,
        UI,
        UiHunger,
        UI,
        UI,
        Digit7,
        UI,
        UiTemperature,
        UI,
        Digit4,
        Digit2,
    ]
    .into_iter()
    .map(id)
    .chain(blank(width - 14))
    .collect();
    assert_eq!(obs.column(fov + 1).to_vec(), stats);

    let slots: Vec<VocabId> = [UiHands, ItemAxe, UI, UiBackpack, UI, UI, UiDay]
        .into_iter()
        .map(id)
        .chain(blank(width - 7))
        .collect();
    assert_eq!(obs.column(fov).to_vec(), slots);

    assert_eq!(obs[[width / 2, fov / 2]], id(AgentRight));
}
