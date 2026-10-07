use super::*;

/// Eating uses the berry up and fills hunger, never past the cap. Only
/// food can be eaten.
#[test]
fn eating_restores_hunger_up_to_the_cap() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        start_hunger: 10,
        // no drain to muddle the sums
        hunger_interval: 1000,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    assert!(!legal(&observe(&env), 0, Eat), "nothing in hand");
    agent(&mut env, 0).hands = Some(Item::Stick);
    assert!(!legal(&observe(&env), 0, Eat), "a stick is no food");
    agent(&mut env, 0).hands = Some(Item::Berry);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Eat]);
    assert_eq!(env.state.agents[0].hunger, 10 + env.config.berry_food);
    assert_eq!(env.state.agents[0].hands, None);
    assert_eq!(achieved(&env, Achievement::EatBerry), 1.0);

    agent(&mut env, 0).hunger = MAX_STAT - 5;
    agent(&mut env, 0).hands = Some(Item::CookedBerry);
    step(&mut env, &mut buffers, &[Eat]);
    assert_eq!(env.state.agents[0].hunger, MAX_STAT);
    assert_eq!(achieved(&env, Achievement::EatCookedBerry), 1.0);

    agent(&mut env, 0).hunger = 10;
    agent(&mut env, 0).hands = Some(Item::Carrot);
    step(&mut env, &mut buffers, &[Eat]);
    assert_eq!(env.state.agents[0].hunger, 10 + env.config.carrot_food);
    assert_eq!(achieved(&env, Achievement::EatCarrot), 1.0);
}

/// Hunger drains one point per `hunger_interval` steps, and while it
/// stays high, health grows back one per `regen_interval`.
#[test]
fn hunger_drains_and_a_fed_agent_heals() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        start_hunger: 120,
        hunger_interval: 4,
        start_health: 50,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);

    let mut buffers = TimeStepBuffers::new(&env);
    for _ in 0..8 {
        step(&mut env, &mut buffers, &[Noop]);
    }
    assert_eq!(env.state.agents[0].hunger, 118);
    assert_eq!(env.state.agents[0].health, 52);
}

/// At zero hunger health drains; at zero health the agent is flagged
/// terminated, leaves what it carried where it stood, and respawns
/// elsewhere with fresh stats and achievements.
#[test]
fn starving_agents_die_drop_their_things_and_respawn() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        start_hunger: 0,
        start_health: 2,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    let survivor = agent(&mut env, 0);
    survivor.hands = Some(Item::Stick);
    survivor.backpack = Some(Item::Stone);
    survivor.unlocked = !0;

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.agents[0].health, 1);
    assert!(!buffers.terminated[0]);

    step(&mut env, &mut buffers, &[Noop]);
    assert!(buffers.terminated[0]);
    assert_eq!(env.state.map[start.idx()], ItemStick);
    let dropped = DIRECTIONS
        .iter()
        .filter(|&&d| env.state.map[(start + d).idx()] == ItemStone)
        .count();
    assert_eq!(dropped, 1);

    let respawned = env.state.agents[0];
    assert_ne!(respawned.position, start);
    assert_eq!(env.state.map[respawned.position.idx()], respawned.tile());
    assert_eq!((respawned.health, respawned.hunger), (2, 0));
    assert_eq!((respawned.hands, respawned.backpack), (None, None));
    assert_eq!(respawned.unlocked, 0);

    let metrics = env.consume_metrics();
    assert_eq!(metrics["deaths"], 1.0);
    assert_eq!(metrics["achievements"]["make_axe"], 0.0);
}
