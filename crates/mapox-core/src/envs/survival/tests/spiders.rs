use super::*;
use crate::envs::survival::spiders;

/// Puts a spider from `nest` out on `position`.
fn release_spider(env: &mut Survival, position: Position, nest: Position) {
    env.state.map[position.idx()] = Spider;
    env.state.spiders.push(spiders::Spider { position, nest });
}

fn spiders_on_map(env: &Survival) -> usize {
    env.state.map.iter().filter(|&&t| t == Spider).count()
}

/// Nests hatch a spider each at nightfall, beside them; the spiders hunt
/// through the night and are home again soon after dawn, biting no one on
/// the way.
#[test]
fn spiders_hatch_at_nightfall_and_go_home_at_dawn() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        day_length: 4,
        dusk_length: 0,
        night_length: 2,
        ..Default::default()
    });
    let start = center(&env);
    let nest = start + UP * 3;
    env.set_ground(nest, TileSpiderEggs);
    env.state.eggs.push(nest);
    spawn_facing(&mut env, start + DOWN * 4, 0);

    let mut buffers = TimeStepBuffers::new(&env);
    for _ in 0..3 {
        step(&mut env, &mut buffers, &[Noop]);
    }
    assert_eq!(spiders_on_map(&env), 0, "still day");

    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.spiders.len(), 1, "nightfall");
    assert_eq!(spiders_on_map(&env), 1);
    assert_eq!(env.state.spiders[0].nest, nest);

    // the night's second step, then dawn and the walk home
    let mut steps_out = 0;
    while !env.state.spiders.is_empty() {
        step(&mut env, &mut buffers, &[Noop]);
        steps_out += 1;
        assert!(steps_out < 6, "the spider never got home");
    }
    assert_eq!(spiders_on_map(&env), 0);
    assert_eq!(
        env.state.base_map[nest.idx()],
        TileSpiderEggs,
        "the nest stays"
    );
    assert_eq!(env.metrics.spider_bites, 0.0);
}

/// By day a spider takes the shortest way home, one cell closer every
/// step, and burrows in on the step after it reaches the nest.
#[test]
fn a_spider_walks_the_shortest_way_home() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        day_length: 50,
        dusk_length: 0,
        ..Default::default()
    });
    let start = center(&env);
    let nest = start + UP * 5;
    env.set_ground(nest, TileSpiderEggs);
    env.state.eggs.push(nest);
    spawn_facing(&mut env, start + RIGHT * 6, 0);
    release_spider(&mut env, start + DOWN * 3, nest);

    let mut buffers = TimeStepBuffers::new(&env);
    let mut distances = Vec::new();
    while let Some(spider) = env.state.spiders.first() {
        let gap = spider.position - nest;
        distances.push(gap.x.abs() + gap.y.abs());
        step(&mut env, &mut buffers, &[Noop]);
    }
    assert_eq!(distances, [8, 7, 6, 5, 4, 3, 2, 1]);
    assert_eq!(spiders_on_map(&env), 0);
}

/// A spider caught by a fire lit beside it walks out of the light by the
/// shortest way, and stays out.
#[test]
fn a_spider_in_the_light_walks_out_of_it() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        day_length: 1,
        dusk_length: 0,
        night_length: 50,
        fire_light_radius: 2,
        // nothing to track, so it only has the light to get away from
        spider_hunt_radius: 0,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start + LEFT * 8, 0);
    release_spider(&mut env, start, start + UP * 8);
    light_fire(&mut env, start + RIGHT);
    assert!(env.state.lit[start.idx()]);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.spiders[0].position, start + LEFT);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.spiders[0].position, start + LEFT * 2);
    for _ in 0..10 {
        assert!(!env.state.lit[env.state.spiders[0].position.idx()]);
        step(&mut env, &mut buffers, &[Noop]);
    }
}

/// A nest whose spider is still out doesn't hatch another.
#[test]
fn a_nest_waits_for_its_spider() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        day_length: 1,
        dusk_length: 0,
        night_length: 5,
        ..Default::default()
    });
    let start = center(&env);
    let nest = start + UP * 3;
    env.set_ground(nest, TileSpiderEggs);
    env.state.eggs.push(nest);
    spawn_facing(&mut env, start + LEFT * 8, 0);
    release_spider(&mut env, start + DOWN * 5, nest);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.spiders.len(), 1);
}

/// A spider walks up to an agent in the dark and bites it. Once the agent
/// stands in firelight, the spider can neither bite it nor step closer.
#[test]
fn spiders_hunt_in_the_dark_but_fear_fire() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        day_length: 1,
        dusk_length: 0,
        night_length: 50,
        start_hunger: 50,
        start_health: 100,
        spider_damage: 10,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    release_spider(&mut env, start + RIGHT * 3, start + UP * 8);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.spiders[0].position, start + RIGHT * 2);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.spiders[0].position, start + RIGHT);
    assert_eq!(env.state.agents[0].health, 100);
    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.agents[0].health, 90, "bitten");
    assert_eq!(env.metrics.spider_bites, 1.0);

    light_fire(&mut env, start + DOWN);
    for _ in 0..3 {
        step(&mut env, &mut buffers, &[Noop]);
    }
    assert_eq!(env.state.agents[0].health, 90, "safe in the light");
}

/// Spiders wander the dark but never set foot on lit ground, and can't
/// track an agent standing in it.
#[test]
fn spiders_never_step_into_the_light() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        day_length: 1,
        dusk_length: 0,
        night_length: 100,
        fire_light_radius: 2,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    light_fire(&mut env, start + DOWN);
    release_spider(&mut env, start + RIGHT * 5, start + UP * 8);
    release_spider(&mut env, start + UP * 5, start + UP * 8);

    let mut buffers = TimeStepBuffers::new(&env);
    for _ in 0..40 {
        step(&mut env, &mut buffers, &[Noop]);
        for spider in &env.state.spiders {
            assert!(
                !env.state.lit[spider.position.idx()],
                "a spider at {:?} is in the light",
                spider.position
            );
        }
    }
    assert_eq!(env.metrics.spider_bites, 0.0);
}
