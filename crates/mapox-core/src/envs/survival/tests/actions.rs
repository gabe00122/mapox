use super::*;

/// A move into something solid turns the agent to face it without
/// moving, which is how an agent lines up on a tree. Moves are never masked
/// for that reason.
#[test]
fn a_blocked_move_turns_the_agent_in_place() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        ..Default::default()
    });
    let start = center(&env);
    env.set_ground(start + RIGHT, TileTree);
    spawn_facing(&mut env, start, 0);
    let mut buffers = TimeStepBuffers::new(&env);

    step(&mut env, &mut buffers, &[MoveRight]);
    assert_eq!(env.state.agents[0].position, start);
    assert_eq!(env.state.map[start.idx()], AgentRight);
    for action in [MoveUp, MoveRight, MoveDown, MoveLeft] {
        assert!(legal(&buffers, 0, action));
    }

    step(&mut env, &mut buffers, &[MoveUp]);
    assert_eq!(env.state.agents[0].position, start + UP);
    assert_eq!(env.state.map[(start + UP).idx()], AgentUp);
    assert_eq!(env.state.map[start.idx()], TileEmpty);
}

/// Items lie underfoot: an agent walks onto one, hiding it from every
/// view while it stands there, and it shows again once the agent steps
/// off. Standing on it doesn't reach it; only the tile in front does.
#[test]
fn an_agent_walks_over_an_item_and_hides_it() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        ..Default::default()
    });
    let start = center(&env);
    env.set_ground(start + UP, ItemStick);
    spawn_facing(&mut env, start, 0);
    let mut buffers = TimeStepBuffers::new(&env);

    step(&mut env, &mut buffers, &[MoveUp]);
    assert_eq!(env.state.agents[0].position, start + UP);
    assert_eq!(env.state.map[(start + UP).idx()], AgentUp);
    assert!(!legal(&buffers, 0, Grab), "the stick is underfoot");

    step(&mut env, &mut buffers, &[MoveUp]);
    assert_eq!(env.state.map[(start + UP).idx()], ItemStick);
}

/// Grab takes the item in front into empty hands and leaves floor
/// behind; put sets it back down in front. Collecting counts once per
/// life however often the agent picks the same kind of thing up.
#[test]
fn grab_and_put_work_on_the_tile_in_front() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        ..Default::default()
    });
    let start = center(&env);
    env.set_ground(start + UP, ItemStone);
    spawn_facing(&mut env, start, 0);

    let buffers = observe(&env);
    assert!(legal(&buffers, 0, Grab));
    assert!(!legal(&buffers, 0, Put), "nothing in hand to put");

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Grab]);
    assert_eq!(env.state.agents[0].hands, Some(Item::Stone));
    assert_eq!(env.state.map[(start + UP).idx()], TileEmpty);
    assert!(!legal(&buffers, 0, Grab), "nothing in front to grab");
    assert!(legal(&buffers, 0, Put));

    step(&mut env, &mut buffers, &[Put]);
    assert_eq!(env.state.agents[0].hands, None);
    assert_eq!(env.state.map[(start + UP).idx()], ItemStone);

    step(&mut env, &mut buffers, &[Grab]);
    assert_eq!(achieved(&env, Achievement::CollectStone), 1.0);
}

/// Only open ground takes an item put down: not a tree, an agent, or
/// another item.
#[test]
fn put_needs_open_ground_in_front() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 2,
        ..Default::default()
    });
    let start = center(&env);
    env.set_ground(start + UP, TileTree);
    spawn_facing(&mut env, start, 0);
    spawn_facing(&mut env, start + RIGHT * 2, 3);
    env.set_ground(start + RIGHT, ItemStick);
    agent(&mut env, 0).hands = Some(Item::Stone);
    agent(&mut env, 1).hands = Some(Item::Stone);

    let buffers = observe(&env);
    assert!(!legal(&buffers, 0, Put), "a tree is in front");
    assert!(!legal(&buffers, 1, Put), "a stick is in front");
}

/// Two agents reaching for the same item: whoever goes first in the turn
/// order gets it, and the other's grab does nothing.
#[test]
fn two_agents_cannot_take_the_same_item() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 2,
        ..Default::default()
    });
    let start = center(&env);
    env.set_ground(start + RIGHT, ItemStick);
    spawn_facing(&mut env, start, 1);
    spawn_facing(&mut env, start + RIGHT * 2, 3);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Grab, Grab]);

    let holding: Vec<_> = env.state.agents.iter().map(|a| a.hands).collect();
    assert!(
        holding == [Some(Item::Stick), None] || holding == [None, Some(Item::Stick)],
        "{holding:?}"
    );
    assert_eq!(env.state.map[(start + RIGHT).idx()], TileEmpty);
}

/// With full hands, grab fills an empty backpack instead; with both
/// slots full it is masked.
#[test]
fn grab_with_full_hands_goes_into_the_backpack() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        ..Default::default()
    });
    let start = center(&env);
    env.set_ground(start + UP, ItemStone);
    spawn_facing(&mut env, start, 0);
    agent(&mut env, 0).hands = Some(Item::Stick);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Grab]);
    let survivor = env.state.agents[0];
    assert_eq!(
        (survivor.hands, survivor.backpack),
        (Some(Item::Stick), Some(Item::Stone))
    );
    assert_eq!(achieved(&env, Achievement::CollectStone), 1.0);

    env.set_ground(start + UP, ItemStick);
    assert!(!legal(&observe(&env), 0, Grab), "both slots are full");
}

#[test]
fn swap_trades_hands_and_backpack() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    assert!(!legal(&observe(&env), 0, Swap), "both slots are empty");

    agent(&mut env, 0).hands = Some(Item::Stick);
    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Swap]);
    let survivor = env.state.agents[0];
    assert_eq!(
        (survivor.hands, survivor.backpack),
        (None, Some(Item::Stick))
    );
}

/// Recipes work with either ingredient in hand, put the result in hand,
/// and empty the backpack. Combine is masked for pairs with no recipe.
#[test]
fn combine_follows_the_recipes_either_way_round() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        ..Default::default()
    });
    let start = center(&env);
    spawn_facing(&mut env, start, 0);
    let mut buffers = TimeStepBuffers::new(&env);

    for (hands, backpack, made) in [
        (Item::Stick, Item::Stone, Item::Axe),
        (Item::Stone, Item::Stick, Item::Axe),
        (Item::Wood, Item::Grass, Item::Campfire),
        (Item::Grass, Item::Wood, Item::Campfire),
        // a torch is lit in hand, so it burns through the step it is made
        (Item::Stick, Item::Grass, Item::Torch { burnt: 1 }),
        (Item::Grass, Item::Stick, Item::Torch { burnt: 1 }),
    ] {
        agent(&mut env, 0).hands = Some(hands);
        agent(&mut env, 0).backpack = Some(backpack);
        step(&mut env, &mut buffers, &[Combine]);
        let survivor = env.state.agents[0];
        assert_eq!((survivor.hands, survivor.backpack), (Some(made), None));
    }
    assert_eq!(achieved(&env, Achievement::MakeAxe), 1.0);
    assert_eq!(achieved(&env, Achievement::MakeCampfire), 1.0);
    assert_eq!(achieved(&env, Achievement::MakeTorch), 1.0);

    for (hands, backpack) in [(Item::Stick, Item::Stick), (Item::Wood, Item::Stone)] {
        agent(&mut env, 0).hands = Some(hands);
        agent(&mut env, 0).backpack = Some(backpack);
        assert!(!legal(&observe(&env), 0, Combine));
    }
}

/// Using the axe on a tree locks the agent in place, able only to wait,
/// and after `chop_steps` in all a log lies where the tree stood. The axe
/// stays in hand.
#[test]
fn chopping_holds_the_agent_until_the_log_drops() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        chop_steps: 4,
        ..Default::default()
    });
    let start = center(&env);
    let tree = start + UP;
    spawn_facing(&mut env, start, 0);
    agent(&mut env, 0).hands = Some(Item::Axe);
    assert!(!legal(&observe(&env), 0, Use), "no tree in front");

    env.set_ground(tree, TileTree);
    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Use]);
    for _ in 0..2 {
        assert_eq!(env.state.map[tree.idx()], TileTree);
        let mask = buffers.action_mask.row(0);
        assert_eq!(mask.iter().filter(|&&legal| legal).count(), 1);
        assert!(mask[Noop as usize], "only waiting while at work");
        step(&mut env, &mut buffers, &[Noop]);
    }
    assert_eq!(env.state.map[tree.idx()], TileTree);

    step(&mut env, &mut buffers, &[Noop]);
    assert_eq!(env.state.map[tree.idx()], ItemWood);
    assert_eq!(env.state.agents[0].hands, Some(Item::Axe));
    assert_eq!(achieved(&env, Achievement::ChopTree), 1.0);
    assert!(legal(&buffers, 0, MoveDown), "free again");
}

/// Whatever a busy agent asks for, its turn goes into the job.
#[test]
fn a_busy_agent_only_works() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        chop_steps: 3,
        ..Default::default()
    });
    let start = center(&env);
    env.set_ground(start + UP, TileTree);
    spawn_facing(&mut env, start, 0);
    agent(&mut env, 0).hands = Some(Item::Axe);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Use]);
    step(&mut env, &mut buffers, &[MoveDown]);
    assert_eq!(env.state.agents[0].position, start);
    step(&mut env, &mut buffers, &[Swap]);
    assert_eq!(env.state.agents[0].hands, Some(Item::Axe));
    assert_eq!(env.state.map[(start + UP).idx()], ItemWood);
}

/// Two agents felling the same tree get one log between them: whoever
/// finishes second finds the tree gone and gets nothing.
#[test]
fn two_agents_felling_one_tree_get_one_log() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 2,
        chop_steps: 2,
        ..Default::default()
    });
    let start = center(&env);
    let tree = start + RIGHT;
    env.set_ground(tree, TileTree);
    spawn_facing(&mut env, start, 1);
    spawn_facing(&mut env, start + RIGHT * 2, 3);
    agent(&mut env, 0).hands = Some(Item::Axe);
    agent(&mut env, 1).hands = Some(Item::Axe);

    let mut buffers = TimeStepBuffers::new(&env);
    step(&mut env, &mut buffers, &[Use, Use]);
    step(&mut env, &mut buffers, &[Noop, Noop]);
    assert_eq!(env.state.map[tree.idx()], ItemWood);
    assert!(env.state.agents.iter().all(|a| a.work.is_none()));
    assert_eq!(achieved(&env, Achievement::ChopTree), 1.0);
}

/// The axe clears a bush, ripe, bare or dead, as slow work like felling, and
/// leaves a stick where it grew. A bush picked bare and cleared before it
/// fruits again stays gone.
#[test]
fn the_axe_clears_a_bush_for_a_stick() {
    let mut env = empty_env_with(SurvivalConfig {
        num_agents: 1,
        clear_bush_steps: 2,
        bush_regrow_steps: 4,
        ..Default::default()
    });
    let start = center(&env);
    let bush = start + UP;
    spawn_facing(&mut env, start, 0);
    let mut buffers = TimeStepBuffers::new(&env);

    for tile in [TileBerryBush, TileBush, TileDeadBush] {
        env.set_ground(bush, tile);
        assert!(!legal(&observe(&env), 0, Use), "no axe in hand");
        agent(&mut env, 0).hands = Some(Item::Axe);
        step(&mut env, &mut buffers, &[Use]);
        assert_eq!(env.state.map[bush.idx()], tile, "still clearing");
        step(&mut env, &mut buffers, &[Noop]);
        assert_eq!(env.state.map[bush.idx()], ItemStick);
        assert_eq!(env.state.agents[0].hands, Some(Item::Axe));
        agent(&mut env, 0).hands = None;
    }
    assert_eq!(achieved(&env, Achievement::ClearBush), 1.0);

    env.set_ground(bush, TileBerryBush);
    step(&mut env, &mut buffers, &[Grab]);
    step(&mut env, &mut buffers, &[Swap]);
    agent(&mut env, 0).hands = Some(Item::Axe);
    step(&mut env, &mut buffers, &[Use]);
    step(&mut env, &mut buffers, &[Noop]);
    for _ in 0..4 {
        step(&mut env, &mut buffers, &[Noop]);
    }
    assert_eq!(env.state.map[bush.idx()], ItemStick, "no regrowth");
}
