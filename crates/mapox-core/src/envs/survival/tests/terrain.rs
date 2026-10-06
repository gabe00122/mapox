use super::*;

/// Reset leaves every agent standing on open ground, painted facing the
/// way it faces, with fresh stats; the scatter puts some of everything on
/// the map; and the walls inside are all diggable, inside an edge of solid
/// wall.
#[test]
fn reset_spawns_agents_on_open_ground_among_the_scatter() {
    let config = SurvivalConfig {
        num_agents: 4,
        width: 40,
        height: 40,
        ..Default::default()
    };
    let mut env = Survival::new(&config, 512);
    let mut buffers = TimeStepBuffers::new(&env);
    env.reset(3, &mut buffers.view_mut());

    for agent in &env.state.agents {
        assert_eq!(env.state.map[agent.position.idx()], agent.tile());
        assert!(env.state.base_map[agent.position.idx()].is_floor());
        assert_eq!(agent.hunger, config.start_hunger);
        assert_eq!(agent.health, config.start_health);
        assert_eq!((agent.hands, agent.backpack), (None, None));
    }

    for tile in [
        TileTree,
        TileBerryBush,
        ItemStick,
        ItemStone,
        TileDestructibleWall,
    ] {
        assert!(
            env.state.base_map.iter().any(|&t| t == tile),
            "reset scattered no {tile:?}"
        );
    }

    let interior = env.state.base_map.slice(s![
        env.pad_width as usize..(env.width - env.pad_width) as usize,
        env.pad_height as usize..(env.height - env.pad_height) as usize,
    ]);
    assert!(!interior.iter().any(|&t| t == TileWall));
    assert_eq!(env.state.base_map[[0, 0]], TileWall);

    assert_eq!(env.state.eggs.len(), config.num_spider_eggs);
    for &nest in &env.state.eggs {
        assert_eq!(env.state.base_map[nest.idx()], TileSpiderEggs);
        assert!(
            AROUND
                .iter()
                .all(|&d| env.state.base_map[(nest + d).idx()].walkable()),
            "a nest at {nest:?} hems in a path"
        );
    }
}

/// However the noise falls, every bit of open ground is reachable from
/// every other, and the agents start out of each other's sight.
#[test]
fn reset_joins_the_open_ground_and_spreads_the_agents() {
    let mut env = Survival::new(&SurvivalConfig::default(), 512);
    let mut buffers = TimeStepBuffers::new(&env);
    for seed in 0..10 {
        env.reset(seed, &mut buffers.view_mut());

        let walkable = env.state.base_map.map(|tile| tile.walkable());
        let (_, sizes) = crate::envs::common::map_gen::label_regions(&walkable);
        let regions = sizes.iter().filter(|&&size| size > 0).count();
        assert_eq!(
            regions, 1,
            "seed {seed} left the ground in {regions} pieces"
        );

        let apart = env.pad_width + 1;
        for (i, a) in env.state.agents.iter().enumerate() {
            for b in &env.state.agents[i + 1..] {
                let gap = (a.position.x - b.position.x)
                    .abs()
                    .max((a.position.y - b.position.y).abs());
                assert!(gap >= apart, "seed {seed}: agents {gap} apart");
            }
        }
    }
}
