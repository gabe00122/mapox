use ndarray::{Array2, s};
use rand::{RngExt, SeedableRng, rngs::SmallRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::{
    env::Environment,
    envs::common::{Position, UI_HEIGHT, fov, vocab_enum::VocabEnum},
    render::env::{GridRenderSettings, GridRenderState},
    spec::{ActionSpec, ObservationSpec},
    symbols::{
        AGENT_GENERIC, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, NOOP, TILE_EMPTY,
        TILE_FLAG_UNLOCKED, TILE_MASK, TILE_UI, TILE_WALL,
    },
    timestep::TimeStepMut,
    vocab::{VocabId, Vocabulary},
    vocab_enum,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MazeConfig {
    pub num_agents: usize,

    /// The maze's cells sit on the even coordinates of the interior with a
    /// wall or a passage between each pair, so an odd size uses every row
    /// and column; an even one leaves its last row or column solid wall.
    pub width: i32,
    pub height: i32,
    pub view_width: i32,
    pub view_height: i32,

    pub goal_reward: f32,
    /// Every agent starts, and restarts after each goal, from one shared
    /// spawn instead of a spawn of its own.
    pub shared_spawn: bool,
}

impl Default for MazeConfig {
    fn default() -> Self {
        Self {
            num_agents: 8,
            width: 31,
            height: 31,
            view_width: 15,
            view_height: 15,
            goal_reward: 1.0,
            shared_spawn: false,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct MazeAgent {
    position: Position,
    /// Where the agent starts, and where reaching the goal sends it back to.
    spawn: Position,
}

#[derive(Debug, Clone)]
struct MazeState {
    rngs: SmallRng,
    agents: Vec<MazeAgent>,
    time: usize,

    base_map: Array2<MazeObs>, // the bottom layer of the map without agents
    map: Array2<MazeObs>,      // the base map plus the agents
    cells: Vec<Position>,      // every maze cell, for picking the goal and spawns
}

vocab_enum!(MazeObs {
    UI => TILE_UI,
    Mask => TILE_MASK,
    TileEmpty => TILE_EMPTY,
    TileWall => TILE_WALL,
    TileGoal => TILE_FLAG_UNLOCKED,
    AgentGeneric => AGENT_GENERIC,
});

impl fov::ViewTile for MazeObs {
    const MASK: Self = MazeObs::Mask;

    fn opaque(self) -> bool {
        self == MazeObs::TileWall
    }
}

vocab_enum!(MazeAction {
    MoveUp => MOVE_UP,
    MoveRight => MOVE_RIGHT,
    MoveDown => MOVE_DOWN,
    MoveLeft => MOVE_LEFT,
    Noop => NOOP,
});

impl MazeAction {
    fn direction(self) -> Position {
        use MazeAction::*;
        match self {
            MoveUp => Position::new(0, 1),
            MoveRight => Position::new(1, 0),
            MoveDown => Position::new(0, -1),
            MoveLeft => Position::new(-1, 0),
            Noop => Position::new(0, 0),
        }
    }
}

#[derive(Debug, Default, Clone)]
struct MazeMetrics {
    reward: f64,
    goals_reached: f64,
}

/// A procedurally generated maze with a single goal. Agents never block one
/// another; each walks from its own spawn to the goal and is sent straight
/// back to that spawn to do it again.
#[derive(Debug, Clone)]
pub struct Maze {
    pub config: MazeConfig,
    state: MazeState,
    metrics: MazeMetrics,

    // max steps for a single episode
    length: usize,

    pad_width: i32,
    pad_height: i32,

    // full map size including wall padding on both sides
    width: i32,
    height: i32,

    obs_spec: ObservationSpec,
    action_spec: ActionSpec,

    obs_vocab: Vocabulary,
    action_vocab: Vocabulary,
}

impl Maze {
    pub fn new(config: &MazeConfig, length: usize) -> Self {
        assert!(
            config.width >= 1 && config.height >= 1 && config.width.max(config.height) >= 3,
            "a maze needs room for at least two cells, got {}x{}",
            config.width,
            config.height,
        );

        let action_vocab = MazeAction::vocab();
        let obs_vocab = MazeObs::vocab();

        let pad_width = config.view_width / 2;
        let pad_height = config.view_height / 2;

        let view_height = config.view_height + UI_HEIGHT as i32;

        let width = config.width + 2 * pad_width;
        let height = config.height + 2 * pad_height;

        let obs_spec = ObservationSpec::new(config.view_width, view_height, obs_vocab.len());
        let action_spec = ActionSpec::new(action_vocab.len());

        let dim = (width as usize, height as usize);
        Self {
            config: config.clone(),
            state: MazeState {
                rngs: SmallRng::seed_from_u64(0),
                agents: Vec::with_capacity(config.num_agents),
                time: 0,
                base_map: Array2::from_elem(dim, MazeObs::TileWall),
                map: Array2::from_elem(dim, MazeObs::TileWall),
                cells: Vec::new(),
            },
            metrics: MazeMetrics::default(),
            length,

            pad_width,
            pad_height,
            width,
            height,

            obs_spec,
            action_spec,

            obs_vocab,
            action_vocab,
        }
    }

    /// Carves a perfect maze into `base_map` with a randomized depth-first
    /// search: every cell is reachable from every other by exactly one path.
    /// Cells sit on the even interior coordinates, and the wall padding
    /// around the interior is the maze's outer wall.
    fn carve_maze(&mut self) {
        let MazeState {
            rngs,
            base_map,
            cells,
            ..
        } = &mut self.state;

        base_map.fill(MazeObs::TileWall);
        cells.clear();

        let cells_x = (self.config.width + 1) / 2;
        let cells_y = (self.config.height + 1) / 2;
        let origin = Position::new(self.pad_width, self.pad_height);
        let to_map = |cell: Position| origin + cell * 2;

        let mut visited = Array2::from_elem((cells_x as usize, cells_y as usize), false);
        let start = Position::new(rngs.random_range(0..cells_x), rngs.random_range(0..cells_y));
        visited[start.idx()] = true;
        base_map[to_map(start).idx()] = MazeObs::TileEmpty;
        cells.push(to_map(start));
        let mut stack = vec![start];

        let steps = [
            Position::new(0, 1),
            Position::new(1, 0),
            Position::new(0, -1),
            Position::new(-1, 0),
        ];
        while let Some(&cell) = stack.last() {
            let mut options = [Position::default(); 4];
            let mut count = 0;
            for step in steps {
                let next = cell + step;
                if next.x >= 0
                    && next.x < cells_x
                    && next.y >= 0
                    && next.y < cells_y
                    && !visited[next.idx()]
                {
                    options[count] = next;
                    count += 1;
                }
            }

            if count == 0 {
                stack.pop();
                continue;
            }

            let next = options[rngs.random_range(0..count)];
            visited[next.idx()] = true;
            // knock out the wall between the two cells, then open the new one
            base_map[(to_map(cell) + next - cell).idx()] = MazeObs::TileEmpty;
            base_map[to_map(next).idx()] = MazeObs::TileEmpty;
            cells.push(to_map(next));
            stack.push(next);
        }
    }

    fn encode_observations(&self, timestep: &mut TimeStepMut) {
        let fov_height = self.config.view_height as usize;
        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            // wall padding keeps the view window inside the map
            let mut view = timestep.obs.slice_mut(s![agent_id, .., ..fov_height, 0]);
            fov::observe(&self.state.map, agent.position, &mut view);

            let mut ui = timestep.obs.slice_mut(s![agent_id, .., fov_height.., 0]);
            ui.fill(MazeObs::UI as VocabId);
        }

        timestep.time.fill(self.state.time as i32);
        timestep.terminated.fill(self.state.time == self.length);
        timestep.task_ids.fill(0);
    }

    fn encode_action_mask(&self, timestep: &mut TimeStepMut) {
        for (agent_id, agent) in self.state.agents.iter().enumerate() {
            let mut mask = timestep.action_mask.row_mut(agent_id);
            for &action in MazeAction::TABLE {
                let target = agent.position + action.direction();
                mask[action as usize] = self.state.base_map[target.idx()] != MazeObs::TileWall;
            }
        }
    }
}

impl Environment for Maze {
    fn reset(&mut self, seed: u64, timestep: &mut TimeStepMut) {
        self.state.rngs = SmallRng::seed_from_u64(seed);
        self.state.time = 0;

        self.carve_maze();

        // the goal and the spawns are all distinct cells while there are
        // enough to go round; past that, spawns double up
        let MazeState {
            rngs,
            cells,
            agents,
            base_map,
            map,
            ..
        } = &mut self.state;
        cells.shuffle(rngs);
        let goal = cells[0];
        base_map[goal.idx()] = MazeObs::TileGoal;

        let spawns = &cells[1..];
        agents.clear();
        for agent_id in 0..self.config.num_agents {
            let spawn = if self.config.shared_spawn {
                spawns[0]
            } else {
                spawns[agent_id % spawns.len()]
            };
            agents.push(MazeAgent {
                position: spawn,
                spawn,
            });
        }

        map.assign(base_map);
        for agent in agents.iter() {
            map[agent.position.idx()] = MazeObs::AgentGeneric;
        }

        timestep.reward.fill(0.0);
        timestep.last_action.fill(0);
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn step(&mut self, actions: &[VocabId], timestep: &mut TimeStepMut) {
        let MazeState {
            agents,
            base_map,
            map,
            ..
        } = &mut self.state;

        // agents can share a cell, so lift them all off before any move and
        // paint them all back after
        for agent in agents.iter() {
            map[agent.position.idx()] = base_map[agent.position.idx()];
        }

        for (agent_id, agent) in agents.iter_mut().enumerate() {
            timestep.last_action[agent_id] = actions[agent_id];
            timestep.reward[agent_id] = 0.0;

            let action = MazeAction::from_id(actions[agent_id]);
            let target = agent.position + action.direction();
            if base_map[target.idx()] == MazeObs::TileWall {
                continue;
            }
            agent.position = target;

            if base_map[target.idx()] == MazeObs::TileGoal {
                agent.position = agent.spawn;
                timestep.reward[agent_id] = self.config.goal_reward;
                self.metrics.reward += f64::from(self.config.goal_reward);
                self.metrics.goals_reached += 1.0;
            }
        }

        for agent in agents.iter() {
            map[agent.position.idx()] = MazeObs::AgentGeneric;
        }

        self.state.time += 1;
        self.encode_observations(timestep);
        self.encode_action_mask(timestep);
    }

    fn consume_metrics(&mut self) -> serde_json::Value {
        let metrics = std::mem::take(&mut self.metrics);
        let agents = self.num_agents().max(1) as f64;
        serde_json::json!({
            "reward": metrics.reward / agents,
            "goals_reached": metrics.goals_reached / agents,
        })
    }

    fn observation_spec(&self) -> ObservationSpec {
        self.obs_spec
    }

    fn action_spec(&self) -> ActionSpec {
        self.action_spec
    }

    fn num_agents(&self) -> usize {
        self.config.num_agents
    }

    fn obs_vocab(&self) -> &Vocabulary {
        &self.obs_vocab
    }

    fn action_vocab(&self) -> &Vocabulary {
        &self.action_vocab
    }

    fn get_render_settings(&self) -> GridRenderSettings {
        GridRenderSettings {
            obs_vocab: self.obs_vocab.clone(),
            tile_width: self.config.width as usize,
            tile_height: self.config.height as usize,
            view_width: self.config.view_width as usize,
            view_height: self.config.view_height as usize + UI_HEIGHT,
            ui_height: UI_HEIGHT,
        }
    }

    fn render_state_into(&self, grid_render_state: &mut GridRenderState) {
        let dim = (self.config.width as usize, self.config.height as usize);
        let tilemap = &mut grid_render_state.tilemap;
        if tilemap.dim() != dim {
            *tilemap = Array2::zeros(dim);
        }
        let interior = self.state.map.slice(s![
            self.pad_width as usize..(self.width - self.pad_width) as usize,
            self.pad_height as usize..(self.height - self.pad_height) as usize,
        ]);
        tilemap.zip_mut_with(&interior, |dst, &tile| *dst = tile.into());

        grid_render_state.agent_positions.clear();
        for agent in &self.state.agents {
            let local_pos = agent.position - Position::new(self.pad_width, self.pad_height);
            grid_render_state.agent_positions.push(local_pos);
        }
    }

    fn num_tasks(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envs::common::map_gen::label_regions;
    use crate::timestep::TimeStepBuffers;
    use MazeAction::*;
    use MazeObs::*;
    use rand::RngExt;
    use std::collections::HashSet;

    fn reset_env(config: MazeConfig, seed: u64) -> (Maze, TimeStepBuffers) {
        let mut env = Maze::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        env.reset(seed, &mut buffers.view_mut());
        (env, buffers)
    }

    fn step(env: &mut Maze, buffers: &mut TimeStepBuffers, actions: &[MazeAction]) {
        let actions: Vec<VocabId> = actions.iter().map(|&a| a.into()).collect();
        env.step(&actions, &mut buffers.view_mut());
    }

    fn goal(env: &Maze) -> Position {
        let ([x, y], _) = env
            .state
            .base_map
            .indexed_iter()
            .map(|((x, y), &tile)| ([x, y], tile))
            .find(|&(_, tile)| tile == TileGoal)
            .expect("the maze has a goal");
        Position::new(x as i32, y as i32)
    }

    /// The open cells next to `position`, as the move that reaches each.
    fn open_moves(env: &Maze, position: Position) -> Vec<MazeAction> {
        [MoveUp, MoveRight, MoveDown, MoveLeft]
            .into_iter()
            .filter(|&a| env.state.base_map[(position + a.direction()).idx()] != TileWall)
            .collect()
    }

    /// A perfect maze is a spanning tree over its cells: one connected
    /// region, and exactly one passage per cell but the first, so no loops.
    #[test]
    fn the_maze_is_a_spanning_tree_over_every_cell() {
        for (width, height) in [(31, 31), (20, 9), (3, 1), (1, 7)] {
            let (env, _) = reset_env(
                MazeConfig {
                    width,
                    height,
                    ..Default::default()
                },
                3,
            );
            let cells = ((width + 1) / 2 * ((height + 1) / 2)) as usize;
            let open = env.state.base_map.map(|&tile| tile != TileWall);
            let (_, sizes) = label_regions(&open);

            assert_eq!(env.state.cells.len(), cells);
            assert_eq!(sizes.iter().filter(|&&size| size > 0).count(), 1);
            assert_eq!(
                open.iter().filter(|&&o| o).count(),
                2 * cells - 1,
                "{width}x{height}"
            );

            // nothing open outside the interior
            for ((x, y), &o) in open.indexed_iter() {
                let (x, y) = (x as i32, y as i32);
                if o {
                    assert!(x >= env.pad_width && x < env.width - env.pad_width);
                    assert!(y >= env.pad_height && y < env.height - env.pad_height);
                }
            }
        }
    }

    #[test]
    fn reset_places_one_goal_and_every_agent_on_its_own_spawn() {
        let (env, buffers) = reset_env(MazeConfig::default(), 7);

        let goals = env
            .state
            .base_map
            .iter()
            .filter(|&&t| t == TileGoal)
            .count();
        assert_eq!(goals, 1);

        let spawns: HashSet<_> = env.state.agents.iter().map(|a| a.spawn).collect();
        assert_eq!(spawns.len(), env.num_agents(), "spawns are distinct");
        assert!(!spawns.contains(&goal(&env)));
        for agent in &env.state.agents {
            assert_eq!(agent.position, agent.spawn);
            assert_eq!(env.state.map[agent.position.idx()], AgentGeneric);
        }

        // walls are masked out, the noop never is
        for (agent_id, agent) in env.state.agents.iter().enumerate() {
            let mask = buffers.action_mask.row(agent_id);
            assert!(mask[Noop as usize]);
            for action in [MoveUp, MoveRight, MoveDown, MoveLeft] {
                assert_eq!(
                    mask[action as usize],
                    open_moves(&env, agent.position).contains(&action)
                );
            }
        }
    }

    #[test]
    fn a_shared_spawn_starts_everyone_together() {
        let (env, _) = reset_env(
            MazeConfig {
                shared_spawn: true,
                ..Default::default()
            },
            7,
        );
        let spawns: HashSet<_> = env.state.agents.iter().map(|a| a.spawn).collect();
        assert_eq!(spawns.len(), 1);
        assert!(!spawns.contains(&goal(&env)));
    }

    /// Reaching the goal pays and sends the agent back to its own spawn, the
    /// same one every time.
    #[test]
    fn the_goal_pays_and_returns_the_agent_to_its_spawn() {
        let (mut env, mut buffers) = reset_env(
            MazeConfig {
                num_agents: 2,
                ..Default::default()
            },
            1,
        );
        let goal = goal(&env);
        // stand the agent next to the goal by hand, then step onto it
        let stand = goal + open_moves(&env, goal)[0].direction();
        let into_goal = open_moves(&env, stand)
            .into_iter()
            .find(|a| stand + a.direction() == goal)
            .unwrap();
        let spawn = env.state.agents[0].spawn;

        for _ in 0..2 {
            env.state.map[env.state.agents[0].position.idx()] = TileEmpty;
            env.state.agents[0].position = stand;
            step(&mut env, &mut buffers, &[into_goal, Noop]);

            assert_eq!(buffers.reward[0], env.config.goal_reward);
            assert_eq!(buffers.reward[1], 0.0);
            assert_eq!(env.state.agents[0].position, spawn);
            assert_eq!(env.state.map[spawn.idx()], AgentGeneric);
            assert_eq!(env.state.base_map[goal.idx()], TileGoal);
        }

        assert_eq!(
            env.consume_metrics(),
            serde_json::json!({"reward": 1.0, "goals_reached": 1.0})
        );
    }

    /// Agents walk through each other, and one leaving a shared cell does
    /// not wipe the other off the map.
    #[test]
    fn agents_share_cells() {
        let (mut env, mut buffers) = reset_env(
            MazeConfig {
                num_agents: 2,
                shared_spawn: true,
                ..Default::default()
            },
            5,
        );
        let start = env.state.agents[0].position;
        let away = open_moves(&env, start)[0];

        step(&mut env, &mut buffers, &[away, Noop]);
        assert_eq!(env.state.map[start.idx()], AgentGeneric);
        assert_eq!(
            env.state.map[(start + away.direction()).idx()],
            AgentGeneric
        );

        step(&mut env, &mut buffers, &[Noop, away]);
        assert_eq!(env.state.agents[0].position, env.state.agents[1].position);
        assert_ne!(env.state.map[start.idx()], AgentGeneric);
    }

    #[test]
    fn walls_block_moves() {
        let (mut env, mut buffers) = reset_env(
            MazeConfig {
                num_agents: 1,
                ..Default::default()
            },
            2,
        );
        let start = env.state.agents[0].position;
        let open = open_moves(&env, start);
        let blocked = [MoveUp, MoveRight, MoveDown, MoveLeft]
            .into_iter()
            .find(|a| !open.contains(a))
            .expect("a maze cell has a wall on some side");

        step(&mut env, &mut buffers, &[blocked]);
        assert_eq!(env.state.agents[0].position, start);
    }

    /// A full episode of random moves keeps every agent on open ground and
    /// painted on the map, and the maze itself never changes.
    #[test]
    fn a_random_rollout_keeps_the_world_consistent() {
        let config = MazeConfig {
            width: 15,
            height: 15,
            ..Default::default()
        };
        let mut env = Maze::new(&config, 512);
        let mut buffers = TimeStepBuffers::new(&env);
        let mut rng = SmallRng::seed_from_u64(11);

        for seed in 0..4 {
            env.reset(seed, &mut buffers.view_mut());
            let maze = env.state.base_map.clone();

            for _ in 0..512 {
                let actions: Vec<VocabId> = (0..env.num_agents())
                    .map(|_| rng.random_range(0..env.action_spec.num_actions) as VocabId)
                    .collect();
                env.step(&actions, &mut buffers.view_mut());
            }

            assert_eq!(env.state.base_map, maze, "on seed {seed}");
            for agent in &env.state.agents {
                assert_ne!(env.state.base_map[agent.position.idx()], TileWall);
                assert_ne!(env.state.base_map[agent.position.idx()], TileGoal);
                assert_eq!(env.state.map[agent.position.idx()], AgentGeneric);
            }
            let painted = env.state.map.iter().filter(|&&t| t == AgentGeneric).count();
            let occupied: HashSet<_> = env.state.agents.iter().map(|a| a.position).collect();
            assert_eq!(painted, occupied.len());
        }
        assert!(buffers.terminated.iter().all(|&t| t));
    }
}
