"""Game-logic tests specific to the embodied communication env.

The contract tests in test_env_contract.py cover shapes; this file covers the
reward rules, commit mechanics, and map layout that are unique to ECG.
"""

import jax
from jax import numpy as jnp

import mapox.envs.constance as GW
from mapox.envs.embodied_comm import EmbodiedCommConfig, EmbodiedCommEnv


LENGTH = 32


def _make_env(**overrides):
    return EmbodiedCommEnv(EmbodiedCommConfig(**overrides), LENGTH)


def test_reset_map_has_wall_divider():
    env = _make_env()
    state, _ = env.reset(jax.random.key(0))

    divider_x = env._agent1_origin_x + env.grid_size
    for dy in range(env.grid_size):
        assert state.map[divider_x, env._origin_y + dy] == GW.TILE_WALL


def test_reset_paints_both_sub_grids_with_color_tiles():
    env = _make_env()
    state, _ = env.reset(jax.random.key(0))

    for dx in range(env.grid_size):
        for dy in range(env.grid_size):
            tile_a = state.map[env._agent1_origin_x + dx, env._origin_y + dy]
            tile_b = state.map[env._agent2_origin_x + dx, env._origin_y + dy]
            assert GW.TILE_COLOR_0 <= int(tile_a) <= GW.TILE_COLOR_7
            assert GW.TILE_COLOR_0 <= int(tile_b) <= GW.TILE_COLOR_7


def test_reset_agents_start_in_their_own_sub_grid():
    env = _make_env()
    state, _ = env.reset(jax.random.key(7))

    x0, y0 = state.agents_pos[0]
    x1, y1 = state.agents_pos[1]

    assert env._agent1_origin_x <= int(x0) < env._agent1_origin_x + env.grid_size
    assert env._origin_y <= int(y0) < env._origin_y + env.grid_size

    assert env._agent2_origin_x <= int(x1) < env._agent2_origin_x + env.grid_size
    assert env._origin_y <= int(y1) < env._origin_y + env.grid_size


def test_reset_each_agents_colors_are_distinct():
    env = _make_env()
    state, _ = env.reset(jax.random.key(3))

    for origin_x in (env._agent1_origin_x, env._agent2_origin_x):
        tiles = [
            int(state.map[origin_x + dx, env._origin_y + dy])
            for dx in range(env.grid_size)
            for dy in range(env.grid_size)
        ]
        assert len(set(tiles)) == env.grid_size ** 2


def test_wall_blocks_crossing_into_other_sub_grid():
    env = _make_env()
    state, _ = env.reset(jax.random.key(0))

    rightmost_agent1_x = env._agent1_origin_x + env.grid_size - 1
    state = state._replace(
        agents_pos=state.agents_pos.at[0].set(
            jnp.array([rightmost_agent1_x, env._origin_y], dtype=jnp.int32)
        )
    )

    actions = jnp.array([GW.MOVE_RIGHT, GW.STAY], dtype=jnp.int32)
    new_state, _ = env.step(state, actions, jax.random.key(0))

    assert int(new_state.agents_pos[0, 0]) == rightmost_agent1_x


def test_commit_locks_agent_in_place():
    env = _make_env()
    state, _ = env.reset(jax.random.key(0))

    actions = jnp.array([GW.PRIMARY_ACTION, GW.STAY], dtype=jnp.int32)
    state, _ = env.step(state, actions, jax.random.key(0))

    pos_after_commit = state.agents_pos[0]
    assert bool(state.committed[0])

    actions = jnp.array([GW.MOVE_RIGHT, GW.STAY], dtype=jnp.int32)
    state, _ = env.step(state, actions, jax.random.key(0))

    assert jnp.array_equal(state.agents_pos[0], pos_after_commit)


def _force_matching_state(env, matching_color=3):
    state, _ = env.reset(jax.random.key(0))

    tile_id = jnp.int8(GW.TILE_COLOR_0 + matching_color)
    new_map = state.map.at[env._agent1_origin_x, env._origin_y].set(tile_id)
    new_map = new_map.at[env._agent2_origin_x, env._origin_y].set(tile_id)

    agents_pos = jnp.stack(
        [
            jnp.array([env._agent1_origin_x, env._origin_y], dtype=jnp.int32),
            jnp.array([env._agent2_origin_x, env._origin_y], dtype=jnp.int32),
        ],
        axis=0,
    )

    return state._replace(map=new_map, agents_pos=agents_pos)


def test_joint_commit_on_matching_colors_rewards_both_agents():
    env = _make_env()
    state = _force_matching_state(env)

    actions = jnp.array([GW.PRIMARY_ACTION, GW.PRIMARY_ACTION], dtype=jnp.int32)
    _, ts = env.step(state, actions, jax.random.key(0))

    assert float(ts.reward[0]) == 1.0
    assert float(ts.reward[1]) == 1.0
    assert bool(jnp.all(ts.terminated))


def test_joint_commit_on_mismatched_colors_gives_zero_reward():
    env = _make_env()
    state, _ = env.reset(jax.random.key(0))

    new_map = state.map.at[env._agent1_origin_x, env._origin_y].set(
        jnp.int8(GW.TILE_COLOR_0)
    )
    new_map = new_map.at[env._agent2_origin_x, env._origin_y].set(
        jnp.int8(GW.TILE_COLOR_1)
    )

    state = state._replace(
        map=new_map,
        agents_pos=jnp.stack(
            [
                jnp.array([env._agent1_origin_x, env._origin_y], dtype=jnp.int32),
                jnp.array([env._agent2_origin_x, env._origin_y], dtype=jnp.int32),
            ],
            axis=0,
        ),
    )

    actions = jnp.array([GW.PRIMARY_ACTION, GW.PRIMARY_ACTION], dtype=jnp.int32)
    _, ts = env.step(state, actions, jax.random.key(0))

    assert float(ts.reward[0]) == 0.0
    assert float(ts.reward[1]) == 0.0
    assert bool(jnp.all(ts.terminated))


def test_single_commit_does_not_terminate():
    env = _make_env()
    state, _ = env.reset(jax.random.key(0))

    actions = jnp.array([GW.PRIMARY_ACTION, GW.STAY], dtype=jnp.int32)
    new_state, ts = env.step(state, actions, jax.random.key(0))

    assert bool(new_state.committed[0])
    assert not bool(new_state.committed[1])
    assert not bool(jnp.any(ts.terminated))
    assert float(ts.reward[0]) == 0.0


def test_timeout_penalty_on_last_step_without_commit():
    env = _make_env()
    state, _ = env.reset(jax.random.key(0))
    state = state._replace(time=jnp.int32(LENGTH - 2))

    actions = jnp.array([GW.STAY, GW.STAY], dtype=jnp.int32)
    _, ts = env.step(state, actions, jax.random.key(0))

    assert float(ts.reward[0]) == -1.0
    assert float(ts.reward[1]) == -1.0
    assert bool(jnp.all(ts.terminated))


def test_env_step_is_jittable():
    env = _make_env()
    state, _ = env.reset(jax.random.key(0))

    jit_step = jax.jit(env.step)
    actions = jnp.array([GW.MOVE_UP, GW.MOVE_RIGHT], dtype=jnp.int32)
    new_state, ts = jit_step(state, actions, jax.random.key(1))

    assert ts.obs.shape == (env.num_agents, *env.observation_spec.shape)
    assert int(new_state.time) == 1


def test_auto_reset_clears_state_after_terminal_step():
    env = _make_env()
    state = _force_matching_state(env)

    actions = jnp.array([GW.PRIMARY_ACTION, GW.PRIMARY_ACTION], dtype=jnp.int32)
    new_state, ts = env.step(state, actions, jax.random.key(1))

    assert float(ts.reward[0]) == 1.0
    assert bool(jnp.all(ts.terminated))
    assert not bool(jnp.any(new_state.committed))
    assert int(new_state.time) == 0
    assert int(jnp.all(ts.time == 0))


def test_auto_reset_gives_fresh_map_on_next_step():
    env = _make_env()
    state = _force_matching_state(env)

    actions = jnp.array([GW.PRIMARY_ACTION, GW.PRIMARY_ACTION], dtype=jnp.int32)
    reset_state, _ = env.step(state, actions, jax.random.key(42))

    next_actions = jnp.array([GW.MOVE_UP, GW.MOVE_UP], dtype=jnp.int32)
    next_state, next_ts = env.step(reset_state, next_actions, jax.random.key(43))

    assert not bool(jnp.any(next_state.committed))
    assert int(next_state.time) == 1
    assert float(next_ts.reward[0]) == 0.0


def test_other_agents_colors_are_fogged_in_observation():
    env = _make_env()
    state, _ = env.reset(jax.random.key(0))

    state = state._replace(
        agents_pos=jnp.stack(
            [
                jnp.array([env._agent1_origin_x, env._origin_y], dtype=jnp.int32),
                jnp.array([env._agent2_origin_x, env._origin_y], dtype=jnp.int32),
            ],
            axis=0,
        ),
    )

    actions = jnp.array([GW.STAY, GW.STAY], dtype=jnp.int32)
    _, ts = env.step(state, actions, jax.random.key(0))

    tile_channel = ts.obs[..., 0]
    num_color_cells = env.grid_size ** 2

    def is_color(x):
        return (x >= GW.TILE_COLOR_0) & (x <= GW.TILE_COLOR_7)

    for agent_idx in range(env.num_agents):
        view = tile_channel[agent_idx]
        num_colors = int(jnp.sum(is_color(view)))
        num_grass = int(jnp.sum(view == GW.TILE_GRASS))
        num_agents_visible = int(jnp.sum(view == GW.AGENT_GENERIC))

        assert num_colors == num_color_cells - 1
        assert num_grass == num_color_cells - 1
        assert num_agents_visible == 2


def test_human_render_state_is_not_fogged():
    env = _make_env()
    state, _ = env.reset(jax.random.key(0))

    render = env.get_render_state(state)
    tile_channel = render.tilemap[..., 0]

    def is_color(x):
        return (x >= GW.TILE_COLOR_0) & (x <= GW.TILE_COLOR_7)

    total_color_cells = 2 * env.grid_size ** 2
    num_colors = int(jnp.sum(is_color(tile_channel)))
    num_grass = int(jnp.sum(tile_channel == GW.TILE_GRASS))

    assert num_colors == total_color_cells - 2
    assert num_grass == 0


def test_auto_reset_no_effect_when_not_terminal():
    env = _make_env()
    state, _ = env.reset(jax.random.key(0))

    actions = jnp.array([GW.MOVE_UP, GW.STAY], dtype=jnp.int32)
    new_state, ts = env.step(state, actions, jax.random.key(1))

    assert int(new_state.time) == 1
    assert not bool(jnp.any(ts.terminated))


def test_env_vmaps_over_seeds():
    env = _make_env()
    keys = jax.random.split(jax.random.key(0), 8)

    states, timesteps = jax.vmap(env.reset)(keys)

    assert states.agents_pos.shape == (8, env.num_agents, 2)
    assert timesteps.obs.shape == (
        8,
        env.num_agents,
        *env.observation_spec.shape,
    )
