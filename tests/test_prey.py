"""Prey mechanics — movement, fullness gating, catches, starvation, food
lifecycle, and grass concealment.

States are crafted directly (PreyState is a NamedTuple) so every scenario is
deterministic. Coordinates are in padded space: with a 16x16 board and an
11x11 view the pad is 5, so playable cells span 5..20 on both axes.

Assertions avoid anything tied to the movement-resolution order or to how
many rng keys step() consumes, so refactors that reshuffle randomness don't
invalidate the scenarios.
"""

import jax
from jax import numpy as jnp
import pytest

from mapox.envs.prey import PreyConfig, PreyEnv

# Local action ids: prey registers SB.MOVES via add_block (asserted to be
# 0..3 in __init__) and STAY immediately after.
MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT = range(4)
STAY = 4

LENGTH = 128

# Out-of-the-way cells for agents a scenario doesn't care about.
FAR_A = (18, 18)
FAR_B = (5, 18)


@pytest.fixture
def env():
    return PreyEnv(
        PreyConfig(num_sneakers=2, num_chasers=1, width=16, height=16, num_food=2),
        LENGTH,
    )


@pytest.fixture
def rng_key():
    return jax.random.key(0)


def make_state(
    env,
    rng_key,
    sneakers,
    chasers,
    food_cells=(),
    grass_cells=(),
    fullness=None,
    food_timer=None,
):
    """Build a state on a flat map: empty interior, wall border, no decor.

    sneakers/chasers: (x, y) cells in padded coordinates. fullness is per
    agent, sneakers first; defaults to initial_fullness. food_cells fill the
    first food slots (timer 0 = present); remaining slots are parked in the
    far corner where no scenario walks.
    """
    state, _ = env.reset(rng_key)
    cfg = env._config

    tiles = jnp.full(
        (env.padded_width, env.padded_height), env._tile_wall, dtype=jnp.uint16
    )
    tiles = tiles.at[
        env.pad_width : env.pad_width + cfg.width,
        env.pad_height : env.pad_height + cfg.height,
    ].set(env._tile_empty)
    for x, y in grass_cells:
        tiles = tiles.at[x, y].set(env._tile_grass)

    parked = (env.pad_width + cfg.width - 1, env.pad_height + cfg.height - 1)
    cells = list(food_cells) + [parked] * (cfg.num_food - len(food_cells))

    if fullness is None:
        fullness = [cfg.initial_fullness] * env.num_agents
    if food_timer is None:
        food_timer = [0] * cfg.num_food

    return state._replace(
        sneaker_pos=jnp.array(sneakers, dtype=jnp.int32),
        chaser_pos=jnp.array(chasers, dtype=jnp.int32),
        tiles=tiles,
        food_pos=jnp.array(cells, dtype=jnp.int32),
        food_timer=jnp.array(food_timer, dtype=jnp.int32),
        fullness=jnp.array(fullness, dtype=jnp.int32),
    )


# --- Movement ---


def test_moves_follow_directions(env, rng_key):
    state = make_state(env, rng_key, sneakers=[(8, 8), FAR_A], chasers=[FAR_B])
    actions = jnp.array([MOVE_RIGHT, MOVE_UP, STAY], dtype=jnp.uint16)

    state, _ = env.step(state, actions, rng_key)

    assert state.sneaker_pos[0].tolist() == [9, 8]
    assert state.sneaker_pos[1].tolist() == [18, 19]
    assert state.chaser_pos[0].tolist() == list(FAR_B)


def test_wall_blocks_movement(env, rng_key):
    # (5, 8) is on the west edge of the playable area; west of it is border.
    state = make_state(env, rng_key, sneakers=[(5, 8), FAR_A], chasers=[FAR_B])
    actions = jnp.array([MOVE_LEFT, STAY, STAY], dtype=jnp.uint16)

    state, _ = env.step(state, actions, rng_key)

    assert state.sneaker_pos[0].tolist() == [5, 8]


def test_agent_blocks_movement(env, rng_key):
    # Sneaker 1 stays, so its tile is occupied whenever sneaker 0 resolves,
    # regardless of the random execution order.
    state = make_state(env, rng_key, sneakers=[(8, 8), (9, 8)], chasers=[FAR_B])
    actions = jnp.array([MOVE_RIGHT, STAY, STAY], dtype=jnp.uint16)

    state, _ = env.step(state, actions, rng_key)

    assert state.sneaker_pos[0].tolist() == [8, 8]
    assert state.sneaker_pos[1].tolist() == [9, 8]


# --- Food and fullness ---


def test_eating_food_gains_fullness(env, rng_key):
    state = make_state(
        env,
        rng_key,
        sneakers=[(8, 8), FAR_A],
        chasers=[FAR_B],
        food_cells=[(9, 8)],
    )
    actions = jnp.array([MOVE_RIGHT, STAY, STAY], dtype=jnp.uint16)

    state, _ = env.step(state, actions, rng_key)

    cfg = env._config
    # Fullness decrements 1 each step, then the food adds fullness_per_food.
    assert state.fullness[0] == cfg.initial_fullness - 1 + cfg.fullness_per_food
    assert state.fullness[1] == cfg.initial_fullness - 1
    # Eaten food starts regrowing (timer was set, then ticked once).
    assert state.food_timer[0] == cfg.food_regrow_time - 1


def test_full_sneaker_does_not_eat(env, rng_key):
    cfg = env._config
    state = make_state(
        env,
        rng_key,
        sneakers=[(8, 8), FAR_A],
        chasers=[FAR_B],
        food_cells=[(9, 8)],
        fullness=[cfg.full_threshold + 10, cfg.initial_fullness, cfg.initial_fullness],
    )
    actions = jnp.array([MOVE_RIGHT, STAY, STAY], dtype=jnp.uint16)

    state, _ = env.step(state, actions, rng_key)

    assert state.fullness[0] == cfg.full_threshold + 9  # decrement only
    assert state.food_timer[0] == 0  # food untouched


def test_food_regrows_and_respawns(env, rng_key):
    # Slot 0 is mid-regrow (timer 2). It should tick to 1, then respawn as
    # present on the following step.
    state = make_state(
        env,
        rng_key,
        sneakers=[(16, 16), FAR_A],
        chasers=[(16, 18)],
        food_cells=[(9, 8)],
        food_timer=[2, 0],
    )
    actions = jnp.array([STAY, STAY, STAY], dtype=jnp.uint16)
    k1, k2 = jax.random.split(rng_key)

    state, _ = env.step(state, actions, k1)
    assert state.food_timer[0] == 1

    state, _ = env.step(state, actions, k2)
    assert state.food_timer[0] == 0  # present again
    x, y = state.food_pos[0].tolist()
    assert state.tiles[x, y] != env._tile_wall  # respawned on an open tile


# --- Catching ---


def test_catch_terminates_sneaker_and_feeds_chaser(env, rng_key):
    cfg = env._config
    state = make_state(env, rng_key, sneakers=[(8, 8), FAR_A], chasers=[(10, 8)])
    actions = jnp.array([STAY, STAY, MOVE_LEFT], dtype=jnp.uint16)

    state, ts = env.step(state, actions, rng_key)

    assert ts.terminated.tolist() == [True, False, False]
    assert state.fullness[2] == cfg.initial_fullness - 1 + cfg.fullness_per_catch
    # Caught sneaker respawns with fresh fullness and no reward this step.
    assert state.fullness[0] == cfg.initial_fullness
    assert float(ts.reward[0]) == 0.0


def test_full_chaser_cannot_catch(env, rng_key):
    cfg = env._config
    state = make_state(
        env,
        rng_key,
        sneakers=[(8, 8), FAR_A],
        chasers=[(9, 8)],  # already adjacent
        fullness=[
            cfg.initial_fullness,
            cfg.initial_fullness,
            cfg.full_threshold + 10,
        ],
    )
    actions = jnp.array([STAY, STAY, STAY], dtype=jnp.uint16)

    state, ts = env.step(state, actions, rng_key)

    assert not ts.terminated.any()
    assert state.fullness[2] == cfg.full_threshold + 9  # decrement, no gain


# --- Starvation and rewards ---


def test_starvation_terminates_and_respawns(env, rng_key):
    cfg = env._config
    state = make_state(
        env,
        rng_key,
        sneakers=[(8, 8), FAR_A],
        chasers=[FAR_B],
        fullness=[1, cfg.initial_fullness, cfg.initial_fullness],
    )
    actions = jnp.array([STAY, STAY, STAY], dtype=jnp.uint16)

    state, ts = env.step(state, actions, rng_key)

    assert ts.terminated.tolist() == [True, False, False]
    assert state.fullness[0] == cfg.initial_fullness
    assert float(ts.reward[0]) == 0.0


def test_survival_reward_scales_with_fullness(env, rng_key):
    cfg = env._config
    state = make_state(env, rng_key, sneakers=[(8, 8), FAR_A], chasers=[FAR_B])
    actions = jnp.array([STAY, STAY, STAY], dtype=jnp.uint16)

    _, ts = env.step(state, actions, rng_key)

    expected = cfg.survival_reward_scale * (cfg.initial_fullness - 1) / cfg.max_fullness
    for r in ts.reward.tolist():
        assert r == pytest.approx(expected, rel=1e-5)


def test_time_up_terminates_everyone(rng_key):
    env = PreyEnv(
        PreyConfig(num_sneakers=2, num_chasers=1, width=16, height=16, num_food=2),
        length=1,
    )
    state, _ = env.reset(rng_key)
    actions = jnp.array([STAY, STAY, STAY], dtype=jnp.uint16)

    _, ts = env.step(state, actions, rng_key)

    assert ts.terminated.all()


# --- Concealment ---


def _observe(env, state):
    zeros_i = jnp.zeros(env.num_agents, dtype=jnp.uint16)
    zeros_f = jnp.zeros(env.num_agents, dtype=jnp.float32)
    zeros_b = jnp.zeros(env.num_agents, dtype=jnp.bool_)
    return env.encode_observations(state, zeros_i, zeros_f, zeros_b)


def test_agent_on_grass_is_concealed(env, rng_key):
    # Sneaker 0 stands on grass; sneaker 1 stands in the open. The chaser at
    # (13, 8) sees both cells: its 11x11 view spans x 8..18, y 3..13.
    state = make_state(
        env,
        rng_key,
        sneakers=[(9, 8), (11, 8)],
        chasers=[(13, 8)],
        grass_cells=[(9, 8)],
    )
    ts = _observe(env, state)
    chaser_view = ts.obs[2]

    # Concealed sneaker: the cell reads as plain grass, health channel empty.
    assert chaser_view[1, 5, 0] == env._tile_grass
    assert chaser_view[1, 5, 3] == 0

    # Open-ground sneaker is visible, with its fullness on the health channel.
    assert chaser_view[3, 5, 0] == env._agent_sneaker
    assert chaser_view[3, 5, 3] == 1  # initial fullness == half => low

    # And the chaser is visible to the sneakers.
    sneaker_view = ts.obs[0]
    assert sneaker_view[9, 5, 0] == env._agent_chaser


def test_concealment_hides_agent_from_its_own_view(env, rng_key):
    # Documents a quirk: conceal applies to every view, so an agent on grass
    # does not see itself at the center of its own observation.
    state = make_state(
        env,
        rng_key,
        sneakers=[(9, 8), FAR_A],
        chasers=[FAR_B],
        grass_cells=[(9, 8)],
    )
    ts = _observe(env, state)

    assert ts.obs[0][5, 5, 0] == env._tile_grass


# --- Rollout invariants ---


def test_rollout_invariants(env):
    key = jax.random.key(1)
    state, _ = env.reset(key)
    step = jax.jit(env.step)
    cfg = env._config

    for _ in range(128):
        key, akey, skey = jax.random.split(key, 3)
        actions = jax.random.randint(
            akey, (env.num_agents,), 0, env.action_spec.n, dtype=jnp.uint16
        )
        state, ts = step(state, actions, skey)

        all_pos = jnp.concatenate([state.sneaker_pos, state.chaser_pos], axis=0)
        agent_tiles = state.tiles[all_pos[:, 0], all_pos[:, 1]]
        assert (agent_tiles != env._tile_wall).all(), "agent standing in a wall"

        food_tiles = state.tiles[state.food_pos[:, 0], state.food_pos[:, 1]]
        assert (food_tiles != env._tile_wall).all(), "food inside a wall"

        assert (state.fullness > 0).all() and (state.fullness <= cfg.max_fullness).all()
        assert (state.food_timer >= 0).all()
        assert (state.food_timer <= cfg.food_regrow_time).all()


@pytest.mark.xfail(
    strict=True,
    reason="_spawn samples open tiles with replacement, so agents can spawn "
    "on top of each other (both at reset and on respawn)",
)
def test_spawned_agents_never_overlap():
    env = PreyEnv(
        PreyConfig(num_sneakers=6, num_chasers=4, width=16, height=16, num_food=2),
        LENGTH,
    )
    for seed in range(30):
        state, _ = env.reset(jax.random.key(seed))
        pos = jnp.concatenate([state.sneaker_pos, state.chaser_pos], axis=0)
        unique = {tuple(p) for p in pos.tolist()}
        assert len(unique) == env.num_agents, f"overlap at reset, seed {seed}"
