"""Snake mechanics — movement, growth, collisions, and corpse food drops.

States are crafted directly (SnakeState is a NamedTuple) so every scenario is
deterministic. Coordinates are in padded space: with a 12x12 board and an 11x11
view the pad is 5, so playable cells span 5..16 on both axes.
"""

import jax
from jax import numpy as jnp
import pytest

import mapox.symbols as SB
from mapox.envs.snake import SnakeConfig, SnakeEnv

# Local action ids: snake registers SB.MOVES via add_block, so the moves are
# always 0..3 and they are the whole action space.
MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT = range(4)
NON_MOVE = 4  # out-of-vocab id: snake coerces any non-move input to straight

LENGTH = 128


@pytest.fixture
def env():
    return SnakeEnv(
        SnakeConfig(
            num_agents=2,
            width=12,
            height=12,
            initial_food=0,
            food_spawn_prob=0.0,
        ),
        LENGTH,
    )


@pytest.fixture
def rng_key():
    return jax.random.key(0)


def make_state(env, rng_key, snakes, food_cells=()):
    """Build a state with exact snake placements.

    snakes: per agent, a (cells, direction) pair with cells ordered tail
    first and head last; consecutive cells must be adjacent. Cells fill the
    ring buffer in order, so the head lands at slot len(cells) - 1.
    """
    state, _ = env.reset(rng_key)

    body = jnp.zeros_like(state.body)
    head_slot = []
    head_pos = []
    direction = []
    length = []
    for i, (cells, d) in enumerate(snakes):
        for slot, (x, y) in enumerate(cells):
            body = body.at[i, slot].set(jnp.array([x, y], dtype=jnp.int32))
        head_slot.append(len(cells) - 1)
        head_pos.append(cells[-1])
        direction.append(d)
        length.append(len(cells))

    food = jnp.zeros_like(state.food)
    for x, y in food_cells:
        food = food.at[x, y].set(True)

    return state._replace(
        body=body,
        head_slot=jnp.array(head_slot, dtype=jnp.int32),
        head_pos=jnp.array(head_pos, dtype=jnp.int32),
        direction=jnp.array(direction, dtype=jnp.int32),
        length=jnp.array(length, dtype=jnp.int32),
        food=food,
    )


def occupied_cells(env, state, agent):
    """The set of cells the agent's body covers, read from the ring buffer."""
    max_length = env._config.max_length
    offsets = (state.head_slot[agent] - jnp.arange(max_length)) % max_length
    alive = offsets < state.length[agent]
    return {
        tuple(cell)
        for cell, live in zip(state.body[agent].tolist(), alive.tolist())
        if live
    }


FAR_SNAKE = ([(15, 15)], MOVE_UP)


def test_moves_forward(env, rng_key):
    state = make_state(env, rng_key, [([(8, 8), (9, 8)], MOVE_RIGHT), FAR_SNAKE])
    actions = jnp.array([MOVE_RIGHT, MOVE_UP], dtype=jnp.int32)

    state, ts = env.step(state, actions, rng_key)

    assert jnp.array_equal(state.head_pos[0], jnp.array([10, 8]))
    assert not ts.terminated[0]
    assert state.length[0] == 2
    # Tail vacated, body followed the head.
    assert occupied_cells(env, state, 0) == {(9, 8), (10, 8)}


def test_reverse_is_coerced_straight(env, rng_key):
    state = make_state(env, rng_key, [([(8, 8), (9, 8)], MOVE_RIGHT), FAR_SNAKE])
    actions = jnp.array([MOVE_LEFT, MOVE_UP], dtype=jnp.int32)

    state, ts = env.step(state, actions, rng_key)

    assert jnp.array_equal(state.head_pos[0], jnp.array([10, 8]))
    assert state.direction[0] == MOVE_RIGHT
    assert not ts.terminated[0]


def test_non_move_is_coerced_straight(env, rng_key):
    state = make_state(env, rng_key, [([(8, 8), (9, 8)], MOVE_RIGHT), FAR_SNAKE])
    actions = jnp.array([NON_MOVE, MOVE_UP], dtype=jnp.int32)

    state, _ = env.step(state, actions, rng_key)

    assert jnp.array_equal(state.head_pos[0], jnp.array([10, 8]))


def test_eating_grows(env, rng_key):
    state = make_state(
        env,
        rng_key,
        [([(8, 8), (9, 8)], MOVE_RIGHT), FAR_SNAKE],
        food_cells=[(10, 8)],
    )
    actions = jnp.array([MOVE_RIGHT, MOVE_UP], dtype=jnp.int32)

    state, ts = env.step(state, actions, rng_key)

    assert state.length[0] == 3
    assert ts.reward[0] == env._config.food_reward
    assert not state.food[10, 8]
    # Tail did not move: the snake occupies three cells now.
    assert occupied_cells(env, state, 0) == {(8, 8), (9, 8), (10, 8)}


def test_growth_caps_at_max_length(rng_key):
    env = SnakeEnv(
        SnakeConfig(
            num_agents=2,
            width=12,
            height=12,
            initial_length=3,
            max_length=3,
            initial_food=0,
            food_spawn_prob=0.0,
        ),
        LENGTH,
    )
    state = make_state(
        env,
        rng_key,
        [([(7, 8), (8, 8), (9, 8)], MOVE_RIGHT), FAR_SNAKE],
        food_cells=[(10, 8)],
    )
    actions = jnp.array([MOVE_RIGHT, MOVE_UP], dtype=jnp.int32)

    state, ts = env.step(state, actions, rng_key)

    # The pellet is consumed and rewarded, but the tail vacates as usual.
    assert state.length[0] == 3
    assert ts.reward[0] == env._config.food_reward
    assert not state.food[10, 8]
    assert occupied_cells(env, state, 0) == {(8, 8), (9, 8), (10, 8)}


def test_ring_buffer_wraps(rng_key):
    env = SnakeEnv(
        SnakeConfig(
            num_agents=2,
            width=12,
            height=12,
            initial_length=3,
            max_length=3,
            initial_food=0,
            food_spawn_prob=0.0,
        ),
        LENGTH,
    )
    state = make_state(
        env, rng_key, [([(6, 8), (7, 8), (8, 8)], MOVE_RIGHT), FAR_SNAKE]
    )
    actions = jnp.array([MOVE_RIGHT, MOVE_UP], dtype=jnp.int32)

    # A full-length snake overwrites its own tail slot every step; four steps
    # cycle through the 3-slot buffer more than once.
    for i in range(4):
        state, ts = env.step(state, actions, jax.random.fold_in(rng_key, i))
        assert not ts.terminated[0]
        head_x = 9 + i
        expected = {(head_x - 2, 8), (head_x - 1, 8), (head_x, 8)}
        assert occupied_cells(env, state, 0) == expected


def test_wall_death_drops_food(env, rng_key):
    # Playable x spans 5..16, so moving right from 16 hits the wall.
    state = make_state(env, rng_key, [([(15, 8), (16, 8)], MOVE_RIGHT), FAR_SNAKE])
    actions = jnp.array([MOVE_RIGHT, MOVE_UP], dtype=jnp.int32)

    state, ts = env.step(state, actions, rng_key)

    assert ts.terminated[0]
    assert ts.reward[0] == env._config.death_reward
    # Alternating segments from the tail become food (here just the tail).
    assert state.food[15, 8]
    assert not state.food[16, 8]
    # Respawned fresh somewhere else.
    assert state.length[0] == env._config.initial_length
    assert not jnp.array_equal(state.head_pos[0], jnp.array([17, 8]))


def test_body_collision_kills(env, rng_key):
    state = make_state(
        env,
        rng_key,
        [
            ([(8, 8), (9, 8)], MOVE_RIGHT),
            ([(10, 10), (10, 9), (10, 8)], MOVE_DOWN),
        ],
    )
    actions = jnp.array([MOVE_RIGHT, MOVE_DOWN], dtype=jnp.int32)

    state, ts = env.step(state, actions, rng_key)

    assert ts.terminated[0]
    assert not ts.terminated[1]


def test_head_on_collision_kills_both(env, rng_key):
    state = make_state(
        env,
        rng_key,
        [
            ([(8, 8), (9, 8)], MOVE_RIGHT),
            ([(12, 8), (11, 8)], MOVE_LEFT),
        ],
    )
    actions = jnp.array([MOVE_RIGHT, MOVE_LEFT], dtype=jnp.int32)

    state, ts = env.step(state, actions, rng_key)

    assert ts.terminated[0]
    assert ts.terminated[1]
    # Both respawned on distinct open cells.
    assert not jnp.array_equal(state.head_pos[0], state.head_pos[1])
    assert jnp.all(state.length == env._config.initial_length)


def test_swap_collision_kills_both(env, rng_key):
    # Length-1 snakes passing through each other: only the swap rule catches
    # this, since both cells are vacated this step.
    state = make_state(
        env,
        rng_key,
        [
            ([(9, 8)], MOVE_RIGHT),
            ([(10, 8)], MOVE_LEFT),
        ],
    )
    actions = jnp.array([MOVE_RIGHT, MOVE_LEFT], dtype=jnp.int32)

    _, ts = env.step(state, actions, rng_key)

    assert ts.terminated[0]
    assert ts.terminated[1]


def test_can_follow_own_tail(env, rng_key):
    # A 4-cell snake looping through a 2x2 block: the head enters the tail
    # cell exactly as it vacates.
    state = make_state(
        env,
        rng_key,
        [([(8, 8), (8, 9), (9, 9), (9, 8)], MOVE_DOWN), FAR_SNAKE],
    )
    actions = jnp.array([MOVE_LEFT, MOVE_UP], dtype=jnp.int32)

    state, ts = env.step(state, actions, rng_key)

    assert not ts.terminated[0]
    assert jnp.array_equal(state.head_pos[0], jnp.array([8, 8]))


def test_own_body_collision_kills(env, rng_key):
    # Same loop but one cell longer: the target cell is now the neck, which
    # does not vacate this step.
    state = make_state(
        env,
        rng_key,
        [([(8, 7), (8, 8), (8, 9), (9, 9), (9, 8)], MOVE_DOWN), FAR_SNAKE],
    )
    actions = jnp.array([MOVE_LEFT, MOVE_UP], dtype=jnp.int32)

    _, ts = env.step(state, actions, rng_key)

    assert ts.terminated[0]


def test_action_mask_blocks_reverse(env, rng_key):
    state = make_state(env, rng_key, [([(8, 8), (9, 8)], MOVE_RIGHT), FAR_SNAKE])
    actions = jnp.array([MOVE_RIGHT, MOVE_UP], dtype=jnp.int32)

    _, ts = env.step(state, actions, rng_key)

    assert not ts.action_mask[0, MOVE_LEFT]
    assert ts.action_mask[0, MOVE_UP]
    assert ts.action_mask[0, MOVE_RIGHT]
    assert ts.action_mask[0, MOVE_DOWN]
    # The action space is only the four moves.
    assert ts.action_mask.shape[1] == 4


def test_observation_is_egocentric(env, rng_key):
    state = make_state(
        env,
        rng_key,
        [
            ([(8, 8), (9, 8)], MOVE_RIGHT),
            ([(12, 8), (11, 8)], MOVE_LEFT),
        ],
    )
    zeros_i = jnp.zeros(env.num_agents, dtype=jnp.int32)
    zeros_f = jnp.zeros(env.num_agents, dtype=jnp.float32)
    zeros_b = jnp.zeros(env.num_agents, dtype=jnp.bool_)

    ts = env.encode_observations(state, zeros_i, zeros_f, zeros_b)

    cx, cy = env.view_width // 2, env.view_height // 2
    head = env.obs_vocab.id(SB.AGENT_SNAKE_HEAD)
    body = env.obs_vocab.id(SB.AGENT_SNAKE_BODY)
    # Own head at the view center, marked as team 1.
    assert ts.obs[0, cx, cy, 0] == head
    assert ts.obs[0, cx, cy, 2] == 1
    # Own body behind the head.
    assert ts.obs[0, cx - 1, cy, 0] == body
    assert ts.obs[0, cx - 1, cy, 2] == 1
    # The other snake's head, two cells ahead, is marked as team 2.
    assert ts.obs[0, cx + 2, cy, 0] == head
    assert ts.obs[0, cx + 2, cy, 2] == 2


def test_rollout_invariants(rng_key):
    # Fuzz the ring-buffer bookkeeping: random legal actions for many steps
    # (crossing a time-up reset), checking structural invariants every step.
    env = SnakeEnv(
        SnakeConfig(
            num_agents=4,
            width=12,
            height=12,
            initial_food=6,
            food_spawn_prob=0.3,
            max_length=8,
        ),
        LENGTH,
    )
    state, ts = env.reset(rng_key)

    for i in range(150):
        rng_key, act_key, step_key = jax.random.split(rng_key, 3)
        logits = jax.random.uniform(act_key, ts.action_mask.shape)
        actions = jnp.argmax(jnp.where(ts.action_mask, logits, -1.0), axis=-1)
        state, ts = env.step(state, actions.astype(jnp.int32), step_key)

        cells = [occupied_cells(env, state, a) for a in range(env.num_agents)]
        for a in range(env.num_agents):
            # Head slot, head position, and grid agree.
            assert tuple(state.body[a, state.head_slot[a]].tolist()) == tuple(
                state.head_pos[a].tolist()
            )
            assert tuple(state.head_pos[a].tolist()) in cells[a]
            # Heads never rest on walls.
            wall = env.obs_vocab.id(SB.TILE_WALL)
            assert state.tiles[state.head_pos[a, 0], state.head_pos[a, 1]] != wall
            # Consecutive segments are adjacent (or stacked, right after a
            # respawn while the body still grows out of the spawn cell).
            max_length = env._config.max_length
            slots = (state.head_slot[a] - jnp.arange(state.length[a])) % max_length
            segments = state.body[a, slots].tolist()
            for (x0, y0), (x1, y1) in zip(segments, segments[1:]):
                assert abs(x0 - x1) + abs(y0 - y1) <= 1
            # No two snakes overlap.
            for b in range(a + 1, env.num_agents):
                assert not (cells[a] & cells[b])


def test_food_spawns_over_time(rng_key):
    env = SnakeEnv(
        SnakeConfig(
            num_agents=2,
            width=12,
            height=12,
            initial_food=0,
            food_spawn_prob=1.0,
        ),
        LENGTH,
    )
    state, ts = env.reset(rng_key)
    assert jnp.sum(state.food) == 0

    for i in range(3):
        rng_key, step_key = jax.random.split(rng_key)
        actions = jnp.argmax(ts.action_mask, axis=-1).astype(jnp.int32)
        state, ts = env.step(state, actions, step_key)

    assert jnp.sum(state.food) >= 1
