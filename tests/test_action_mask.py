"""make_action_mask — shape, values, and dtype."""

from jax import numpy as jnp

from mapox.envs.constance import (
    make_action_mask,
    NUM_ACTIONS,
    MOVE_UP,
    MOVE_DOWN,
    MOVE_RIGHT,
    STAY,
)


def test_shape():
    mask = make_action_mask([MOVE_UP, MOVE_DOWN, STAY], num_agents=3)
    assert mask.shape == (3, NUM_ACTIONS)


def test_values():
    mask = make_action_mask([MOVE_UP, STAY], num_agents=2)

    assert jnp.all(mask[:, MOVE_UP])
    assert jnp.all(mask[:, STAY])
    assert not jnp.any(mask[:, MOVE_RIGHT])
    assert not jnp.any(mask[:, MOVE_DOWN])


def test_dtype():
    mask = make_action_mask([MOVE_UP], num_agents=1)
    assert mask.dtype == jnp.bool_
