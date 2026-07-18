"""make_action_mask — shape, values, and dtype."""

from jax import numpy as jnp

from mapox.envs.common import make_action_mask

NUM_ACTIONS = 6


def test_shape():
    mask = make_action_mask([0, 2, 4], NUM_ACTIONS, num_agents=3)
    assert mask.shape == (3, NUM_ACTIONS)


def test_values():
    mask = make_action_mask([0, 4], NUM_ACTIONS, num_agents=2)

    assert jnp.all(mask[:, 0])
    assert jnp.all(mask[:, 4])
    assert not jnp.any(mask[:, 1])
    assert not jnp.any(mask[:, 2])


def test_dtype():
    mask = make_action_mask([0], NUM_ACTIONS, num_agents=1)
    assert mask.dtype == jnp.bool_
