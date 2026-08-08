import pytest
from jax import numpy as jnp

from mapox.envs.common import make_action_mask, make_obs_spec


def test_make_obs_spec():
    spec = make_obs_spec(5, 7, 21)
    assert spec.shape == (5, 7, 4)
    assert spec.dtype == jnp.uint16
    assert spec.max_value[0] == 21


def test_make_obs_spec_enforces_uint16_ceiling():
    make_obs_spec(5, 5, 1 << 16)
    with pytest.raises(ValueError, match="65537"):
        make_obs_spec(5, 5, (1 << 16) + 1)


def test_make_action_mask():
    mask = make_action_mask([0, 2], num_actions=4, num_agents=3)
    assert mask.shape == (3, 4)
    assert mask.dtype == jnp.bool_
    assert mask.tolist() == [[True, False, True, False]] * 3
