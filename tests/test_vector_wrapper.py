"""VectorWrapper — flattening (vec_count, agents) must be correct."""

import jax
from jax import numpy as jnp

from mapox.envs.find_return import FindReturnConfig, FindReturnEnv
from mapox.envs.constance import MOVE_UP
from mapox.wrappers.vector import VectorWrapper

LENGTH = 32
VEC_COUNT = 3


def _make_find_return():
    return FindReturnEnv(FindReturnConfig(num_agents=2, num_flags=2), LENGTH)


def test_num_agents():
    inner = _make_find_return()
    venv = VectorWrapper(inner, VEC_COUNT)
    assert venv.num_agents == VEC_COUNT * inner.num_agents


def test_reset_flattens():
    inner = _make_find_return()
    venv = VectorWrapper(inner, VEC_COUNT)
    key = jax.random.key(0)

    _, ts = jax.jit(venv.reset)(key)
    assert ts.obs.shape[0] == venv.num_agents
    assert ts.obs.shape == (venv.num_agents, *inner.observation_spec.shape)
    assert ts.reward.shape == (venv.num_agents,)
    assert ts.action_mask.shape == (venv.num_agents, inner.action_spec.n)
    assert ts.terminated.shape == (venv.num_agents,)


def test_step_works():
    inner = _make_find_return()
    venv = VectorWrapper(inner, VEC_COUNT)
    k1, k2 = jax.random.split(jax.random.key(0))

    state, _ = jax.jit(venv.reset)(k1)
    actions = jnp.full((venv.num_agents,), MOVE_UP, dtype=jnp.int32)
    _, ts = jax.jit(venv.step)(state, actions, k2)

    assert ts.obs.shape == (venv.num_agents, *inner.observation_spec.shape)
    assert ts.reward.shape == (venv.num_agents,)


def test_teams_none():
    inner = _make_find_return()
    venv = VectorWrapper(inner, VEC_COUNT)
    assert venv.teams is None
