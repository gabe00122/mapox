"""Shape contract tests — the most critical tests in the suite.

Every environment must produce correctly shaped outputs from reset() and step().
Training pipelines silently break if these shapes are wrong.
"""

import jax
from jax import numpy as jnp

from mapox.envs.constance import MOVE_UP


def test_reset_obs_shape(env, rng_key):
    _, ts = env.reset(rng_key)
    expected = (env.num_agents, *env.observation_spec.shape)
    assert ts.obs.shape == expected


def test_reset_action_mask_shape(env, rng_key):
    _, ts = env.reset(rng_key)
    assert ts.action_mask.shape == (env.num_agents, env.action_spec.n)


def test_reset_reward_shape(env, rng_key):
    _, ts = env.reset(rng_key)
    assert ts.reward.shape == (env.num_agents,)


def test_reset_terminated_shape(env, rng_key):
    _, ts = env.reset(rng_key)
    assert ts.terminated.shape == (env.num_agents,)


def test_reset_terminated_all_false(env, rng_key):
    _, ts = env.reset(rng_key)
    assert not jnp.any(ts.terminated)


def test_reset_time_zero(env, rng_key):
    _, ts = env.reset(rng_key)
    assert jnp.all(ts.time == 0)


def test_step_preserves_shapes(env, rng_key):
    k1, k2 = jax.random.split(rng_key)
    state, ts_reset = env.reset(k1)

    actions = jnp.full((env.num_agents,), MOVE_UP, dtype=jnp.int32)
    _, ts_step = env.step(state, actions, k2)

    assert ts_step.obs.shape == ts_reset.obs.shape
    assert ts_step.action_mask.shape == ts_reset.action_mask.shape
    assert ts_step.reward.shape == ts_reset.reward.shape
    assert ts_step.terminated.shape == ts_reset.terminated.shape


def test_step_time_increments(env, rng_key):
    k1, k2 = jax.random.split(rng_key)
    state, _ = env.reset(k1)

    actions = jnp.full((env.num_agents,), MOVE_UP, dtype=jnp.int32)
    _, ts = env.step(state, actions, k2)
    assert jnp.all(ts.time == 1)
