"""JIT compatibility — all envs must work under jax.jit."""

import jax
from jax import numpy as jnp

from mapox.envs.constance import MOVE_UP


def test_jit_reset(env, rng_key):
    state, ts = jax.jit(env.reset)(rng_key)
    assert ts.obs.shape == (env.num_agents, *env.observation_spec.shape)
    assert ts.reward.shape == (env.num_agents,)


def test_jit_step(env, rng_key):
    k1, k2 = jax.random.split(rng_key)
    state, _ = jax.jit(env.reset)(k1)

    actions = jnp.full((env.num_agents,), MOVE_UP, dtype=jnp.int32)
    _, ts = jax.jit(env.step)(state, actions, k2)

    assert ts.obs.shape == (env.num_agents, *env.observation_spec.shape)
    assert ts.reward.shape == (env.num_agents,)


def test_jit_deterministic(env, rng_key):
    jit_reset = jax.jit(env.reset)
    _, ts1 = jit_reset(rng_key)
    _, ts2 = jit_reset(rng_key)
    assert jnp.array_equal(ts1.obs, ts2.obs)
