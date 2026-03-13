"""MultiTaskWrapper — action slicing and timestep concatenation."""

import jax
from jax import numpy as jnp

from mapox.envs.constance import MOVE_UP
from mapox.config import EnvironmentFactory, MultiTaskConfig, MultiTaskEnvConfig


LENGTH = 32


def _make_wrapper():
    config = MultiTaskConfig(
        envs=(
            MultiTaskEnvConfig(
                name="fr",
                env={"env_type": "find_return", "num_agents": 2, "num_flags": 2},
            ),
            MultiTaskEnvConfig(
                name="ts",
                env={"env_type": "traveling_salesman", "num_agents": 2, "num_flags": 3},
            ),
        ),
    )
    factory = EnvironmentFactory()
    wrapper, num_tasks = factory.create_env(config, LENGTH)
    return wrapper, num_tasks


def test_num_agents():
    wrapper, _ = _make_wrapper()
    assert wrapper.num_agents == 4


def test_num_tasks():
    _, num_tasks = _make_wrapper()
    assert num_tasks == 2


def test_reset_concatenates():
    wrapper, _ = _make_wrapper()
    key = jax.random.key(0)
    _, ts = wrapper.reset(key)

    assert ts.obs.shape[0] == wrapper.num_agents
    assert ts.reward.shape == (wrapper.num_agents,)
    assert ts.terminated.shape == (wrapper.num_agents,)
    assert ts.action_mask.shape[0] == wrapper.num_agents


def test_task_ids():
    wrapper, _ = _make_wrapper()
    key = jax.random.key(0)
    _, ts = wrapper.reset(key)

    assert ts.task_ids is not None
    assert ts.task_ids.shape == (wrapper.num_agents,)
    expected = jnp.array([0, 0, 1, 1])
    assert jnp.array_equal(ts.task_ids, expected)


def test_step_action_slicing():
    wrapper, _ = _make_wrapper()
    k1, k2 = jax.random.split(jax.random.key(0))

    states, _ = wrapper.reset(k1)
    actions = jnp.full((wrapper.num_agents,), MOVE_UP, dtype=jnp.int32)
    _, ts = wrapper.step(states, actions, k2)

    assert ts.obs.shape[0] == wrapper.num_agents
    assert ts.reward.shape == (wrapper.num_agents,)


def test_action_spec():
    wrapper, _ = _make_wrapper()
    assert wrapper.action_spec.n >= 5
