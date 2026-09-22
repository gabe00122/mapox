import jax
import numpy as np
import pytest
from jax import numpy as jnp
from mapox.envs.rust_env_jax import RustEnvJax
from mapox.envs.rust_env_numpy import (
    RustEnvNumpy,
    RustFindReturnConfig,
    RustScoutsConfig,
    RustSnakeConfig,
)

CONFIGS = {
    "find_return": RustFindReturnConfig(num_agents=2, width=12, height=12),
    "scouts": RustScoutsConfig(num_scouts=1, num_harvesters=1, width=12, height=12),
    "snake": RustSnakeConfig(num_agents=2, width=12, height=12),
}


@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS)
def test_rust_bridge_uses_uint16_observations_and_actions(config):
    env = RustEnvJax(config, 128)

    _, reset_ts = env.reset(jax.random.key(0))
    assert reset_ts.obs.dtype == jnp.uint16
    assert reset_ts.last_action.dtype == jnp.uint16
    assert env.observation_spec.dtype == jnp.uint16
    assert env.action_spec.dtype == jnp.uint16

    actions = jnp.zeros((env.num_agents,), dtype=jnp.uint16)
    _, step_ts = env.step(None, actions, jax.random.key(1))
    assert step_ts.obs.dtype == jnp.uint16
    assert step_ts.last_action.dtype == jnp.uint16
    assert jnp.array_equal(step_ts.last_action, actions)


def test_numpy_env_steps_on_shared_buffers_without_jax():
    env = RustEnvNumpy(RustFindReturnConfig(num_agents=2, width=12, height=12), 128)

    reset_ts = env.reset(0)
    assert isinstance(reset_ts.obs, np.ndarray)

    actions = np.zeros((env.num_agents,), np.uint16)
    step_ts = env.step(actions)
    # the rust side writes one set of buffers in place; the timestep views them
    assert step_ts.obs is reset_ts.obs
    assert np.array_equal(step_ts.last_action, actions)
