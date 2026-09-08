import jax
import pytest
from jax import numpy as jnp
from mapox.envs.rust_env import (
    RustEnv,
    RustFindReturnConfig,
    RustPacmanConfig,
    RustScoutsConfig,
    RustSnakeConfig,
)

CONFIGS = {
    "find_return": RustFindReturnConfig(num_agents=2, width=12, height=12),
    "scouts": RustScoutsConfig(num_scouts=1, num_harvesters=1, width=12, height=12),
    "snake": RustSnakeConfig(num_agents=2, width=12, height=12),
    "pacman": RustPacmanConfig(),
}


@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS)
def test_rust_bridge_uses_uint16_observations_and_actions(config):
    env = RustEnv(config, 128)

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
